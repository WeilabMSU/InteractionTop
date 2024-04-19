""" Summary: python code for the abstract simplicial complex interaction homology
    Currently, only consider the interaction of 1 pair simplicial complex

    Author:
        Dong Chen
    Create:
        2023-04-21
    Modify:
        2023-05-02
    Dependencies:
        python                    3.7.4
        numpy                     1.21.5
        scipy                     1.6.2
"""


import numpy as np
import itertools
from functools import wraps
import copy
import argparse
import sys
import time
from scipy.spatial import distance


def timeit(func):
    """ Timer """
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f"{'='*5} Function - {func.__name__} - took {total_time:.3f} seconds {'='*5}")
        return result
    return timeit_wrapper


class SimplicialComplexInteractionHomology(object):
    def __init__(self, eigenvalue_method='numpy_eigvalsh', eigvalue_num_limit=None):
        self.distance_matrix = None
        if eigenvalue_method == 'numpy_eigvalsh':
            self.eigvalue_calculator = np.linalg.eigvalsh

    def utils_powersets(self, nodes: list, max_dim: int = 2) -> dict:
        complete_edge_dict = {i: [] for i in range(max_dim)}

        max_len = min([len(nodes), max_dim])
        for i in range(max_len+1):
            complete_edge_dict[i] = list(itertools.combinations(nodes, i+1))
        return complete_edge_dict

    def max_adjacent_mat_to_complex(
        self,
        distance_matrix: np.array,
        max_adjacent_matrix: np.array,
        min_adjacent_matrix: np.array,
        max_threshold_dis: float,
        max_dim: int = 1,
        overlap_indices: list = None,
        add_idx: int = 0,
    ):
        adjacent_matrix = ((
            (distance_matrix <= max_threshold_dis) * max_adjacent_matrix + min_adjacent_matrix) > 0)

        n = adjacent_matrix.shape[0]
        simplicial_complex = {dim: [] for dim in range(max_dim+1)}
        simplicial_complex_form_distance = {dim: [] for dim in range(max_dim+1)}

        target_dim = min(max_dim, n)
        for k in range(0, target_dim+1):
            for S in itertools.combinations(range(n), k+1):
                if all(adjacent_matrix[i,j] for i in S for j in S if i < j):
                    max_local_distance = np.max([distance_matrix[d_i,d_j] for d_i in S for d_j in S])

                    if overlap_indices is not None:
                        S = tuple([s_ele if s_ele in overlap_indices else s_ele + add_idx for s_ele in S])
                    simplicial_complex[k].append(tuple(S))
                    simplicial_complex_form_distance[k].append(max_local_distance)
            local_distance_order = np.argsort(simplicial_complex_form_distance[k])
            simplicial_complex[k] = [simplicial_complex[k][idx] for idx in local_distance_order]
            simplicial_complex_form_distance[k] = [
                simplicial_complex_form_distance[k][idx] for idx in local_distance_order]

        return simplicial_complex, simplicial_complex_form_distance

    def max_complexes_to_interaction_complex(self, complex_1, complex_dis_1, complex_2, complex_dis_2, max_dim) -> tuple:
        interaction_complex = {dim: [] for dim in range(max_dim+1)}
        interaction_simplex_form_distance = {dim: [] for dim in range(max_dim+1)}

        for dim_key_1, simplex_list_1 in complex_1.items():
            for dim_key_2, simplex_list_2 in complex_2.items():
                interaction_dim = dim_key_1 + dim_key_2
                
                if interaction_dim not in interaction_complex:
                    continue
                for i1, simplex_1 in enumerate(simplex_list_1):
                    for i2, simplex_2 in enumerate(simplex_list_2):
                        if len(set(simplex_1).intersection(set(simplex_2))) > 0:
                            interaction_complex[interaction_dim].append(tuple([simplex_1, simplex_2]))
                            interaction_simplex_form_distance[interaction_dim].append(
                                np.max([complex_dis_1[dim_key_1][i1], complex_dis_2[dim_key_2][i2]])
                            )
        global_interaction_simplex_form_distance = []
        global_idx_corresponding_dim = []
        for dim, _ in interaction_complex.items():
            local_distance_order = np.argsort(interaction_simplex_form_distance[dim])
            interaction_complex[dim] = [interaction_complex[dim][idx] for idx in local_distance_order]
            global_interaction_simplex_form_distance += [
                interaction_simplex_form_distance[dim][idx] for idx in local_distance_order]
            global_idx_corresponding_dim += [dim]*len(local_distance_order)

        return interaction_complex, global_interaction_simplex_form_distance, global_idx_corresponding_dim

    def interaction_complex_to_boundary_matrix(self, interaction_complex: dict) -> dict:
        
        boundary_matrix_dict = {dim_n: None for dim_n in interaction_complex.keys()}
        for dim_n in sorted(interaction_complex.keys()):
            if dim_n == 0:
                boundary_matrix_dict[0] = np.zeros([1, len(interaction_complex[0])])
                continue

            interaction_simplex_dim_n = interaction_complex[dim_n]
            interaction_simplex_dim_n_minus_1 = interaction_complex[dim_n-1]
            dim_n_minus_1_index_dict = {sim: sim_idx for sim_idx, sim in enumerate(interaction_simplex_dim_n_minus_1)}

            if len(interaction_simplex_dim_n) == 0:
                break

            boundary_matrix_dict[dim_n] = np.zeros(
                [len(interaction_simplex_dim_n_minus_1), len(interaction_simplex_dim_n)])
            for idx_n, interaction_simplex in enumerate(interaction_simplex_dim_n):
                x = interaction_simplex[0]
                y = interaction_simplex[1]

                for omitted_n in range(len(x)):
                    omitted_simplex = tuple(np.delete(x, omitted_n))
                    x_y = tuple([omitted_simplex, y])
                    if x_y in interaction_simplex_dim_n_minus_1:
                        omitted_simplex_idx = dim_n_minus_1_index_dict[x_y]
                        boundary_matrix_dict[dim_n][omitted_simplex_idx, idx_n] += (-1)**omitted_n
                
                for omitted_n in range(len(y)):
                    omitted_simplex = tuple(np.delete(y, omitted_n))
                    x_y = tuple([x, omitted_simplex])
                    if x_y in interaction_simplex_dim_n_minus_1:
                        omitted_simplex_idx = dim_n_minus_1_index_dict[x_y]
                        boundary_matrix_dict[dim_n][omitted_simplex_idx, idx_n] += (-1)**(len(x)-1+omitted_n)

        self.has_boundary_max_dim = dim_n
        return boundary_matrix_dict

    @staticmethod
    def matrix_reduction_column(boundary_matrix: np.matrix) -> tuple:
        '''boundary_matrix: binary matrix, elements are either 1 or 0
            pseudocode:
                for j=1 to n:
                    while exist j0 < j with pivot(j0) == pivot(j)
                        add column j0 to column j
                    end
                end
        '''
        boundary_matrix = boundary_matrix %2
        _, col_num = boundary_matrix.shape

        pivot_pairs_dict = {}
        for j in range(col_num):
            if np.sum(boundary_matrix[:, j]) == 0:
                continue
            else:
                while (np.sum(boundary_matrix[:, j]) > 0) and (np.nonzero(boundary_matrix[:, j])[0][-1] in pivot_pairs_dict):
                    pivot_row_idx = np.nonzero(boundary_matrix[:, j])[0][-1]
                    boundary_matrix[:, j] = (boundary_matrix[:, j] + boundary_matrix[:, pivot_pairs_dict[pivot_row_idx]]) % 2
                
                if np.sum(boundary_matrix[:, j]) == 0:
                    continue
                else:
                    pivot_pairs_dict[np.nonzero(boundary_matrix[:, j])[0][-1]] = j

        return boundary_matrix, pivot_pairs_dict

    @staticmethod
    def reorder_input_data(cloudpoints_1, cloudpoints_2):
        pair_distance_matrix = distance.cdist(cloudpoints_1, cloudpoints_2)
        zero_position = np.where(pair_distance_matrix < 1e-5)
        cloudpoints_1_order = np.append(
            zero_position[0],
            list(set(np.arange(len(cloudpoints_1), dtype=int)).difference(set(zero_position[0])))
        )
        cloudpoints_2_order = np.append(
            zero_position[1],
            list(set(np.arange(len(cloudpoints_2), dtype=int)).difference(set(zero_position[1])))
        )

        overlap_indices = np.arange(len(zero_position[0]), dtype=int)
        return cloudpoints_1[cloudpoints_1_order], cloudpoints_2[cloudpoints_2_order], overlap_indices

    def persistent_interaction_homology(
        self, input_data_1: np.array = None, max_adjacent_matrix_1: np.array = None,
        min_adjacent_matrix_1: np.array = None, max_dim_1: int = 1, is_distance_matrix_1: bool = False, 
        input_data_2: np.array = None, max_adjacent_matrix_2: np.array = None,
        min_adjacent_matrix_2: np.array = None, max_dim_2: int = 1, is_distance_matrix_2: bool = False,
        interaction_max_dim: int = 1, filtration: np.array = None,
        cutoff_distance: float = None, step_dis: float = None,
        overlap_indices: list = None,
    ):
        """
            If the inputs are distance matrices, the overlap_indices should be set.
        """
        if is_distance_matrix_1:
            distance_matrix_1 = input_data_1
            points_num_1 = distance_matrix_1.shape[0]
        else:
            cloudpoints_1 = input_data_1
            points_num_1 = cloudpoints_1.shape[0]
            distance_matrix_1 = distance.cdist(cloudpoints_1, cloudpoints_1)

        if max_adjacent_matrix_1 is None:
            max_adjacent_matrix_1 = np.ones([points_num_1, points_num_1], dtype=int)
            np.fill_diagonal(max_adjacent_matrix_1, 0)
        
        if min_adjacent_matrix_1 is None:
            min_adjacent_matrix_1 = np.zeros([points_num_1, points_num_1], dtype=int)

        if is_distance_matrix_2:
            distance_matrix_2 = input_data_2
            points_num_2 = distance_matrix_2.shape[0]
        else:
            cloudpoints_2 = input_data_2
            points_num_2 = cloudpoints_2.shape[0]
            distance_matrix_2 = distance.cdist(cloudpoints_2, cloudpoints_2)

        if max_adjacent_matrix_2 is None:
            max_adjacent_matrix_2 = np.ones([points_num_2, points_num_2], dtype=int)
            np.fill_diagonal(max_adjacent_matrix_2, 0)
        
        if min_adjacent_matrix_2 is None:
            min_adjacent_matrix_2 = np.zeros([points_num_2, points_num_2], dtype=int)

        if filtration is None:
            filtration = np.arange(0, cutoff_distance, step_dis)
        max_threshold_dis_1 = np.max(filtration)
        max_threshold_dis_2 = np.max(filtration)
        
        if points_num_1 < points_num_2:
            min_adjacent_matrix_1, min_adjacent_matrix_2 = min_adjacent_matrix_2, min_adjacent_matrix_1
            man_adjacent_matrix_1, man_adjacent_matrix_2 = man_adjacent_matrix_2, man_adjacent_matrix_1
            distance_matrix_1, distance_matrix_2 = distance_matrix_2, distance_matrix_1
        add_idx = np.max([points_num_1, points_num_2])

        complex_1, complex_dis_1 = self.max_adjacent_mat_to_complex(
            distance_matrix_1, max_adjacent_matrix_1, min_adjacent_matrix_1, max_threshold_dis_1, max_dim_1+1)
        complex_2, complex_dis_2 = self.max_adjacent_mat_to_complex(
            distance_matrix_2, max_adjacent_matrix_2, min_adjacent_matrix_2, max_threshold_dis_2, max_dim_2+1, overlap_indices, add_idx)
        interaction_complex, global_interaction_simplex_form_distance, global_idx_corresponding_dim = self.max_complexes_to_interaction_complex(
            complex_1, complex_dis_1, complex_2, complex_dis_2, interaction_max_dim+1)
        print(interaction_complex)
        boundary_matrix_dict = self.interaction_complex_to_boundary_matrix(interaction_complex)

        global_idx_c = boundary_matrix_dict[0].shape[-1]
        global_idx_r = 0
        global_pivot_pair_dict = {}
        global_pivot_row_idx_dict = {}
        global_pivot_col_idx_dict = {}
        for dim in range(1, interaction_max_dim+1+1):
            if boundary_matrix_dict[dim] is None:
                continue
            reduced_boundary_matrix, pivot_pairs_dict = self.matrix_reduction_column(boundary_matrix_dict[dim])
            rows, cols = reduced_boundary_matrix.shape
            for local_row_idx, local_col_idx in pivot_pairs_dict.items():
                global_pivot_pair_dict[global_idx_r+local_row_idx] = global_idx_c+local_col_idx
                global_pivot_row_idx_dict[global_idx_r+local_row_idx] = 1
                global_pivot_col_idx_dict[global_idx_c+local_col_idx] = 1
            global_idx_c += cols
            global_idx_r += rows

        persistent_barcode = {dim: [] for dim in range(interaction_max_dim+1)}
        for col_idx, dim in enumerate(global_idx_corresponding_dim):
            if (col_idx not in global_pivot_col_idx_dict) and (dim <= interaction_max_dim):
                birth_idx = col_idx
                birth_distance = global_interaction_simplex_form_distance[col_idx]

                if (birth_idx in global_pivot_row_idx_dict):
                    death_idx = global_pivot_pair_dict[col_idx]
                    death_distance = global_interaction_simplex_form_distance[death_idx]
                    if np.abs(death_distance - birth_distance) < 1e-10:
                        continue
                    persistent_barcode[dim].append([birth_distance, death_distance])
                else:
                    persistent_barcode[dim].append([birth_distance, np.inf])

        return persistent_barcode

    # ##############################

    def adjacent_mat_to_simplex(self, adjacent_matrix: np.array, max_dim: int = 1, overlap_indices: list = None, add_idx: int = 0) -> dict:
        """
            Given an adjacency matrix A for an undirected graph, construct the clique complex of the graph.
        """
        n = adjacent_matrix.shape[0] 
        simplicial_complex = {dim: [] for dim in range(max_dim+1)}  

        target_dim = min(max_dim, n)
        for k in range(0, target_dim+1):
            for S in itertools.combinations(range(n), k+1):
                if all(adjacent_matrix[i,j] for i in S for j in S if i < j):
                    if overlap_indices is not None:
                        S = tuple([s_ele if s_ele in overlap_indices else s_ele + add_idx for s_ele in S])
                    simplicial_complex[k].append(tuple(S))
        return simplicial_complex
    
    def complexes_to_interaction_complex(self, complex_1: dict, complex_2: dict, max_dim: int = 1) -> dict:
        
        interaction_complex = {dim: [] for dim in range(max_dim+1)}

        for dim_key_1, simplex_list_1 in complex_1.items():
            for dim_key_2, simplex_list_2 in complex_2.items():
                interaction_dim = dim_key_1 + dim_key_2
                if interaction_dim not in interaction_complex:
                    continue
                for simplex_1 in simplex_list_1:
                    for simplex_2 in simplex_list_2:
                        if len(set(simplex_1).intersection(set(simplex_2))) > 0:
                            interaction_complex[interaction_dim].append(tuple([simplex_1, simplex_2]))

        return interaction_complex

    def interaction_homology_from_complexes(self, complex_1: dict, complex_2: dict, max_dim: int = 1)-> np.array:
        self.max_dim = max_dim
        self.max_boundary_dim = max_dim + 1
        self.interaction_complex = self.complexes_to_interaction_complex(complex_1, complex_2, self.max_boundary_dim)
        boundary_matrix_dict = self.interaction_complex_to_boundary_matrix(self.interaction_complex)

        betti_numbers = []
        for dim_n in range(max_dim+1):
            dn = boundary_matrix_dict[dim_n]
            dn1 = boundary_matrix_dict[dim_n+1]
            dim_dn = 0 if dn is None else dn.shape[-1]
            rank_dn = 0 if dn is None else np.linalg.matrix_rank(dn)
            rank_dn1 = 0 if dn1 is None else np.linalg.matrix_rank(dn1) 
            betti_numbers.append(dim_dn - rank_dn - rank_dn1)
        return betti_numbers

    def interaction_homology_from_adjacent_matrix(
        self, adjacent_matrix_1: np.array,
        adjacent_matrix_2: np.array,
        max_dim: int = 1, overlap_indices: list = None,
    ) -> np.array:
        self.max_dim = max_dim
        self.max_boundary_dim = max_dim + 1
        data_n1 = adjacent_matrix_1.shape[-1]
        data_n2 = adjacent_matrix_2.shape[-1]
        if data_n1 < data_n2:
            adjacent_matrix_1, adjacent_matrix_2 = adjacent_matrix_2, adjacent_matrix_1
        add_idx = np.max([data_n1, data_n2])

        complex_1 = self.adjacent_mat_to_simplex(adjacent_matrix_1, self.max_boundary_dim)  
        complex_2 = self.adjacent_mat_to_simplex(adjacent_matrix_2, self.max_boundary_dim, overlap_indices, add_idx) 

        self.complex_1 = complex_1
        self.complex_2 = complex_2
        self.interaction_complex = self.complexes_to_interaction_complex(complex_1, complex_2, self.max_boundary_dim)
        boundary_matrix_dict = self.interaction_complex_to_boundary_matrix(self.interaction_complex)

        betti_numbers = []
        for dim_n in range(max_dim+1):
            dn = boundary_matrix_dict[dim_n]
            dn1 = boundary_matrix_dict[dim_n+1]
            dim_dn = 0 if dn is None else dn.shape[-1]
            rank_dn = 0 if dn is None else np.linalg.matrix_rank(dn)
            rank_dn1 = 0 if dn1 is None else np.linalg.matrix_rank(dn1) 
            betti_numbers.append(dim_dn - rank_dn - rank_dn1)
        return betti_numbers

    def boundary_to_laplacian_matrix(self, boundary_matrix_dict: dict) -> dict:
        laplacian_matrix_dict = {}
        for dim_n in sorted(boundary_matrix_dict.keys()):
            boundary_matrix = boundary_matrix_dict[dim_n]
            if dim_n >= self.has_boundary_max_dim:
                break
            elif dim_n == 0 and boundary_matrix_dict[dim_n+1] is not None:
                laplacian_matrix_dict[dim_n] = np.dot(boundary_matrix_dict[dim_n+1], boundary_matrix_dict[dim_n+1].T)
            elif dim_n == 0 and boundary_matrix_dict[dim_n+1] is None:
                laplacian_matrix_dict[dim_n] = np.zeros([boundary_matrix_dict[0].shape[1]]*2)
            elif dim_n > 0 and boundary_matrix_dict[dim_n+1] is None:
                laplacian_matrix_dict[dim_n] = np.dot(boundary_matrix.T, boundary_matrix)
                break
            else:
                laplacian_matrix_dict[dim_n] = np.dot(
                    boundary_matrix_dict[dim_n+1], boundary_matrix_dict[dim_n+1].T) + np.dot(boundary_matrix.T, boundary_matrix)
        return laplacian_matrix_dict

    def interaction_laplacian_from_adjacent_mat(
        self, adjacent_matrix_1: np.array,
        adjacent_matrix_2: np.array, max_dim: int = 1
    ) -> np.array:
        self.max_dim = max_dim
        self.max_boundary_dim = max_dim + 1
        complex_1 = self.adjacent_mat_to_simplex(adjacent_matrix_1, self.max_boundary_dim)
        complex_2 = self.adjacent_mat_to_simplex(adjacent_matrix_2, self.max_boundary_dim)
        interaction_complex = self.complexes_to_interaction_complex(complex_1, complex_2, self.max_boundary_dim)
        boundary_matrix_dict = self.interaction_complex_to_boundary_matrix(interaction_complex)
        laplacian_matrix_dict = self.boundary_to_laplacian_matrix(boundary_matrix_dict)

        laplacian_eigenv = {}
        for dim_n in range(self.max_dim+1):
            if dim_n in laplacian_matrix_dict:
                laplacian_matrix = laplacian_matrix_dict[dim_n]
                eig_value = self.eigvalue_calculator(laplacian_matrix)
                eig_value = eig_value.real
                laplacian_eigenv[dim_n] = sorted(np.round(eig_value, 5))
            else:
                laplacian_eigenv[dim_n] = None
        return laplacian_eigenv

    def interaction_laplacian_from_complexes(
        self, complex_1: dict, complex_2: dict, max_dim: int = 1
    ) -> np.array:
        self.max_dim = max_dim
        self.max_boundary_dim = max_dim + 1
        interaction_complex = self.complexes_to_interaction_complex(complex_1, complex_2, self.max_boundary_dim)
        boundary_matrix_dict = self.interaction_complex_to_boundary_matrix(interaction_complex)
        laplacian_matrix_dict = self.boundary_to_laplacian_matrix(boundary_matrix_dict)

        laplacian_eigenv = {}
        for dim_n in range(self.max_dim+1):
            if dim_n in laplacian_matrix_dict:
                laplacian_matrix = laplacian_matrix_dict[dim_n]
                eig_value = self.eigvalue_calculator(laplacian_matrix)
                eig_value = eig_value.real
                laplacian_eigenv[dim_n] = sorted(np.round(eig_value, 5))
            else:
                laplacian_eigenv[dim_n] = None
        return laplacian_eigenv


def main():
    input_data_1 = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    input_data_2 = np.array([[1, 0], [1, 1], [2, 0]])
    interaction_max_dim = max_dim_1 = max_dim_2 = 1
    filtration = np.array([0, 1, 1.1, 1.2, np.sqrt(2), 1.5])
    pih = SimplicialComplexInteractionHomology()
    input_data_1, input_data_2, overlap_indices = pih.reorder_input_data(input_data_1, input_data_2)
    print(input_data_1)
    print(input_data_2)
    print(overlap_indices)
    persistent_barcode = pih.persistent_interaction_homology(
        input_data_1=input_data_1, input_data_2=input_data_2,
        max_dim_1=max_dim_1, max_dim_2=max_dim_2,
        interaction_max_dim=interaction_max_dim, filtration=filtration,
        overlap_indices=overlap_indices,
    )

    for k, v in persistent_barcode.items():
        print(f'dim: {k}', np.round(v, 3))
    return None


if __name__ == "__main__":
    main()
