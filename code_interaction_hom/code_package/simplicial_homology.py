""" Summary: python code for the abstract simplicial complex interaction homology
    Currently, only consider the interaction of 1 pair simplicial complex

    Author:
        Dong Chen
    Create:
        2023-04-21
    Modify:
        2023-04-21
    Dependencies:
        python                    3.7.4
        numpy                     1.21.5
"""


import numpy as np
import itertools
from functools import wraps
import copy
import argparse
import sys
import time
from functools import wraps


def timeit(func):
    """ Timer """
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        # print(f"{'='*5} Function {func.__name__}{args} {kwargs} Took {total_time:.3f} seconds {'='*5}")
        print(f"{'='*5} Function - {func.__name__} - took {total_time:.3f} seconds {'='*5}")
        return result
    return timeit_wrapper


class statistic_eigvalues(object):
    '''Input is 1-D array'''
    def __init__(self, eigvalues: np.array) -> None:
        digital = 5
        values = np.round(eigvalues, 5)
        self.all_values = sorted(values)
        self.nonzero_values = values[np.nonzero(values)]
        self.count_zero = len(values) - np.count_nonzero(values)
        self.max = np.max(values)
        self.sum = np.round(np.sum(values), digital)

        if len(self.nonzero_values) > 0:
            self.nonzero_mean = np.round(np.mean(self.nonzero_values), digital)
            self.nonzero_std = np.round(np.std(self.nonzero_values), digital)
            self.nonzero_min = np.round(np.min(self.nonzero_values), digital)
            self.nonzero_var = np.round(np.var(self.nonzero_values), digital)
        else:
            # if none nonzero min, set it as 0
            self.nonzero_mean = 0
            self.nonzero_std = 0
            self.nonzero_min = 0
            self.nonzero_var = 0


def mod2_addition(a: int, b: int) -> int:
    return (a+b) % 2


def matrix_reduction_column(boundary_matrix: np.matrix) -> tuple:
    '''boundary_matrix: binary matrix, elements are either 1 or 0
        pseudocode:
            for j=1 to n:
                while exist j0 < j with pivot(j0) == pivot(j)
                    add column j0 to column j
                end
            end
    '''
    row_num, col_num = boundary_matrix.shape

    pivot_col_indices = {} # row_idx: col_idx
    for j in range(col_num):
        if np.sum(boundary_matrix[:, j]) == 0:
            continue
        else:
            while (np.sum(boundary_matrix[:, j]) > 0) and (np.nonzero(boundary_matrix[:, j])[0][-1] in pivot_col_indices):
                pivot_row_idx = np.nonzero(boundary_matrix[:, j])[0][-1]
                boundary_matrix[:, j] = (boundary_matrix[:, j] + boundary_matrix[:, pivot_col_indices[pivot_row_idx]]) % 2
            
            if np.sum(boundary_matrix[:, j]) == 0:
                continue
            else:
                pivot_col_indices[np.nonzero(boundary_matrix[:, j])[0][-1]] = j
    return boundary_matrix


class SimplicialHomology(object):
    def __init__(self, eigenvalue_method='numpy_eig', eigvalue_num_limit=None):
        self.distance_matrix = None
        if eigenvalue_method == 'numpy_eig':
            self.eigvalue_calculator = np.linalg.eig

    def utils_powersets(self, nodes: list, max_dim: int = 2) -> dict:
        complete_edge_dict = {i: [] for i in range(max_dim)}

        max_len = min([len(nodes), max_dim])
        for i in range(max_len+1):
            complete_edge_dict[i] = list(itertools.combinations(nodes, i+1))
        return complete_edge_dict

    def max_adjacency_mat_to_simplex(
        self,
        distance_matrix: np.array,
        max_adjacency_matrix: np.array,
        min_adjacency_matrix: np.array,
        max_threshold_dis: float,
        max_dim: int = 1
    ):
        adjacency_matrix = ((
            (distance_matrix <= max_threshold_dis) * max_adjacency_matrix + min_adjacency_matrix) > 0)

        # Number of nodes in the graph
        n = adjacency_matrix.shape[0]
        # List of simplices in the clique complex
        simplicial_complex = {dim: [] for dim in range(max_dim+1)}
        # List of forming distance simplices in the clique complex
        simplicial_complex_form_distance = {dim: [] for dim in range(max_dim+1)}

        # Add the 0-simplices (nodes)
        simplicial_complex[0] = [(i, ) for i in range(n)]
        simplicial_complex_form_distance[0] = [0 for i in range(n)]

        # Add higher-dimensional simplices corresponding to cliques of size > 1
        target_dim = min(max_dim, n)
        for k in range(1, target_dim+1):
            for S in itertools.combinations(range(n), k+1):
                if all(adjacency_matrix[i,j] for i in S for j in S if i < j):
                    simplicial_complex[k].append(tuple(S))
                    max_local_distance = np.round(np.max([distance_matrix[d_i,d_j] for d_i in S for d_j in S]), 5)
                    simplicial_complex_form_distance[k].append(max_local_distance)
            local_distance_order = np.argsort(simplicial_complex_form_distance[k])
            simplicial_complex[k] = [simplicial_complex[k][idx] for idx in local_distance_order]
            simplicial_complex_form_distance[k] = [
                simplicial_complex_form_distance[k][idx] for idx in local_distance_order]

        return simplicial_complex, simplicial_complex_form_distance

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
        row_num, col_num = boundary_matrix.shape

        pivot_col_indices = {} # row_idx: col_idx
        for j in range(col_num):
            if np.sum(boundary_matrix[:, j]) == 0:
                continue
            else:
                while (np.sum(boundary_matrix[:, j]) > 0) and (np.nonzero(boundary_matrix[:, j])[0][-1] in pivot_col_indices):
                    pivot_row_idx = np.nonzero(boundary_matrix[:, j])[0][-1]
                    boundary_matrix[:, j] = (boundary_matrix[:, j] + boundary_matrix[:, pivot_col_indices[pivot_row_idx]]) % 2
                
                if np.sum(boundary_matrix[:, j]) == 0:
                    continue
                else:
                    pivot_col_indices[np.nonzero(boundary_matrix[:, j])[0][-1]] = j
        return boundary_matrix

    # ##############################

    def adjacency_mat_to_simplex(self, adjacency_matrix: np.array, max_dim: int = 1) -> dict:
        """
            Given an adjacency matrix A for an undirected graph, construct the clique complex of the graph.
        """
        n = adjacency_matrix.shape[0]  # Number of nodes in the graph
        simplicial_complex = {dim: [] for dim in range(max_dim+1)}  # List of simplices in the clique complex

        # Add the 0-simplices (nodes)
        simplicial_complex[0] = [(i, ) for i in range(n)]

        # Add higher-dimensional simplices corresponding to cliques of size > 1
        target_dim = min(max_dim, n)
        for k in range(1, target_dim+1):
            for S in itertools.combinations(range(n), k+1):
                if all(adjacency_matrix[i,j] for i in S for j in S if i < j):
                    simplicial_complex[k].append(tuple(S))
        return simplicial_complex

    def complex_to_boundary_matrix(self, complex: dict,) -> dict:
        
        # initial
        boundary_matrix_dict = {dim_n: None for dim_n in complex.keys()}
        for dim_n in sorted(complex.keys()):
            # for dim_0, boundary matrix shape [len(node), 1]
            if dim_n == 0:
                # boundary_matrix_dict[0] = np.zeros([len(complex[0]), 1])
                boundary_matrix_dict[0] = np.zeros([1, len(complex[0])])
                continue

            # for dim >= 1
            simplex_dim_n = complex[dim_n]
            simplex_dim_n_minus_1 = complex[dim_n-1]

            if len(simplex_dim_n) == 0:
                break

            # boundary rows: simplex_{n-1} cols: simplex_{n}
            boundary_matrix_dict[dim_n] = np.zeros(
                [len(simplex_dim_n_minus_1), len(simplex_dim_n)])
            for idx_n, simplex in enumerate(simplex_dim_n):
                for omitted_n in range(len(simplex)):
                    omitted_simplex = tuple(np.delete(simplex, omitted_n))
                    omitted_simplex_idx = simplex_dim_n_minus_1.index(omitted_simplex)
                    boundary_matrix_dict[dim_n][omitted_simplex_idx, idx_n] += 1

        self.has_boundary_max_dim = dim_n
        return boundary_matrix_dict

    def persistent_homology(
        self, input_data: np.array = None,
        max_adjacency_matrix: np.array = None, min_adjacency_matrix: np.array = None,
        is_distance_matrix: bool = False, max_dim: int = 1, filtration: np.array = None,
        cutoff_distance: float = None, step_dis: float = None, print_by_step: bool = True,
    ) -> np.array:
        # the default data is cloudpoints
        if is_distance_matrix:
            distance_matrix = input_data
            points_num = distance_matrix.shape[0]
        else:
            cloudpoints = input_data
            points_num = cloudpoints.shape[0]
            distance_matrix = np.zeros([points_num, points_num], dtype=float)
            for i in range(points_num):
                for j in range(i+1, points_num):
                    # distance function here
                    distance = np.linalg.norm(cloudpoints[i, :] - cloudpoints[j, :])
                    distance_matrix[i, j] = distance
                    distance_matrix[j, i] = distance

        if max_adjacency_matrix is None:
            max_adjacency_matrix = np.ones([points_num, points_num], dtype=int)
            np.fill_diagonal(max_adjacency_matrix, 0)
        
        if min_adjacency_matrix is None:
            min_adjacency_matrix = np.zeros([points_num, points_num], dtype=int)

        if filtration is None:
            filtration = np.arange(0, cutoff_distance, step_dis)
        
        simplicial_complex, simplicial_complex_form_distance = self.max_adjacency_mat_to_simplex(
            distance_matrix=distance_matrix,
            max_adjacency_matrix=max_adjacency_matrix,
            min_adjacency_matrix=min_adjacency_matrix,
            max_threshold_dis=np.max(filtration),
            max_dim=max_dim,
        )
        boundary_matrix_dict = self.complex_to_boundary_matrix(simplicial_complex)

        # persistent barcodes
        persistent_barcode = {dim: [] for dim in range(max_dim+1)}
        persistent_barcode[0].append([0, np.inf])
        for dim in range(1, max_dim+1):
            print(self.has_boundary_max_dim)
            reduced_boundary_matrix = matrix_reduction_column(boundary_matrix_dict[dim])
            rows, cols = reduced_boundary_matrix.shape
            for c in range(cols):
                for r in range(rows-1, 0-1, -1):
                    if (reduced_boundary_matrix[r, c] != 0):
                        if np.abs(simplicial_complex_form_distance[dim][c] - 
                            simplicial_complex_form_distance[dim-1][r]) < 1e-5:
                            break
                        persistent_barcode[dim-1].append(
                            [
                                simplicial_complex_form_distance[dim-1][r],
                                simplicial_complex_form_distance[dim][c],
                            ]
                        )
                        break

        return persistent_barcode


def main():
    aa = SimplicialHomology()
    adjacency_matrix = np.array([
        [0, 1, 0, 1, 0],
        [1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [1, 0, 1, 0, 1],
        [0, 0, 0, 1, 0],
    ])
    adjacency_matrix = np.array([
        [0, 1, 1, 0],
        [1, 0, 1, 0],
        [1, 1, 0, 1],
        [0, 0, 1, 0],
    ])
    # adjacency_matrix = np.array([
    #     [0, 1, 0, 0, 1, 0],
    #     [0, 0, 1, 0, 0, 0],
    #     [0, 0, 0, 1, 0, 0],
    #     [0, 0, 0, 0, 0, 0],
    #     [0, 0, 1, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0],
    # ])
    ww = aa.adjacency_mat_to_simplex(adjacency_matrix, max_dim=2)
    print(ww)
    # print(aa.neighborhood_complex)
    feat = aa.interaction_laplacian_from_connected_mat(adjacency_matrix, max_dim=2)
    print(feat)
    return None


if __name__ == "__main__":
    main()
