import numpy as np

import numba
from numba import cuda
#from pyculib import rand


@numba.jit(nopython=True)
def permute_array_cpu(array, index_array):
    tmp_len = len(array)
    for i in range(tmp_len):
        index_array[i] = index_array[i] * tmp_len
        array[i], array[int(index_array[i])] = \
            array[int(index_array[i])], array[i]

@cuda.jit
def permute_array_gpu(array, index_array):
    tmp_len = len(array)
    for i in range(tmp_len):
        index_array[i] = index_array[i] * tmp_len
        array[i], array[int(index_array[i])] = \
            array[int(index_array[i])], array[i]


def create_permutation_matrix(y_bin_vars, n_reps=0):
    # Validate n_reps parameter
    assert n_reps >= 0, "n_reps value must be greater than or equal to 0!"

    if isinstance(y_bin_vars['bins'], np.ndarray):
        # execute on cpu

        # create a matrix that is going to hold the
        # original 'bins' series as well as all
        # the permutations. Flipping row and column
        # representation for easier execution
        mcpt_matrix = np.tile(y_bin_vars['bins'], (n_reps+1, 1))

        # copy 'bins' into first column of mcpt_matrix
        # and permute 'bins' n_reps number of times for
        # all other columns in mcpt_matrix
        for i in range(mcpt_matrix.shape[0]):
            if i > 0:
                rand_array = np.random.uniform(size=mcpt_matrix.shape[1])
                permute_array_cpu(mcpt_matrix[i], rand_array)
    else:
        # execute on gpu
        # copy y_bin_vars['bins'] back to the cpu in order to create
        # the mcpt_matrix
        bins = y_bin_vars['bins'].copy_to_host()
        # create the mcpt_matrix and copy back to gpu
        mcpt_matrix = np.tile(bins, (n_reps+1, 1))
        mcpt_matrix_gpu = cuda.to_device(mcpt_matrix)
        for i in range(mcpt_matrix_gpu.shape[0]):
            if i > 0:
                rand_array = \
                    rand.uniform(size=mcpt_matrix_gpu.shape[1], device=True)
                permute_array_gpu(mcpt_matrix_gpu[i], rand_array)
        mcpt_matrix = mcpt_matrix_gpu

    return mcpt_matrix
