import numba
from numba import cuda

import numpy as np

import math

#############################################
## Univariate mutual information functions ##
#############################################
@numba.jit(nopython=True)
def populate_marginals_grid(bins_x_array, bins_y_array, 
                            marginal_x_array, marginal_y_array,
                            grid_array):
    
    assert(len(bins_x_array) == len(bins_y_array))
    
    ncases = len(bins_x_array)
    
    nbins_x = len(marginal_x_array)
    nbins_y = len(marginal_y_array)

    # Populate the marginal array and grid matrix
    for i in range(ncases):
        ix = bins_x_array[i]
        iy = bins_y_array[i]
        # going to explicitly cast 'ix' and 'iy' to integers
        marginal_x_array[np.int64(ix)] += 1
        marginal_y_array[np.int64(iy)] += 1
        grid_array[np.int64(ix*nbins_y + iy)] += 1
        
        
@numba.jit(nopython=True, parallel=True)
def ur_calc(ur_array, bins_x, bins_y_matrix, marginal_x, marginal_y, grid):

    for i in numba.prange(bins_y_matrix.shape[0]):
        
        populate_marginals_grid(bins_x, bins_y_matrix[i],
                                marginal_x[i], marginal_y[i], grid[i])

        nbins_x = marginal_x[i].shape[0]
        nbins_y = marginal_y[i].shape[0]

        # Test for single bin row or column
        if nbins_x < 2 or nbins_y < 2:
            # Assign value of 0
            UR = 0.0
        else:                       
            # calculate number of cases
            ncases = 0
            for j in range(nbins_x):
                ncases += marginal_x[i,j]

            UR = ur_array[i]
            entropy_x = 0.0
            entropy_y = 0.0
            entropy_joint = 0.0
                    
            for j in range(nbins_x):
                px = marginal_x[i,j]/ncases
                entropy_x -= px * math.log(px)

            for k in range(nbins_y):
                py = marginal_y[i,k]/ncases
                entropy_y -= py * math.log(py)

            for j in range(nbins_x):
                for k in range(nbins_y):
                    pxy = grid[i,j*nbins_y + k]/ncases
                    entropy_joint -= pxy * math.log(pxy)
    
            if entropy_y > 0:
                UR = (entropy_x + entropy_y - entropy_joint) / entropy_y
            else:
                UR = 0.0
        
        ur_array[i] = UR

        
@numba.jit(nopython=True, parallel=True)
def ur_calc_parallel(ur_matrix, bins_x, bins_y, 
                     marginal_x, marginal_y, grid):
    
    for var in numba.prange(bins_x.shape[0]):
    
        for i in numba.prange(bins_y.shape[0]):

            populate_marginals_grid(bins_x[var], bins_y[i], 
                                    marginal_x[var,i], marginal_y[var,i],
                                    grid[var,i])

            nbins_x = marginal_x[var,i].shape[0]
            nbins_y = marginal_y[var,i].shape[0]

            # calculate number of cases first
            ncases = 0
            for j in range(nbins_x):
                ncases += marginal_x[var,i,j]

            # Test for single bin row or column
            if nbins_x < 2 or nbins_y < 2:
                # Assign value of 0
                UR = 0.0
            else:    
                UR = ur_matrix[var,i]
                entropy_x = 0.0
                entropy_y = 0.0
                entropy_joint = 0.0

                for j in range(nbins_x):
                    px = marginal_x[var,i,j]/ncases
                    entropy_x -= px * math.log(px)
                    
                for k in range(nbins_y):
                    py = marginal_y[var,i,k]/ncases
                    entropy_y -= py * math.log(py)
                    
                for j in range(nbins_x):
                    for k in range(nbins_y):
                        pxy = grid[var,i,j*nbins_y + k]/ncases
                        entropy_joint -= pxy * math.log(pxy)

                if entropy_y > 0:
                    UR = (entropy_x + entropy_y - entropy_joint) / entropy_y
                else:
                    UR = 0.0
            
                ur_matrix[var,i] = UR
        
        
@numba.jit(nopython=True)
def mutinf_discrete_calc(bins_x, bins_y):
    
    nbins_x = np.unique(bins_x).shape[0]
    nbins_y = np.unique(bins_y).shape[0]
    marginal_x = np.zeros(nbins_x, np.int64)
    marginal_y = np.zeros(nbins_y, np.int64)
    grid = np.zeros(nbins_x * nbins_y, np.int64)
    
    populate_marginals_grid(bins_x, bins_y, 
                            marginal_x, marginal_y, grid)

    # calculate number of cases by summing either of the
    # marginal arrays
    ncases = 0
    for i in range(nbins_x):
        ncases += marginal_x[i]

    MI = 0.0
            
    for i in range(nbins_x):
        px = marginal_x[i]/ncases
        for j in range(nbins_y):
            py = marginal_y[j]/ncases
            pxy = grid[i*nbins_y + j]/ncases
            if pxy > 0:
                MI += pxy * math.log(pxy / (px*py))
        
    return MI

    
@numba.jit(nopython=True, parallel=True)
def mutinf_discrete_calc_parallel(mi_matrix, bins_x, bins_y, 
                                  marginal_x, marginal_y, grid):
    
    for var in numba.prange(bins_x.shape[0]):
    
        for i in numba.prange(bins_y.shape[0]):

            populate_marginals_grid(bins_x[var], bins_y[i], 
                                    marginal_x[var,i], marginal_y[var,i],
                                    grid[var,i])

            nbins_x = marginal_x[var,i].shape[0]
            nbins_y = marginal_y[var,i].shape[0]

            # calculate number of cases first
            ncases = 0
            for j in range(nbins_x):
                ncases += marginal_x[var,i,j]

            MI = mi_matrix[var,i]

            for j in range(nbins_x):
                px = marginal_x[var,i,j]/ncases
                for k in range(nbins_y):
                    py = marginal_y[var,i,k]/ncases
                    pxy = grid[var,i,j*nbins_y + k]/ncases
                    if pxy > 0:
                        MI += pxy * math.log(pxy / (px*py))

            mi_matrix[var,i] = MI
    

@cuda.jit
def mutinf_discrete_calc_gpu(size, mi_array, bins_x, bins_y_matrix, 
                             marginal_x, marginal_y, grid):
    
    i = cuda.grid(1)
    
    if i < size:
    
        populate_marginals_grid(bins_x, bins_y_matrix[i],
                                marginal_x[i], marginal_y[i], grid[i])

        nbins_x = len(marginal_x[i])
        nbins_y = len(marginal_y[i])

        # calculate number of cases by summing either of the
        # marginal arrays
        ncases = 0
        for j in range(nbins_x):
            ncases += marginal_x[i,j]

        MI = mi_array[i]

        for j in range(nbins_x):
            px = marginal_x[i,j]/ncases
            for k in range(nbins_y):
                py = marginal_y[i,k]/ncases
                pxy = grid[i,j*nbins_y + k]/ncases
                if pxy > 0:
                    MI += pxy * math.log(pxy / (px*py))

        mi_array[i] = MI

        
############################################
## Bivariate mutual information functions ##
############################################
@numba.jit(nopython=True, nogil=True)
def populate_bivariate_marginals_grid(bins_x1_array, bins_x2_array,
                                      bins_y_array,
                                      nbins_x1, nbins_x2,
                                      marginal_x1x2_array,
                                      marginal_y_array,
                                      grid_array):
    
    # make sure that all the 'bin' arrays have the 
    # same number of observations
    assert(len(bins_x1_array) == len(bins_x2_array))
    assert(len(bins_x1_array) == len(bins_y_array))
    
    ncases = len(bins_x1_array)
    
    nbins_y = len(marginal_y_array)

    # Populate the marginal array and grid matrix
    for i in range(ncases):
        ix1 = bins_x1_array[i]
        ix2 = bins_x2_array[i]
        iy = bins_y_array[i]
        k = nbins_x1*ix1 + ix2
        marginal_x1x2_array[np.int64(k)] += 1
        marginal_y_array[np.int64(iy)] += 1
        grid_array[np.int64(k*nbins_y + iy)] += 1
        

@numba.jit(nopython=True, parallel=True)
def bivar_ur_calc(ur_matrix, col_pairs,
                  bins_x, bins_y, nbins_x,
                  marginal_x1x2, marginal_y, grid):
    
    for c in numba.prange(col_pairs.shape[0]):
        
        x1 = col_pairs[c, 0]
        x2 = col_pairs[c, 1]
    
        for i in range(bins_y.shape[0]):

            populate_bivariate_marginals_grid(bins_x[x1], bins_x[x2], bins_y[i],
                                              nbins_x[x1,0], nbins_x[x2,0],
                                              marginal_x1x2[c,i],
                                              marginal_y[c,i],
                                              grid[c,i])

            nbins_x1x2 = nbins_x[x1,0] * nbins_x[x2,0]
            nbins_y = marginal_y[c,i].shape[0]

            # calculate number of cases first
            ncases = 0
            for j in range(nbins_x1x2):
                ncases += marginal_x1x2[c,i,j]

            # Test for single bin row or column
            if nbins_x1x2 < 2 or nbins_y < 2:
                # Assign value of 0
                UR = 0.0
            else:    
                UR = ur_matrix[c, i]
                entropy_x1x2 = 0.0
                entropy_y = 0.0
                entropy_joint = 0.0

                for j in range(nbins_x1x2):
                    px1x2 = marginal_x1x2[c,i,j]/ncases
                    entropy_x1x2 -= px1x2 * math.log(px1x2)
                    
                for k in range(nbins_y):
                    py = marginal_y[c,i,k]/ncases
                    entropy_y -= py * math.log(py)
                    
                for j in range(nbins_x1x2):
                    for k in range(nbins_y):
                        px1x2y = grid[c,i,j*nbins_y + k]/ncases
                        entropy_joint -= px1x2y * math.log(px1x2y)

                if entropy_y > 0:
                    UR = (entropy_x1x2 + entropy_y - entropy_joint) / entropy_y
                else:
                    UR = 0.0
            
                ur_matrix[c,i] = UR        
        
        
        
@numba.jit(nopython=True, nogil=True, parallel=True)
def bivar_mutinf_discrete_calc(mi_matrix, col_pairs,
                               bins_x, bins_y, nbins_x,
                               marginal_x1x2, marginal_y, grid):
    
    # For each combination of variables
    for c in numba.prange(col_pairs.shape[0]):
    
        x1 = col_pairs[c, 0]
        x2 = col_pairs[c, 1]
    
        # For each column/array in the dependent variable matrix
        for i in range(bins_y.shape[0]):
            
            populate_bivariate_marginals_grid(bins_x[x1], bins_x[x2],
                                              bins_y[i],
                                              nbins_x[x1,0], nbins_x[x2,0],
                                              marginal_x1x2[c,i],
                                              marginal_y[c,i],
                                              grid[c,i])

            nbins_x1x2 = nbins_x[x1,0] * nbins_x[x2,0]
            nbins_y = marginal_y[c,i].shape[0]
            ncases = bins_x.shape[1]

            MI = mi_matrix[c,i]

            for j in range(nbins_x1x2):
                px1x2 = marginal_x1x2[c,i,j]/ncases
                for k in range(nbins_y):
                    py = marginal_y[c,i,k]/ncases
                    px1x2y = grid[c,i,j*nbins_y + k]/ncases
                    if px1x2y > 0:
                        MI += px1x2y * math.log(px1x2y / (px1x2*py))
                        
            mi_matrix[c,i] = MI
            
            
