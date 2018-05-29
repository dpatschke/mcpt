import numpy as np
import itertools

from .utils.partition import masters_cut

from .information_measures.discrete import bivar_mutinf_discrete_calc
from .information_measures.discrete import bivar_ur_calc

from .utils.permutation import create_permutation_matrix
from .utils.pvalue import solo_pvalue, unbiased_pvalue, p_median_calc


def bivariate_discrete(X_bin_vars, y_bin_vars, n_reps, criterion='mi'):
    # Confirm that valid criterion is present
    assert criterion in ['mi','ur'], \
        "'criterion' must be either 'mi' or 'ur'."
    # create the marginal and bin arrays
    target = None
    if isinstance(X_bin_vars['bins'], np.ndarray):
        nbins_x = X_bin_vars['n_bins']
        nbins_y = y_bin_vars['n_bins']
        # Test to make sure n_reps value corresponds accordingly
        # to the 'bins_permuted' matrix
        assert (n_reps + 1) == y_bin_vars['bins_permuted'].shape[0], \
            "n_reps value differs from what is expected in 'bins_permuted'."
        target = 'cpu'
    else:
        # run on gpu
        nbins_x = X_bin_vars['n_bins'].copy_to_host()
        nbins_y = y_bin_vars['n_bins'].copy_to_host()
        target = 'gpu'

    # Since variables are going to be combined, need to generate
    # all two-variable combinations
    ncols = nbins_x.shape[0]
    col_pairs = [pair \
        for pair in itertools.combinations(np.arange(ncols), 2)]
    # cast as a numpy array
    col_pairs = np.asarray(col_pairs)
    # reassign the value of ncols to be the total number
    # of combinations
    ncombos = col_pairs.shape[0]

    # Since nbins_x and nbins_y are two-dimensional need to find
    # the max number of bins for each variable being evaluated and
    # create marginal/grid dimensions to these values
    nbins_x = np.max(nbins_x)
    nbins_y = np.max(nbins_y)

    # Furthermore, the bivariate information measure will require
    # that 'nbins_x' be squared for the marginal calculation since
    # the marginals will be 'unrolled' from two dimensions to one
    nbins_x_unrolled = nbins_x * nbins_x

    marginal_x1x2 = np.zeros((ncombos, n_reps+1, nbins_x_unrolled), np.int32)
    marginal_y = np.zeros((ncombos, n_reps+1, nbins_y), np.int32)
    grid = np.zeros((ncombos, n_reps+1, nbins_x_unrolled * nbins_y), np.int32)

    criterion_matrix = np.zeros((ncombos, n_reps+1), np.float32)

    if target == 'cpu':
        if criterion == 'mi':
            bivar_mutinf_discrete_calc(criterion_matrix, col_pairs,
                                       X_bin_vars['bins'],
                                       y_bin_vars['bins_permuted'],
                                       X_bin_vars['n_bins'],
                                       marginal_x1x2, marginal_y, grid)
        else:
            # criterion == 'ur'
            bivar_ur_calc(criterion_matrix, col_pairs,
                          X_bin_vars['bins'],
                          y_bin_vars['bins_permuted'],
                          X_bin_vars['n_bins'],
                          marginal_x1x2, marginal_y, grid)
    else:
        # target == 'gpu'
        # Currently don't have gpu enabled code
        raise ValueError

    return criterion_matrix


def screen_bivariate_calc(X, y,
                          method='discrete',
                          measure='mi',
                          n_bins_x=5, n_bins_y=5,
                          n_reps=100,
                          target='cpu'):

    X_bin_vars = masters_cut(X, nbins=n_bins_x, target=target)
    y_bin_vars = masters_cut(y, nbins=n_bins_y, target=target)

    y_bin_vars['bins_permuted'] = \
        create_permutation_matrix(y_bin_vars, n_reps=n_reps)

    if method == 'discrete':
        information_matrix = \
            bivariate_discrete(X_bin_vars, y_bin_vars,
                                n_reps=n_reps, criterion=measure)
    else:
        raise ValueError

    info = information_matrix[:,0]

    if n_reps > 0:
        solo_pval = solo_pvalue(information_matrix)
        unbiased_pval = unbiased_pvalue(information_matrix)
        # Create new 2d numpy matrix that contains the information
        # value and all the associated p-values
        information_matrix = np.column_stack((info, solo_pval, unbiased_pval))
    else:
        information_matrix = np.reshape(info, (info.shape[0],1))

    return information_matrix


def screen_bivariate(X, y,
                     method='discrete',
                     measure='mi',
                     n_bins_x=5, n_bins_y=5,
                     n_reps=100, target='cpu'):

    kwargs = {'method': method, 'measure': measure, \
             'n_bins_x': n_bins_x, 'n_bins_y': n_bins_y, \
             'target': target}

    info_matrix = screen_bivariate_calc(X, y, n_reps=n_reps, **kwargs)

    return info_matrix
