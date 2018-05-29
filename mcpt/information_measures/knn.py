import numpy as np

import numba

from ..neighbors import kd_tree
from ..neighbors import ball_tree

from ..utils.unique import np_unique
from ..utils.digamma import digamma_cpu

from ..information_measures.discrete import mutinf_discrete_calc


@numba.jit(nopython=True)
def compute_mi_cc(x, y, n_neighbors=3):

    leaf_size = 30

    x = x.reshape((-1, 1))
    y = y.reshape((-1, 1))
    xy = np.hstack((x, y))

    n_samples = xy.shape[0]
    n_features = xy.shape[1]

    radius = np.empty(n_samples)

    # create the objects that are going to be needed for NN
    n_levels = 1 + np.log2(max(1, ((n_samples - 1) // leaf_size)))
    n_nodes = int(2 ** n_levels) - 1
    # allocate arrays for storage
    idx_array = np.arange(n_samples)
    node_radius = np.zeros(n_nodes, dtype=np.float64)
    node_idx_start = np.zeros(n_nodes, dtype=np.int64)
    node_idx_end = np.zeros(n_nodes, dtype=np.int64)
    node_is_leaf = np.zeros(n_nodes, dtype=np.int64)
    node_centroids = np.zeros((n_nodes, n_features), dtype=np.float64)
    # set metric==1 for chebyshev distance
    ball_tree.recursive_build(0, 0, n_samples, xy, node_centroids,
                              node_radius, idx_array, node_idx_start,
                              node_idx_end, node_is_leaf, n_nodes,
                              leaf_size, metric=1)
    # This algorithm returns the point itself as a neighbor, so
    # if n_neighbors need to be returned then '1' needs to be
    # added in order to get the correct value from 'nth'
    # neighbor when the heap is created
    heap_distances, heap_indices = ball_tree.heap_create(n_samples, n_neighbors+1)
    ball_tree.query(0, xy, heap_distances, heap_indices,
                    xy, idx_array, node_centroids, node_radius,
                    node_is_leaf, node_idx_start, node_idx_end,
                    metric=1)
    ball_tree.heap_sort(heap_distances, heap_indices)
    radius = np.nextafter(heap_distances[:, -1], 0)

    # A whole new set of Tree elements need to be created for the KDTree
    # algorithms that are going to be run on both the x and y arrays that
    # were initially passed in.
    #
    # Perform KD-tree NN on x array
    n_samples_kd = x.shape[0]

    # determine number of levels in the tree, and from this
    # the number of nodes in the tree.  This results in leaf nodes
    # with numbers of points betweeen leaf_size and 2 * leaf_size
    n_levels_kd = 1 + np.log2(max(1, ((n_samples_kd - 1) // leaf_size)))
    # having to round first and then apply int in order to calculate
    # correct number of nodes
    n_nodes_kd = int(round((2 ** n_levels_kd))) - 1

    # allocate arrays for storage
    idx_array_kd = np.arange(n_samples_kd)
    node_radius_kd = np.zeros(n_nodes_kd, dtype=np.float64)
    node_idx_start_kd = np.zeros(n_nodes_kd, dtype=np.int64)
    node_idx_end_kd = np.zeros(n_nodes_kd, dtype=np.int64)
    node_is_leaf_kd = np.zeros(n_nodes_kd, dtype=np.int64)
    node_lower_bounds_kd = np.zeros((n_nodes_kd, n_features), dtype=np.float64)
    node_upper_bounds_kd = np.zeros((n_nodes_kd, n_features), dtype=np.float64)

    # use 'chebyshev' distance as metric (metric==1)
    kd_tree.recursive_build(0, 0, n_samples_kd, x,
                            node_lower_bounds_kd, node_upper_bounds_kd,
                            node_radius_kd, idx_array_kd,
                            node_idx_start_kd, node_idx_end_kd,
                            node_is_leaf_kd, n_nodes_kd, leaf_size, metric=1)

    count_only = True
    return_distance = False
    counts_x = \
        kd_tree.radius_neighbors_count(x, radius,
                                       idx_array_kd, node_lower_bounds_kd,
                                       node_upper_bounds_kd, node_radius_kd,
                                       node_is_leaf_kd, node_idx_start_kd,
                                       node_idx_end_kd, count_only, return_distance,
                                       metric=1)

    # Perform KD-tree NN on y array
    # Note: The data structures to perform the KD-tree build and search should
    # be the same for x and y. In order to preserve memory, re-using the same
    # objects

    # use 'chebyshev' distance as metric (metric==1)
    kd_tree.recursive_build(0, 0, n_samples_kd, y,
                            node_lower_bounds_kd, node_upper_bounds_kd,
                            node_radius_kd, idx_array_kd,
                            node_idx_start_kd, node_idx_end_kd,
                            node_is_leaf_kd, n_nodes_kd, leaf_size, metric=1)

    count_only = True
    return_distance = False
    counts_y = \
        kd_tree.radius_neighbors_count(y, radius,
                                       idx_array_kd, node_lower_bounds_kd,
                                       node_upper_bounds_kd, node_radius_kd,
                                       node_is_leaf_kd, node_idx_start_kd,
                                       node_idx_end_kd, count_only, return_distance,
                                       metric=1)

    mi = (digamma_cpu(n_samples) + digamma_cpu(n_neighbors) -
          np.mean(digamma_cpu(counts_x)) - np.mean(digamma_cpu(counts_y)))

    mi = max(0, mi)

    return mi


@numba.jit(nopython=True)
def compute_mi_cd(c, d, n_neighbors=3):

    leaf_size = 30

    c = c.reshape((-1, 1))

    n_samples = c.shape[0]
    n_features = c.shape[1]

    radius = np.empty(n_samples)
    label_counts = np.empty(n_samples, dtype=np.int64)
    k_all = np.empty(n_samples, dtype=np.int8)

    labels = np_unique(d)
    n_labels = len(labels)

    for idx in range(n_labels):
        label = labels[idx]
        mask = np.where(d.ravel() == label)[0]
        count = mask.shape[0]
        if count > 1:
            # create the objects that are going to be needed for NN
            n_levels = 1 + np.log2(max(1, ((count - 1) // leaf_size)))
            n_nodes = int(2 ** n_levels) - 1
            # allocate arrays for storage
            idx_array = np.arange(count)
            node_radius = np.zeros(n_nodes, dtype=np.float64)
            node_idx_start = np.zeros(n_nodes, dtype=np.int64)
            node_idx_end = np.zeros(n_nodes, dtype=np.int64)
            node_is_leaf = np.zeros(n_nodes, dtype=np.int64)
            node_centroids = np.zeros((n_nodes, n_features), dtype=np.float64)
            ball_tree.recursive_build(0, 0, count, c[mask], node_centroids,
                                      node_radius, idx_array, node_idx_start,
                                      node_idx_end, node_is_leaf, n_nodes,
                                      leaf_size, metric=0)
            # This algorithm returns the point itself as a neighbor, so
            # if n_neighbors need to be returned then '1' needs to be
            # added to 'k' in order to get the correct value from 'nth'
            # neighbor when the heap is created
            k = min(n_neighbors, count-1)
            heap_distances, heap_indices = ball_tree.heap_create(count, k+1)
            ball_tree.query(0, c[mask], heap_distances, heap_indices,
                            c[mask], idx_array, node_centroids, node_radius,
                            node_is_leaf, node_idx_start, node_idx_end,
                            metric=0)
            ball_tree.heap_sort(heap_distances, heap_indices)
            heap_distances = np.sqrt(heap_distances)
            radius[mask] = np.nextafter(heap_distances[:, -1], 0)
            k_all[mask] = k
        label_counts[mask] = count

    # Ignore points with unique labels
    mask_unique = np.array([n if label_counts[n] > 1 else 0 for n in range(n_samples)])

    # A whole new set of Tree elements need to be created since the entire
    # data set is now going to be run throught the algorithm
    n_samples_kd = c[mask_unique].shape[0]

    # determine number of levels in the tree, and from this
    # the number of nodes in the tree.  This results in leaf nodes
    # with numbers of points betweeen leaf_size and 2 * leaf_size
    n_levels_kd = 1 + np.log2(max(1, ((n_samples_kd - 1) // leaf_size)))
    # having to round first and then apply int in order to calculate
    # correct number of nodes
    n_nodes_kd = int(round((2 ** n_levels_kd))) - 1

    # allocate arrays for storage
    idx_array_kd = np.arange(n_samples_kd)
    node_radius_kd = np.zeros(n_nodes_kd, dtype=np.float64)
    node_idx_start_kd = np.zeros(n_nodes_kd, dtype=np.int64)
    node_idx_end_kd = np.zeros(n_nodes_kd, dtype=np.int64)
    node_is_leaf_kd = np.zeros(n_nodes_kd, dtype=np.int64)
    node_lower_bounds_kd = np.zeros((n_nodes_kd, n_features), dtype=np.float64)
    node_upper_bounds_kd = np.zeros((n_nodes_kd, n_features), dtype=np.float64)

    kd_tree.recursive_build(0, 0, n_samples_kd, c[mask_unique],
                            node_lower_bounds_kd, node_upper_bounds_kd,
                            node_radius_kd, idx_array_kd,
                            node_idx_start_kd, node_idx_end_kd,
                            node_is_leaf_kd, n_nodes_kd, leaf_size)

    count_only = True
    return_distance = False
    counts = \
        kd_tree.radius_neighbors_count(c[mask_unique], radius[mask_unique],
                                       idx_array_kd, node_lower_bounds_kd,
                                       node_upper_bounds_kd, node_radius_kd,
                                       node_is_leaf_kd, node_idx_start_kd,
                                       node_idx_end_kd, count_only, return_distance)

    mi = (digamma_cpu(n_samples_kd) + np.mean(digamma_cpu(k_all[mask_unique])) -
          np.mean(digamma_cpu(label_counts[mask_unique])) - np.mean(digamma_cpu(counts)))

    mi = max(0, mi)

    return mi


@numba.jit(nopython=True, parallel=True)
def mutinf_knn_calc(X, y_matrix, x_types, y_type):

    # Number of variables will be number of rows in X
    # Number of mcpt reps will be number of fows in y_matrix
    mi_matrix = np.zeros((X.shape[0], y_matrix.shape[0]))

    for var in numba.prange(X.shape[0]):
        # parallel across all y and permutations will
        # be handled either within the individual functions
        # or within the individual conditions

        # Encoding for variable types:
        # 0: numeric
        # 1: discrete
        x_type = x_types[var]
        if x_type == 0 and y_type == 0:
            for i in numba.prange(y_matrix.shape[0]):
                mi_matrix[var,i] = \
                    compute_mi_cc(y_matrix[i], X[var], n_neighbors=3)
        elif x_type == 1 and y_type == 0:
            for i in numba.prange(y_matrix.shape[0]):
                mi_matrix[var,i] = \
                    compute_mi_cd(y_matrix[i], X[var], n_neighbors=3)
        elif x_type == 0 and y_type == 1:
            for i in numba.prange(y_matrix.shape[0]):
                mi_matrix[var,i] = \
                    compute_mi_cd(X[var], y_matrix[i], n_neighbors=3)
        elif x_type == 1 and y_type == 1:
            for i in numba.prange(y_matrix.shape[0]):
                # When X.values is applied and there are columns with
                # differing dtypes, pandas creates a numpy array of
                # type 'object'.
                # Insure that X[var] is being passed in as a numeric
                # into the mutinf_discrete_calc function
                # unsure how numba is handling arrays of type 'object'
                x = X[var].reshape(-1,1).astype(np.int64).ravel()
                mi_matrix[var,i] = \
                    mutinf_discrete_calc(x, y_matrix[i])

    return mi_matrix

