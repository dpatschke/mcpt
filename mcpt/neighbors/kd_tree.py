import warnings
import numpy as np

import numba

###########################
## Distance computations ##
###########################
# 'rdist' stands for reduced distance. In the case of the euclidean
# distance, the sum of squared distances will be calculated and used
# through the tree-associated functions. Only at the end, will the
# the square root of the 'rdist' be taken for the actual 'dist'
@numba.jit(nopython=True)
def rdist(X1, i1, X2, i2, metric):
    d = 0.0
    # euclidean distance
    if metric == 0:
        for k in range(X1.shape[1]):
            tmp = (X1[i1, k] - X2[i2, k])
            d += tmp * tmp
    # chebyshev distance
    elif metric == 1:
        for k in range(X1.shape[1]):
            tmp = np.abs(X1[i1, k] - X2[i2, k])
            d = max(d, tmp)
    return d


@numba.jit(nopython=True)
def min_rdist(node_lower_bounds, node_upper_bounds, i_node, X, i, metric):
    rdist = 0.0
    # euclidean distance
    if metric == 0:
        for j in range(X.shape[1]):
            d_lo = node_lower_bounds[i_node, j] - X[i,j]
            d_hi = X[i,j] - node_upper_bounds[i_node, j]
            d = (d_lo + np.abs(d_lo)) + (d_hi + np.abs(d_hi))
            rdist += np.square(0.5 * d)
    # chebyshev distance
    elif metric == 1:
        for j in range(X.shape[1]):
            d_lo = node_lower_bounds[i_node, j] - X[i,j]
            d_hi = X[i,j] - node_upper_bounds[i_node, j]
            d = (d_lo + np.abs(d_lo)) + (d_hi + np.abs(d_hi))
            rdist = max(rdist, 0.5 * d)
    return rdist


@numba.jit(nopython=True)
def max_rdist(node_lower_bounds, node_upper_bounds, i_node, X, i, metric):
    rdist = 0.0
    # euclidean distance
    if metric == 0:
        for j in range(X.shape[1]):
            d_lo = np.abs(X[i,j] - node_lower_bounds[i_node, j])
            d_hi = np.abs(X[i,j] - node_upper_bounds[i_node, j])
            rdist += np.square(max(d_lo, d_hi))
    # chebyshev distance
    elif metric == 1:
        for j in range(X.shape[1]):
            rdist = max(rdist, np.abs(X[i,j] - node_lower_bounds[i_node, j]))
            rdist = max(rdist, np.abs(X[i,j] - node_upper_bounds[i_node, j]))

    return rdist


@numba.jit(nopython=True)
def min_max_dist(i_node, X, i, dist_bounds_arr,
                 node_lower_bounds, node_upper_bounds,
                 metric):
    # euclidean distance
    if metric == 0:
        d_lo = 0.0
        d_hi = 0.0
        d = 0.0
        for j in range(X.shape[1]):
            d_lo = node_lower_bounds[i_node, j] - X[i, j]
            d_hi = X[i, j] - node_upper_bounds[i_node, j]
            d = (d_lo + np.abs(d_lo)) + (d_hi + np.abs(d_hi))
            dist_bounds_arr[0] += (0.5 * d) ** 2
            dist_bounds_arr[1] += (max(np.abs(d_lo), np.abs(d_hi))) ** 2
        dist_bounds_arr[0] = np.sqrt(dist_bounds_arr[0])
        dist_bounds_arr[1] = np.sqrt(dist_bounds_arr[1])
    # chebyshev distance
    elif metric == 1:
        d_lo = 0.0
        d_hi = 0.0
        d = 0.0
        for j in range(X.shape[1]):
            d_lo = node_lower_bounds[i_node, j] - X[i, j]
            d_hi = X[i, j] - node_upper_bounds[i_node, j]
            d = (d_lo + np.abs(d_lo)) + (d_hi + np.abs(d_hi))
            dist_bounds_arr[0] = max(dist_bounds_arr[0], 0.5 * d)
            dist_bounds_arr[1] = max(dist_bounds_arr[1],
                                     np.abs(X[i,j] - node_lower_bounds[i_node, j]))
            dist_bounds_arr[1] = max(dist_bounds_arr[1],
                                     np.abs(X[i,j] - node_upper_bounds[i_node, j]))


#########################################
## Tree Building functions for KD-tree ##
#########################################
@numba.jit(nopython=True)
def partition_indices(data, idx_array, idx_start, idx_end, split_index):
    # Find the split dimension
    n_features = data.shape[1]

    split_dim = 0
    max_spread = 0

    for j in range(n_features):
        max_val = -np.inf
        min_val = np.inf
        for i in range(idx_start, idx_end):
            val = data[idx_array[i], j]
            max_val = max(max_val, val)
            min_val = min(min_val, val)
        if max_val - min_val > max_spread:
            max_spread = max_val - min_val
            split_dim = j

    # Partition using the split dimension
    left = idx_start
    right = idx_end - 1

    while True:
        midindex = left
        for i in range(left, right):
            d1 = data[idx_array[i], split_dim]
            d2 = data[idx_array[right], split_dim]
            if d1 < d2:
                tmp = idx_array[i]
                idx_array[i] = idx_array[midindex]
                idx_array[midindex] = tmp
                midindex += 1
        tmp = idx_array[midindex]
        idx_array[midindex] = idx_array[right]
        idx_array[right] = tmp
        if midindex == split_index:
            break
        elif midindex < split_index:
            left = midindex + 1
        else:
            right = midindex - 1


@numba.jit(nopython=True)
def recursive_build(i_node, idx_start, idx_end, data,
                    node_lower_bounds, node_upper_bounds,
                    node_radius, idx_array,
                    node_idx_start, node_idx_end,
                    node_is_leaf, n_nodes, leaf_size,
                    metric=0):
    # Initialize radius
    # if metric == 0, then euclidean and radius is squared radius
    # if metric == 1, then chebyshev
    radius = 0.0

    # determine node upper/lower bounds and radius
    for j in range(data.shape[1]):
        node_lower_bounds[i_node, j] = np.inf
        node_upper_bounds[i_node, j] = -np.inf

    for i in range(idx_start, idx_end):
        for j in range(data.shape[1]):
            value = data[idx_array[i], j]
            if value < node_lower_bounds[i_node, j]:
                node_lower_bounds[i_node, j] = value
            elif value > node_upper_bounds[i_node, j]:
                node_upper_bounds[i_node, j] = value
        diff = node_upper_bounds[i_node, j] - node_lower_bounds[i_node, j]
        if metric == 0:
            # euclidean
            radius += np.square(0.5 * diff)
        elif metric == 1:
            # chebyshev
            radius = max(radius, 0.5 * np.abs(diff))

    # set node properties
    if metric == 0:
        node_radius[i_node] = np.sqrt(radius)
    elif metric == 1:
        node_radius[i_node] = radius
    node_idx_start[i_node] = idx_start
    node_idx_end[i_node] = idx_end

    i_child = 2 * i_node + 1

    # recursively create subnodes
    if i_child >= n_nodes:
        node_is_leaf[i_node] = True
        if idx_end - idx_start > 2 * leaf_size:
            # this shouldn't happen if our memory allocation is correct.
            # We'll proactively prevent memory errors, but raise a
            # warning saying we're doing so.
            #warnings.warn("Internal: memory layout is flawed: "
            #              "not enough nodes allocated")
            pass

    elif idx_end - idx_start < 2:
        # again, this shouldn't happen if our memory allocation is correct.
        #warnings.warn("Internal: memory layout is flawed: "
        #              "too many nodes allocated")
        node_is_leaf[i_node] = True

    else:
        # split node and recursively construct child nodes.
        node_is_leaf[i_node] = False
        n_mid = int((idx_end + idx_start) // 2)
        partition_indices(data, idx_array, idx_start, idx_end, n_mid)
        recursive_build(i_child, idx_start, n_mid, data,
                        node_lower_bounds, node_upper_bounds,
                        node_radius, idx_array,
                        node_idx_start, node_idx_end,
                        node_is_leaf, n_nodes, leaf_size, metric)
        recursive_build(i_child + 1, n_mid, idx_end, data,
                        node_lower_bounds, node_upper_bounds,
                        node_radius, idx_array,
                        node_idx_start, node_idx_end,
                        node_is_leaf, n_nodes, leaf_size, metric)


############################################
## Radius neighbors functions for KD-tree ##
############################################
@numba.jit(nopython=True)
def query_radius_recursive(i_node, X, i_pt, r, idx_array,
                           node_lower_bounds, node_upper_bounds,
                           node_radius, node_is_leaf,
                           node_idx_start, node_idx_end,
                           count, indices, distances,
                           count_only, return_distance,
                           metric=0):

    # if metric == 0, then euclidean and radius is squared radius
    # if metric == 1, then chebyshev
    dist_bounds_arr = np.zeros(2, dtype=np.float64)
    min_max_dist(i_node, X, i_pt, dist_bounds_arr,
                 node_lower_bounds, node_upper_bounds, metric)
    dist_LB = dist_bounds_arr[0]
    dist_UB = dist_bounds_arr[1]

    #------------------------------------------------------------
    # Case 1: all node points are outside distance r.
    #         prune this branch.
    if dist_LB > r:
        pass

    #------------------------------------------------------------
    # Case 2: all node points are within distance r
    #         add all points to neighbors
    elif dist_UB <= r:
        if count_only:
            count += (node_idx_end[i_node] - node_idx_start[i_node])
        else:
            for i in range(node_idx_start[i_node],
                           node_idx_end[i_node]):
                if (count < 0) or (count >= X.shape[0]):
                    raise ValueError("Fatal: count too big: "
                                     "this should never happen")
                indices[count] = idx_array[i]
                if return_distance:
                    dist_pt = rdist(X, i_pt, X, idx_array[i], metric)
                    if metric == 0:
                        distances[count] = np.sqrt(dist_pt)
                    elif metric == 1:
                        distances[count] = dist_pt
                count += 1

    #------------------------------------------------------------
    # Case 3: this is a leaf node.  Go through all points to
    #         determine if they fall within radius
    elif node_is_leaf[i_node]:
        if metric == 0:
            radius = r ** 2
        elif metric == 1:
            radius = r

        for i in range(node_idx_start[i_node],
                           node_idx_end[i_node]):
            dist_pt = rdist(X, i_pt, X, idx_array[i], metric)
            if dist_pt <= radius:
                if (count < 0) or (count >= X.shape[0]):
                    raise ValueError("Fatal: count out of range. "
                                     "This should never happen.")
                if count_only:
                    pass
                else:
                    indices[count] = idx_array[i]
                    if return_distance:
                        if metric == 0:
                            distances[count] = np.sqrt(dist_pt)
                        elif metric == 1:
                            distances[count] = dist_pt
                count += 1

    #------------------------------------------------------------
    # Case 4: Node is not a leaf.  Recursively query subnodes
    else:
        i1 = 2 * i_node + 1
        i2 = i1 + 1
        count = query_radius_recursive(i1, X, i_pt, r, idx_array,
                                        node_lower_bounds, node_upper_bounds,
                                        node_radius, node_is_leaf,
                                        node_idx_start, node_idx_end,
                                        count, indices, distances,
                                        count_only, return_distance, metric)
        count = query_radius_recursive(i2, X, i_pt, r, idx_array,
                                        node_lower_bounds, node_upper_bounds,
                                        node_radius, node_is_leaf,
                                        node_idx_start, node_idx_end,
                                        count, indices, distances,
                                        count_only, return_distance, metric)

    return count


@numba.jit(nopython=True)
def radius_neighbors_count(X, r_arr, idx_array,
                           node_lower_bounds, node_upper_bounds,
                           node_radius, node_is_leaf,
                           node_idx_start, node_idx_end,
                           count_only=False, return_distance=False,
                           metric=0):

    # if metric == 0, then euclidean and radius is squared radius
    # if metric == 1, then chebyshev
    indices = np.zeros(X.shape[0], dtype=np.int32)
    distances = np.zeros(X.shape[0], dtype=np.float64)
    counts = np.zeros(X.shape[0], dtype=np.int32)

    for i in range(X.shape[0]):
        i_node = 0
        count = 0
        counts[i] = \
            query_radius_recursive(i_node, X, i, r_arr[i], idx_array,
                                   node_lower_bounds, node_upper_bounds,
                                   node_radius, node_is_leaf,
                                   node_idx_start, node_idx_end,
                                   count, indices, distances,
                                   count_only, return_distance, metric)

    return counts


########################
## The KD-Tree object ##
########################
class KDTree(object):
    def __init__(self, data, leaf_size=40, metric='euclidean'):
        self.data = data
        self.leaf_size = leaf_size

        # set the metric to be numeric in order to use numba
        if metric == 'euclidean':
            self.metric = 0
        elif metric == 'chebyshev':
            self.metric = 1

        # validate data
        if self.data.size == 0:
            raise ValueError("X is an empty array")

        if leaf_size < 1:
            raise ValueError("leaf_size must be greater than or equal to 1")

        self.n_samples = self.data.shape[0]
        self.n_features = self.data.shape[1]

        # determine number of levels in the tree, and from this
        # the number of nodes in the tree.  This results in leaf nodes
        # with numbers of points betweeen leaf_size and 2 * leaf_size
        self.n_levels = 1 + np.log2(max(1, ((self.n_samples - 1)
                                            // self.leaf_size)))
        self.n_nodes = int(2 ** self.n_levels) - 1

        # allocate arrays for storage
        self.idx_array = np.arange(self.n_samples, dtype=int)
        self.node_radius = np.zeros(self.n_nodes, dtype=float)
        self.node_idx_start = np.zeros(self.n_nodes, dtype=int)
        self.node_idx_end = np.zeros(self.n_nodes, dtype=int)
        self.node_is_leaf = np.zeros(self.n_nodes, dtype=int)
        self.node_lower_bounds = np.zeros((self.n_nodes, self.n_features), dtype=float)
        self.node_upper_bounds = np.zeros((self.n_nodes, self.n_features), dtype=float)

        # Allocate tree-specific data from TreeBase
        recursive_build(0, 0, self.n_samples, self.data,
                         self.node_lower_bounds, self.node_upper_bounds,
                         self.node_radius, self.idx_array,
                         self.node_idx_start, self.node_idx_end,
                         self.node_is_leaf, self.n_nodes, self.leaf_size, self.metric)


    def query_radius_neighbors_count(self, X, r, return_distance=False,
                                     count_only=False, sort_results=True):
        X = np.asarray(X, dtype=float)

        if X.shape[-1] != self.n_features:
            raise ValueError("query data dimension must "
                             "match training data dimension")

        # prepare r for query
        r = np.asarray(r, dtype=float)
        r = np.atleast_1d(r)
        if r.shape == (1,):
            r = r[0] + np.zeros(X.shape[:X.ndim - 1], dtype=float)
        else:
            if r.shape != X.shape[:X.ndim - 1]:
                raise ValueError("r must be broadcastable to X.shape")

        rarr = r.reshape(-1)  # store explicitly to keep in scope

        # prepare variables for iteration
        #if not count_only:
        #    indices = np.zeros(X.shape[0], dtype='object')
        #    if return_distance:
        #        distances = np.zeros(X.shape[0], dtype='object')

        counts = radius_neighbors_count(X, rarr, self.idx_array,
                                        self.node_lower_bounds,
                                        self.node_upper_bounds,
                                        self.node_radius, self.node_is_leaf,
                                        self.node_idx_start, self.node_idx_end,
                                        count_only, return_distance, self.metric)

        return counts
