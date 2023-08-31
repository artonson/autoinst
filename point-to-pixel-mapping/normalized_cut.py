import numpy as np
import networkx as nx
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg

''' Mainly copied from scipy skimage library, with some modifications to return combined graph'''

def argmin2(array):
    """Return the index of the 2nd smallest value in an array.

    Parameters
    ----------
    array : array
        The array to process.

    Returns
    -------
    min_idx2 : int
        The index of the second smallest value.
    """
    min1 = np.inf
    min2 = np.inf
    min_idx1 = 0
    min_idx2 = 0
    i = 0
    n = array.shape[0]

    for i in range(n):
        x = array[i]
        if x < min1:
            min2 = min1
            min_idx2 = min_idx1
            min1 = x
            min_idx1 = i
        elif x > min1 and x < min2:
            min2 = x
            min_idx2 = i
        i += 1

    return min_idx2

def cut_cost(cut, W_data, W_indices, W_indptr, num_cols):
    """Return the total weight of crossing edges in a bi-partition.

    Parameters
    ----------
    cut : array
        A array of booleans. Elements set to `True` belong to one
        set.
    W_data : array
        The data of the sparse weight matrix of the graph.
    W_indices : array
        The indices of the sparse weight matrix of the graph.
    W_indptr : array
        The index pointers of the sparse weight matrix of the graph.
    num_cols : int
        The number of columns in the sparse weight matrix of the graph.

    Returns
    -------
    cost : float
        The total weight of crossing edges.
    """
    cut_mask = np.array(cut, dtype=np.uint8)
    cost = 0

    for col in range(num_cols):
        for row_index in range(W_indptr[col], W_indptr[col + 1]):
            row = W_indices[row_index]
            if cut_mask[row] != cut_mask[col]:
                cost += W_data[row_index]

    return cost * 0.5

def ncut_cost(cut, D, W):
    """Returns the N-cut cost of a bi-partition of a graph.

    Parameters
    ----------
    cut : ndarray
        The mask for the nodes in the graph. Nodes corresponding to a `True`
        value are in one set.
    D : csc_matrix
        The diagonal matrix of the graph.
    W : csc_matrix
        The weight matrix of the graph.

    Returns
    -------
    cost : float
        The cost of performing the N-cut.

    References
    ----------
    .. [1] Normalized Cuts and Image Segmentation, Jianbo Shi and
           Jitendra Malik, IEEE Transactions on Pattern Analysis and Machine
           Intelligence, Page 889, Equation 2.
    """
    cut = np.array(cut)
    cost = cut_cost(cut, W.data, W.indices, W.indptr, num_cols=W.shape[0])

    # D has elements only along the diagonal, one per node, so we can directly
    # index the data attribute with cut.
    assoc_a = D.data[cut].sum()
    assoc_b = D.data[~cut].sum()

    return (cost / assoc_a) + (cost / assoc_b)

def DW_matrices(graph):
    """Returns the diagonal and weight matrices of a graph.

    Parameters
    ----------
    graph : RAG
        A Region Adjacency Graph.

    Returns
    -------
    D : csc_matrix
        The diagonal matrix of the graph. ``D[i, i]`` is the sum of weights of
        all edges incident on `i`. All other entries are `0`.
    W : csc_matrix
        The weight matrix of the graph. ``W[i, j]`` is the weight of the edge
        joining `i` to `j`.
    """
    # sparse.eighsh is most efficient with CSC-formatted input
    W = nx.to_scipy_sparse_array(graph, format='csc')
    entries = W.sum(axis=0)
    D = sparse.dia_matrix((entries, 0), shape=W.shape).tocsc()

    return D, W

def get_min_ncut(ev, d, w, num_cuts):
    """Threshold an eigenvector evenly, to determine minimum ncut.

    Parameters
    ----------
    ev : array
        The eigenvector to threshold.
    d : ndarray
        The diagonal matrix of the graph.
    w : ndarray
        The weight matrix of the graph.
    num_cuts : int
        The number of evenly spaced thresholds to check for.

    Returns
    -------
    mask : array
        The array of booleans which denotes the bi-partition.
    mcut : float
        The value of the minimum ncut.
    """
    mcut = np.inf
    mn = ev.min()
    mx = ev.max()

    # If all values in `ev` are equal, it implies that the graph can't be
    # further sub-divided. In this case the bi-partition is the the graph
    # itself and an empty set.
    min_mask = np.zeros_like(ev, dtype=bool)
    if np.allclose(mn, mx):
        return min_mask, mcut

    # Refer Shi & Malik 2001, Section 3.1.3, Page 892
    # Perform evenly spaced n-cuts and determine the optimal one.
    for t in np.linspace(mn, mx, num_cuts, endpoint=False):
        mask = ev > t
        cost = ncut_cost(mask, d, w)
        if cost < mcut:
            min_mask = mask
            mcut = cost

    return min_mask, mcut

def partition_by_cut(cut, rag):
    """Compute resulting subgraphs from given bi-partition.

    Parameters
    ----------
    cut : array
        A array of booleans. Elements set to `True` belong to one
        set.
    rag : RAG
        The Region Adjacency Graph.

    Returns
    -------
    sub1, sub2 : RAG
        The two resulting subgraphs from the bi-partition.
    """
    # `cut` is derived from `D` and `W` matrices, which also follow the
    # ordering returned by `rag.nodes()` because we use
    # nx.to_scipy_sparse_matrix.

    # Example
    # rag.nodes() = [3, 7, 9, 13]
    # cut = [True, False, True, False]
    # nodes1 = [3, 9]
    # nodes2 = [7, 10]

    nodes1 = [n for i, n in enumerate(rag.nodes()) if cut[i]]
    nodes2 = [n for i, n in enumerate(rag.nodes()) if not cut[i]]

    sub1 = rag.subgraph(nodes1)
    sub2 = rag.subgraph(nodes2)

    return sub1, sub2

def normalized_cut(rag, thresh, num_cuts, rng = None, in_place=False):
    """Perform Normalized Graph cut on the Region Adjacency Graph.

    Recursively partition the graph into 2, until further subdivision
    yields a cut greater than `thresh` or such a cut cannot be computed.
    For such a subgraph, indices to labels of all its nodes map to a single
    unique value.

    Parameters
    ----------
    rag : RAG
        The region adjacency graph.
    thresh : float
        The threshold. A subgraph won't be further subdivided if the
        value of the N-cut exceeds `thresh`.
    num_cuts : int
        The number or N-cuts to perform before determining the optimal one.
    random_generator : `numpy.random.Generator`
        Provides initial values for eigenvalue solver.
    """
    random_generator = np.random.default_rng(rng)
    
    if not in_place:
        rag = rag.copy()
    
    for node in rag.nodes():
        rag.add_edge(node, node, weight=1.0)

    d, w = DW_matrices(rag)
    m = w.shape[0]

    if m > 2:
        d2 = d.copy()
        # Since d is diagonal, we can directly operate on its data
        # the inverse of the square root
        d2.data = np.reciprocal(np.sqrt(d2.data, out=d2.data), out=d2.data)

        # Refer Shi & Malik 2001, Equation 7, Page 891
        A = d2 * (d - w) * d2
        # Initialize the vector to ensure reproducibility.
        v0 = random_generator.random(A.shape[0])
        vals, vectors = linalg.eigsh(A, which='SM', v0=v0,
                                     k=min(100, m - 2))

        # Pick second smallest eigenvector.
        # Refer Shi & Malik 2001, Section 3.2.3, Page 893
        vals, vectors = np.real(vals), np.real(vectors)
        index2 = argmin2(vals)
        ev = vectors[:, index2]

        cut_mask, mcut = get_min_ncut(ev, d, w, num_cuts)
        if (mcut < thresh):
            # Sub divide and perform N-cut again
            # Refer Shi & Malik 2001, Section 3.2.5, Page 893
            sub1, sub2 = partition_by_cut(cut_mask, rag)

            sub1_partioned = normalized_cut(sub1, thresh, num_cuts, random_generator)
            sub2_partioned = normalized_cut(sub2, thresh, num_cuts, random_generator)

            graphs = [sub1_partioned, sub2_partioned]
            combined_graphs = nx.union_all(graphs)

            return combined_graphs
        else: 
            #We cannot subdivide further
            return rag
    else:
        return rag