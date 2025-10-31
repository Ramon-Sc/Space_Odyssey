import networkx as nx
import numpy as np
import scipy.sparse.linalg as spla




# nx-cugraph backend available
class graphanalyzer:
    def __init__(self, config):
        self.config = config
        
    
    def laplacian_matrix(self, graph):
        return nx.laplacian_matrix(graph)

    def laplacian_spectrum(self, graph, k=None):
        """
        Compute the Laplacian spectrum (eigenvalues) of a graph.
        
        Parameters:
        -----------
        graph : networkx.Graph
            The input graph
        k : int, optional
            Number of eigenvalues to compute. If None, compute all eigenvalues.
            For large graphs, computing only k eigenvalues is much more efficient.
        
        Returns:
        --------
        eigenvalues : ndarray
            The Laplacian eigenvalues, sorted in ascending order
        """
        L = nx.laplacian_matrix(graph)
        n = L.shape[0]
        
        # For small graphs, compute all eigenvalues using dense method
        if n < 100 or k is None:
            try:
                return nx.laplacian_spectrum(graph)
            except Exception as e:
                print(f"Warning: Dense eigenvalue computation failed, trying sparse method: {e}")
        
        # For larger graphs or when k is specified, use sparse eigenvalue solver
        if k is not None:
            k = min(k, n - 2)  # scipy requires k < n-1
            if k < 1:
                k = 1
        else:
            # If graph is large and k not specified, compute top 100 eigenvalues
            k = min(100, n - 2)
        
        try:
            # Compute k smallest eigenvalues using sparse solver
            eigenvalues = spla.eigsh(L.astype(np.float64), k=k, which='SM', return_eigenvectors=False)
            return np.sort(eigenvalues)
        except Exception as e:
            print(f"Warning: Sparse eigenvalue computation failed: {e}")
            # Fall back to dense computation for small subset
            try:
                L_dense = L.toarray()
                eigenvalues = np.linalg.eigvalsh(L_dense)
                return np.sort(eigenvalues)
            except Exception as e2:
                print(f"Error: All eigenvalue computation methods failed: {e2}")
                return np.array([0.0])  # Return zero as fallback

