"""
Graph construction for superpixels 
"""
#HCS highliy connected subgraphs

import numpy as np
import networkx as nx
from scipy.spatial import Delaunay, Voronoi
from sklearn.neighbors import kneighbors_graph



class graphbuilder:
    def __init__(self, config):
        self.config = config

    def get_node_features(self, nodes):
        pass

    def graph_construction(self, points, features=None, method='delaunay', **kwargs):
        """
        Construct a graph from node positions.
        
        Parameters:
        -----------
        points : ndarray
            Array of node coordinates (n_nodes, n_dims)
        features : ndarray, optional
            Node features (n_nodes, n_features)
        method : str
            Graph construction method: 'delaunay', 'knn', 'voronoi', 'radius'
        **kwargs : dict
            Additional parameters for specific methods
            - n_neighbors: for knn method (default: 5)
            - radius: for radius method (default: 50)
        
        Returns:
        --------
        graph : object
            nx graph object
        """

        #initialize empty graph (NetworkX graph)
        graph = nx.Graph()
        
        #add nodes to the graph add coordinates as node features
        graph.add_nodes_from(range(len(points)))
        for i, point in enumerate(points):
            graph.nodes[i]['pos'] = point

        def get_edge_lengths(u, v):
            pos_u = np.array(graph.nodes[u]['pos'])
            pos_v = np.array(graph.nodes[v]['pos'])
            return np.linalg.norm(pos_v - pos_u)

        def get_adjacency_matrix(G):
            adjacency_matrix = np.zeros((len(G.nodes()), len(G.nodes())))
            for u, v in G.edges():
                adjacency_matrix[u, v] = 1
            return adjacency_matrix

        if len(points) < 3:
            return None
            
        if method == 'delaunay':

            #initialize empty adjacency matrix with shape (n_nodes, n_nodes)
            #edgelist may be better for sparse matrices
            #edges can be part of multiple triangles either remove duplicates afterwards or dont add edge to adjacency matrix i it already exists
            #adjacency_matrix = np.zeros((len(nodes), len(nodes)))

            #create delaunay triangulation
            tri = Delaunay(points)

            #iterate over all triangles
            #tri.simplices returns a list of simplices (indices of the points that form the triangle!!)
            for simplex in tri.simplices:
                for i in range(3):
                    for j in range(i+1, 3):
                        graph.add_weighted_edges_from([(simplex[i], simplex[j], get_edge_lengths(simplex[i], simplex[j]))])
                
            return graph

        elif method == 'knn':
            n_neighbors = kwargs.get('n_neighbors', 5)
            n_neighbors = min(n_neighbors, len(nodes) - 1)  # Ensure k < n_points
            return kneighbors_graph(nodes, n_neighbors=n_neighbors, mode='connectivity')
        
        elif method == 'voronoi':
            return Voronoi(nodes)
        
        elif method == 'radius':
            from sklearn.neighbors import radius_neighbors_graph
            radius = kwargs.get('radius', 50)
            return radius_neighbors_graph(nodes, radius=radius, mode='connectivity')
        
        else:
            raise ValueError(f"Unknown graph construction method: {method}")