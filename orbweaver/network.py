import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
from numba import jit
from .metrics import Metrics
from .matrix import build_matrix, export_matrix
from .stats import Stats
from .visuals import Visuals

class Network(object):
    
    """Object class that can hold a similarity matrix and a networkx graph. Calculates those from the data
    contained in a DataSet object, which is passed as an argument to the build_matrix function."""
    
    def __init__(self):
        """Instantiate an instance of the Network class."""
        self.matrix = None
        self.node_names = None
        self.graph = None
        self.edgelist = None
        self.rw_graph = None
        self.trim_graph = None
        self.stats = {}

    def build_matrix(self, dataset, metric, org, save=False):
        """Uses one of the distance metrics to build a pairwise similarity matrix for each pair of columns/rows.
        User selects the metric from those available in metrics.py. Pass this function a DataSet object and grab the data."""
    
        # Pull the data from the Dataset object and use build_matrix to construct similarity matrix. 
        data = dataset.data
        self.matrix = build_matrix(data, metric=metric, org=org)
        
        if save == True:
            export_matrix(self.matrix)
        else:
            pass

    def load_matrix(self, matrix_file):
        """Function to load a previously saved matrix (in .npy format) into the Network object."""
        
        if self.matrix == None:
            self.matrix = np.load(matrix_file)
        else:
            raise Exception('This Network object already contains a matrix.')

    def heatmap(self, cmap="plasma"):
        """Uses the heatmap function from visuals to draw a heatmap of the similarity matrix."""

        Visuals.heatmap(self.matrix)
         
    def build_graph(self, name, node_names=None):
        """Converts the similarity matrix object into a networkx graph object.

        Parameters

        ----------

        name: str
            name of reweighted graph
        node_names: dict
            dict of node names 
        """
        
        # Convert the self.matrix into a networkx graph. 
        graph = nx.from_numpy_matrix(self.matrix)

        # Use Pandas datframe slicing to remove self edges between nodes. 
        self.edgelist = nx.to_pandas_edgelist(graph, source='Gene1', target='Gene2')
        self.edgelist = self.edgelist[self.edgelist['Gene1'] != self.edgelist['Gene2']]
        graph = nx.from_pandas_edgelist(self.edgelist, source='Gene1', target ='Gene2', edge_attr='weight')

        self.graph = graph
        self.graph.name = name
        self.stats['weights'] = np.array(self.edgelist['weight'])

        # Rename the graph nodes if the user inputs a dictionary mapping of names to indexes. 
        if node_names != None:
            if type(node_names) == dict:
                nx.relabel_nodes(self.graph, mapping=node_names, copy=False)
                self.node_names=node_names
            else:
                raise Exception('Node names must be a dict mapping node indices to new names.')
        else:
            pass          

    def rename_nodes(self, node_names):
        """Uses networkx to give the graph nodes new names after construction rather than upon construction."""

        if type(node_names) == dict:
            nx.relabel_nodes(self.graph, mapping=node_names, copy=False)
            self.node_names=node_names
        else:
            raise Exception('Node names must be a dict mapping node indices to new names.')

        self.node_names = node_names        

    def quickstats(self):
        """Function to return some quick stats regarding the graph edge weights, etc."""

        self.stats['zscores'] = Stats.zscores(self.stats['weights'])
        self.stats['mad'] = Stats.mad(self.stats['weights'])
        self.stats['robust'] = Stats.robust(self.stats['weights'])
        self.stats['eigenvalues'], self.stats['eigenvectors'] = Stats.eigen(self.matrix)

    @jit 
    def reweight_graph(self, key, threshold=None, name="reweighted_graph", binary=False):
        """Function creates a copy of the graph object and replaces the weights with newly calculated values i.e. Z scores.

        Parameters

        ----------

        key: 1D array
            key for self.stats
        threshold: int
            cutoff to remove edges
        name: str
            name of reweighted graph
        """

        # Make a copy of the graph (rw_graph) and replace edge weights with "key" stats. 
        if self.graph != None:
            self.rw_graph = deepcopy(self.graph)
             
            edges = self.rw_graph.edges(data=True)
            values = self.stats[key]

            for e, v in zip(edges, values):
                e[2]['weight'] = v
            
            # Check if a threshold has been set and then remove edges according to value.
            if threshold != None:

                # Check whether the user wants the graph to be binary or to keep the new weights. 
                if binary == False:
                    trim_weights = []
                    graph_copy = deepcopy(self.rw_graph)

                    for e in self.rw_graph.edges(data=True):
                        if e[2]['weight'] <= -1*threshold or e[2]['weight'] >= threshold:
                            trim_weights.append(e[2]['weight'])
                        else:
                            graph_copy.remove_edge(e[0], e[1])

                    self.trim_graph = graph_copy
                    self.trim_graph.name = "trimmed_graph"
                    self.stats['trim_weights'] = np.array(trim_weights)      

                else:
                    trim_weights = []
                    graph_copy = deepcopy(self.rw_graph)

                    for e in self.rw_graph.edges(data=True):
                        if e[2]['weight'] <= -1*threshold or e[2]['weight'] >= threshold:
                            trim_weights.append(1)
                        else:
                            graph_copy.remove_edge(e[0], e[1])

                    self.trim_graph = graph_copy
                    self.trim_graph.name = "trimmed_graph"
                    self.stats['trim_weights'] = np.array(trim_weights)
        else:
            raise Exception('There is no graph to be re-weighted!')
        
    
def export_graph(G):
    """Uses the built-in writing functions from networkx to save graph objects."""
    
    # Save the entire graph object as a pickle file
    nx.write_gpickle(G, G.name+"_graph.p")

    # Save the edge list with weights
    nx.write_edgelist(G, G.name + '.edgelist', delimiter=',', data=['weight'])




