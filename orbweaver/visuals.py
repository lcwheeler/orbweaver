import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy

class Visuals:
    """Class holding a variety of visualization methods for networks and network attributes/statistics."""

    def __init__(self):
        """Instantiate the Visuals class, which contains some simple plotting functions."""
    
    def heatmap(matrix, cmap="plasma"):
        """Uses seaborn to draw a heatmap of the similarity matrix."""

        sns.heatmap(matrix, cmap=cmap)

    def quickdraw(graph, weights, save=False):
        """Function to draw a simple network diagram of a Network graph object.

        Parameters

        ----------

        graph: networkx graph object
            graph to be drawn
        weights: numpy array
            weights to color edges
        save: bool
            whether or not to save graph image
        """

        figure = nx.draw_spring(graph, edge_color=weights, width=4, with_labels=True, edge_cmap=plt.cm.plasma)
        
        if save == True:
            figname = str(graph.name)
            plt.savefig(figname + ".pdf", format="PDF")

        # Generate a colorbar to display alongside network graph.
        if weights.max() != weights.min():
            a = np.sort(weights)
            a = np.expand_dims(a, axis=0) 
            R = np.linspace(np.min(weights), np.max(weights), num=10, endpoint=True)
            Rr = R.round(2)
            fig, ax1 = plt.subplots(1,1)
            ax1.imshow(a, cmap="plasma", interpolation='nearest')
            ax1.set_xticklabels(Rr, color="black")
            ax1.set_yticklabels('y', color="white")
            ax1.yaxis.set_ticks_position('none')
        else:
            pass

    def scatter(network, keyx, keyy):
        """Use matplotlib.pyplot to generate a simple scatter plot of Network object attributes."""

        plt.scatter(net.stats[keyx], net.stats[keyy])
        plt.xlabel(keyx)
        plt.ylabel(keyy)
        plt.title("Plot of"+keyx+"vs."+keyy)


    def outlier_plot(network, save=False):
        """Generate a scatter plot of the weights vs. robust outlier metric.

        Parameters

        ----------

        network: orbweaver.Network() object
            Network object for plot
        save: bool
            whether or not to save plot
        """    

        if 'weights' in network.stats.keys() and 'robust' in network.stats.keys():
            if save == False:
                plt.scatter(network.stats['weights'], network.stats['robust'])
                plt.axhline(y=np.median(network.stats['robust']), color='Red')
                plt.xlabel("Weights")
                plt.ylabel("Robust outlier metric")
                plt.title("Outlier detection of edge weights")
            else:
                plt.scatter(network.stats['weights'], network.stats['robust'])
                plt.axhline(y=np.median(network.stats['robust']), color='Red')
                plt.xlabel("Weights")
                plt.ylabel("Robust outlier metric")
                plt.title("Outlier detection of edge weights")
                plt.savefig("robust_scatter.pdf", format="PDF")

        else:
            raise Exception('this method requires the Network to have weights and robust stats.')
                






