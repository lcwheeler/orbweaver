import numpy as np
import scipy.spatial.distance
import scipy.stats
from numpy.linalg import norm
#from astropy.stats import biweight_midcorrelation 

class Metrics:
    """Class holding all available similarity metrics used for building similarity matrices."""
    
    @classmethod
    def __init__(self):
        """Instantiate an instance of the Metrics class."""

    @classmethod
    def pearson(u,v):
        """Compute pearson correlation between two 1D arrays (gene expression vectors)."""
        
        return 1 - scipy.spatial.distance.correlation(u, v, w=None, centered=True)

    @classmethod
    def spearman(u,v):
        """Compute spearman correlation between two 1D arrays (gene expression vectors)."""

        r, p = scipy.stats.spearmanr(u, v)
        return r

    @classmethod
    def tanimoto(u,v):
        """Compute the Tanimoto similarity metric for two 1D arrays."""

        return np.dot(u,v)/(norm(u)**2 + norm(v)**2 - np.dot(u,v))

    @classmethod
    def cos_sim(u,v):
        """Compute cosine distance between two 1D arrays (gene expression vectors)."""

        return np.dot(u,v)/(norm(u)*norm(v))

    @classmethod        
    def euclidean(u,v):
        """Compute Minkowski distance between two 1D arrays (gene expression vectors), gets converted to sim in matrix."""

        return scipy.spatial.distance.minkowski(u, v, p=2, w=None)

    @classmethod
    def manhattan(u,v):
        """Compute Minkowski distance between two 1D arrays (gene expression vectors), gets converted to sim in matrix."""

        return scipy.spatial.distance.minkowski(u, v, p=1, w=None)

    @classmethod
    def canberra(u,v):
        """Compute canberra distance between two 1D arrays (gene expression vectors), gets converted to sim in matrix."""

        return scipy.spatial.distance.canberra(u, v, w=None)

    #@classmethod
    #def bicorr(u,v):
        #"""Compute biweight midcorrelation between two 1D arrays (gene expression vectors)."""

        #return biweight_midcorrelation(u, v)




