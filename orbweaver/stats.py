import numpy as np
import scipy.stats 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class Stats:

    def __init__(self):
        """Instantiate an instance of the Stats class."""

    def zscores(weights):
        """Function returning Z-scores for a 1D array of values, i.e. weights of the graph edges."""

        # Use scipy.stats to calculate zscores
        z = scipy.stats.zscore(weights) 
        return z
        
    def mad(weights):
        """Function to return the median absolute deviation (MAD) of edge weights."""

        med = np.median(weights)
        mad = np.median(np.fabs(weights-med))
        return mad

    def robust(weights):
        """Function to return a more robust method for detecting outliers based on the median 
        absolute deviation (MAD) of edge weights. """

        med = np.median(weights)
        mad = np.median(np.fabs(weights-med))
        outlyingness = np.fabs((weights-med))/mad

        return outlyingness

    def eigen(matrix):
        """Use numpy.linal.eig to calculate eigenvalues and right eigenvectors of sim matrix."""

        w, v = np.linalg.eig(matrix)
        return w, v     

    def svd(matrix):
        """Use numpy.linal.eig to calculate eigenvalues and right eigenvectors of sim matrix."""

        U, s, V = np.linalg.svd(matrix, full_matrices=True)
        return U, s, V    

