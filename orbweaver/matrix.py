import numpy as np
import scipy.spatial.distance
import itertools as it
from .metrics import Metrics
from numba import jit 

@jit #This decorator indicates numba will be used to wrap the function and speed up calculation. 
def build_matrix(data, metric, org):
    """Uses one of the distance metrics to build a pairwise similarity matrix for each pair of columns/rows.
    User selects the metric from those available in metrics.py. Pass this function a DataSet object and grab the data.

    Parameters

    ----------

    data: pandas.DataFrame
        dataframe holding expression data
    metric: function
        similarity metric from orbweaver.metrics
    org: str
        str to specificy genes in "columns" or "rows" 
    """


    #Use itertools to iterate over pairs of gene expression vectors. 
    # Perform operation appropriate to organization (org) of input data. User specifices genes as rows or columns. 
    if org == "columns":
        dim = len(data.columns)
        matrix = np.zeros((dim, dim))
        for i, j in it.combinations(range(dim), 2):
            sim = metric(np.array(data[i]), np.array(data[j]))
            matrix[i,j] = sim   
    
    elif org == "rows":
        dim = len(data)
        matrix = np.zeros((dim, dim))
        for i, j in it.combinations(range(dim), 2):
            sim = metric(np.array(data.loc[i,]), np.array(data.loc[j,]))
            matrix[i,j] = sim
            
    else:
        raise Exception('Please specify whether genes are "rows" or "columns" of dataframe.')


    # If the chosen metric is canberra or minkowski distance, these need to be normalized first before converting to sims. 
    if metric == Metrics.canberra or metric == Metrics.euclidean or metric == Metrics.manhattan:
        matrix = 1 - (matrix/matrix.max())
    else:
        pass

    # Reflect the upper triangle across the matrix to fill the lower triangle, making it symmetric. Set self sims to 1. 
    i_lower = np.tril_indices_from(matrix)
    matrix[i_lower] = matrix.T[i_lower]
    np.fill_diagonal(matrix, val=1)

    return matrix


def export_matrix(matrix):
    """Function to export the computed distance matrix as a csv file and a .npy binary file."""

    fname = "matrix"
    np.savetxt(fname+".csv", matrix, delimiter=',', newline='\n')
    np.save(fname, matrix, allow_pickle=False)

    
