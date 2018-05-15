import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings

class Dataset(object):
    
    """Object class that holds a dataset as a pandas dataframe. Dataset can be imported from a file or assigned from \
    a preloaded dataframe."""
    
    def __init__(self):
        """Instantiate an instance of the Dataset class."""
        self.name = None
        self.filename = None
        self.data = None
        
    def load_data(self, filename, ftype, name):
        """Read the data csv/tsv file and hold the data as an attribute of the object."""

        # Record the data file name and give the Dataset object instance itself an internal name. 
        self.filename = filename
        self.name = name
        
        if ftype == "csv":
            self.data = pd.read_csv(self.filename, sep=",", header=None)
        elif ftype == "tsv":
            self.data = pd.read_csv(self.filename, sep="\t", header=None)
        else:
            raise Exception('File type is not supported. Please specify as either "csv" or "tsv".')

        if self.data.any() < 0:
            warnings.warn('There are negative values in your expression data.')
        if self.data.isnull().values.any():
            warnings.warn('There are values missing from your expression data.')

    def assign_data(self, df, name):
        """Allows a pre-loaded pandas dataframe to be assigned to self.data inside the Dataset object."""

        # Protect the object from having its name overwritten if it already exists.
        if self.name == None:
            self.name = name
        else:
            raise Exception('This Dataset object already has a name.') 

        # Protect the object from having its data overwritten if it already exists. Otherwise assign the desired data. 
        if self.data == None:
            if type(df) != pd.core.frame.DataFrame:
                raise Exception('Incorrect data structure. Please assign a pandas dataframe.')
            else:
                self.data = df.copy(deep=True)

        else:
            raise Exception('This Dataset object already contains data!')

        if self.data.any() < 0:
            warnings.warn('There are negative values in your expression data.')
        if self.data.isnull().values.any():
            warnings.warn('There are values missing from your expression data.')


