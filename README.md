# orbweaver
Orbweaver is a python package for performing gene co-expression network analysis

Orbweaver is a simple python API for analyzing gene expression datasets. It's a pet project and it's currently still under development. Pull requests are welcome. However, this package is low priority, so I may not be extremely efficient at responding to contributions. Orbweaver relies heavily on the wonderful [Networkx](https://networkx.github.io/) API. It is thus compatible with the wide range of network analysis tools available in Networkx. Other key dependencies are listed in the setup.py. A demonstration of the current Orbweaver workflow can be found in the *notebooks/demo.ipynb* Jupyter notebook. 

## Current features include: 

1. Straightforward API for handling datasets, matrices, and network objects. 

2. Multiple similarity metrics for comparing gene expression profiles.

3. Simple built-in visualizations for similarity matrices and networks. 

4. Functions for re-weighting graphs and converting from weighted to unweighted. 

5. Exporting of network objects in various formats. 

## Future features (in development) will include:

1. Simulation of networks and data with specified structure and co-expression relationships. 

2. Improved network visualization. 

3. Tools for handling uncertainty across replicate datasets. 

4. Built-in approaches for data clustering. 

5. Functions for caculating certain aspects of network topology. 

## Install

Clone this repository and install a development version using `pip`:
```
pip install -e .
```
