## Contents of this file

 * HiMap Introduction
 * Requirements
 * Authors
 * Installation
 * Usage
 * Troubleshooting

HiMap Introduction
-------

HiMap includes design generation based on statistical optimality. 
Alchemical free energy calculations hold increasing promise 
as an aid to drug discovery efforts. However, applications of 
these techniques in discovery projects have been relatively 
rare, partly because of the difficulty of planning and setting up 
calculations. The lead optimization mapper (LOMAP) was 
introduced as an automated algorithm to plan relative 
free energy calculations between potential ligands. LOMAP was further
developed to be based on free, open-source APIs such as RDKit. HiMap
now includes clustering of ligand series, and optimization of free
energy perturbation networks. 

Requirements
-------

* Lomap2
* Matplotlib 
* python >= 3.8
* R
* rpy2=3.4.5
* kneed=0.7.0
* scikit-learn
* scipy
* numpy<1.24


Authors
-------

* Mary Pitman <mpitman@uci.edu>
* David Mobley <dmobley@uci.edu>
    

Installation
-----

To install HiMap with LOMAP included, build the conda environment and install from file:

[https://github.com/pitmanme/HiMap_dev/blob/main/devtools/conda-envs/himap_env.yml]

with:

`conda env create -f himap_env.yml`


Usage
-----
#### Example scripts are included for various purposes:
* To run HiMap with optimization \
    `python examples/example_optimize.py`
* To read in scores and optimize \
    `python examples/example_optimize_read_data.py`



#### If you would rather use the API directly:
* To generate optimal designs, try:

cd examples/

```python
import lomap
import himap

#-------------------------------------------------------#
# Generate similarity scores.
#-------------------------------------------------------#
# Read molecules from test directory.
db_mol = lomap.DBMolecules('../test/radial/', output=True, radial=True)
    
# Generate the strict and loose symmetric similarity score matrices.
strict, loose = db_mol.build_matrices()
    
# Convert the similarity matrix to numpy array
sim_np = strict.to_numpy_2D_array()

# Clean data if Lomap produces rare error. If score is NaN, replace with 0.0
n_arr = himap.clean_NaN(sim_np)

#-------------------------------------------------------#
# Clustering.
#-------------------------------------------------------#
# Create ID_list from db_mol prior to clustering.
ID_list = himap.db_mol_IDs(db_mol, n_arr)

# Perform clustering.
#   sub_arr, sub_ID:   the n_arr and ID_list subdivided by clusters
#   selected_clusters: user selected clusters during interaction.
sub_arr, sub_ID, selected_clusters = himap.cluster_interactive(n_arr, ID_list)

#-------------------------------------------------------#
# Optimization.
#-------------------------------------------------------#
# Example reference ligand.
ref_ligs = ['ejm_31']

# Send the user selected clusters for optimization.
himap.clusters2optimize(sub_arr, sub_ID, clusters2optim = selected_clusters, ref_ligs=ref_ligs)
```


* To generate optimal designs using external scores or weights, try:

```python
import himap

#-------------------------------------------------------#
# Define input files, read data.
#-------------------------------------------------------#
# Input files for weight scores and ligand names.
sim_scores_in = '../test/optimize/sim_scores.csv'
IDs_in = '../test/optimize/mol_names.txt'

# Read files, clean any potential NaN scores.
#   Added optional parameter:
#             delimiter: default is ','
n_arr, ID_list = himap.read_data(sim_scores_in, IDs = IDs_in)

#-------------------------------------------------------#
# Clustering.
#-------------------------------------------------------#
# Perform clustering.
#   sub_arr, sub_ID:   the n_arr and ID_list subdivided by clusters
#   selected_clusters: user selected clusters during interaction.
sub_arr, sub_ID, selected_clusters = himap.cluster_interactive(n_arr, ID_list)

#-------------------------------------------------------#
# Optimization.
#-------------------------------------------------------#
# Example reference ligands.
ref_ligs = ['mol_0', 'mol_1', 'mol_2', 'mol_3', 'mol_4']

# Send the user selected clusters for optimization.
himap.clusters2optimize(sub_arr, sub_ID, clusters2optim = selected_clusters,
                        ref_ligs=ref_ligs, num_edges = '2n', optim_types = ['A', 'D']
                        )
```

