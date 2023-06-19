import pkg_resources
from himap import clustering
from himap import utils
import multiprocessing
import math
import argparse
import logging
import numpy as np
import pandas as pd
import math

import pytest


def _rf(fn):
    # get path to file from inside himap installation
    f = pkg_resources.resource_filename('himap', 'test/' + fn)
    return f.replace('/himap/test', '/test')


# How to write unit tests with pytest

'''
When a file starts with test_ indicates it is a unit test to python
Test each function with test_(function name)()
every test must include an assert statement
If True, blank output

assert (input) == (ouput)

To run test files written with pytest:
pytest dir_of_test_file/test_*.py

To capture code print statements during run use
pytest -rP dir_of_test_file/test_*.py
'''

@pytest.mark.parametrize('input, expected', [
    (['lig_1', 'lig_2'], ['lig_1', 'lig_2']),
    (['1', '2'], ['lig_1', 'lig_2']),
    (['moo_1', '2'], ['moo_1', 'lig_2']),
])
# Test that lig_ID names are properly renamed so they are not only ints
def test_clean_ID_list(input, expected):
    logging.basicConfig(format='%(message)s', level=logging.CRITICAL)

    
    actual = utils.clean_ID_list(input)
    assert len(actual) == len(expected)
    assert all([a == b for a, b in zip(actual, expected)])


@pytest.mark.parametrize('fn1, arg', [
    (_rf('optimize/sim_scores.csv'), {'IDs': _rf('optimize/mol_names.txt')}),
    (_rf('optimize/sim_scores.csv'), {'delimiter':','}),
    (_rf('optimize/sim_scores.csv'),  {}),
])
# Test reading in data (1) provide lig names, (2) only delimiter, (3) no args
def test_read_data(fn1, arg):
    logging.basicConfig(format='%(message)s', level=logging.CRITICAL)

    
    arr_cleaned, ID_list = utils.read_data(fn1, **arg)
    # Test the shape of the output array
    assert arr_cleaned.shape[0] == 150
    # Test that there are no NaN values
    assert  np.isnan(arr_cleaned).sum() == 0
    # Test the length of the ID_list is the same as the dim of the array
    assert len(ID_list) == arr_cleaned.shape[0]

#----------------------#
# Start of clustering
#----------------------#

class MyError(Exception):
    def __init__(self, m):
        self.m = m

    def __str__(self):
        return self.m

@pytest.mark.parametrize('fn1', [
    (_rf('optimize/sim_scores.csv')),
])
# Test reading in distance data for clustering
def test_k_dist(fn1):
    arr, ID_list = utils.read_data(fn1)
    logging.basicConfig(format='%(message)s', level=logging.CRITICAL)
    
    x, distances = clustering.k_dist(arr)
    assert len(x) == 150
    assert len(x) == len(distances)
    # Check that list is sorted to ascend
    i = 1
    while i < len(distances):
        assert distances[i] >= distances[i-1]
        i += 1
        

# Looking at this function, should I be generating data within the function or should I do that somewhere else and then return the value and insert it?

@pytest.mark.parametrize('fn1', [
    (_rf('optimize/sim_scores.csv')),
])
# Test reading in distance data for clustering
def test_find_shape1(fn1):
    arr, ID_list = utils.read_data(fn1)
    logging.basicConfig(format='%(message)s', level=logging.CRITICAL)
    
    x, distances = clustering.k_dist(arr)
    actual = clustering.find_shape(x, distances)
    y = 1.0 - distances
    actual2 = clustering.find_shape(x, 1-distances)
    assert actual == ('increasing', 'convex')
    assert actual2 == ('decreasing', 'concave')


@pytest.mark.parametrize('fn1', [
    (_rf('optimize/sim_scores.csv')),
])
def dist_data(fn1):
    #fn1 = _rf('optimize/sim_scores.csv')
    arg = {'IDs': _rf('optimize/mol_names.txt')}
    sim_data, ID_list = utils.read_data(fn1, **arg)
    return sim_data, ID_list


@pytest.fixture
def dbscan_data():
    fn1 = _rf('optimize/sim_scores.csv')
    sim_data, ID_list = dist_data(fn1)
    data = 1 - sim_data
    x, dists = clustering.k_dist(data)
    auto_cutoff = clustering.find_max_curvature(x, dists, savefigs=False, verbose=False)
    labels, mask, n_clusters_ = clustering.dbscan(data, dist_cutoff=0.2, min_s = 2)
    sub_arr, sub_ID = clustering.sub_arrays(labels, sim_data, ID_list)
    
    keys=["labels", "sim_data", "sub_arr", "sub_ID", "ID_list"]
    vals=[labels, sim_data, sub_arr, sub_ID, ID_list]

    data_dict={}

    for key, val in zip(keys, vals):
        data_dict[key]=val
    return data_dict
 
 
@pytest.fixture
def clustered_data(dbscan_data):
    data_dict = dbscan_data
    print(data_dict)
    sub_arr, sub_ID = clustering.sub_arrays(data_dict['labels'], data_dict['sim_data'], data_dict['ID_list'])
    
    keys=['sub_arr', 'sub_ID']
    vals=[sub_arr, sub_ID]

    cluster_dict={}

    for key, val in zip(keys, vals):
        cluster_dict[key]=val
    
    return sub_arr, sub_ID


@pytest.mark.parametrize('arg1, arg2, arg3, arg4', [
    ({'clusters2optim' : [0]}, {}, {}, {}),
    ({'clusters2optim' : [0]}, {'ref_ligs' : ['mol_100']}, {}, {}),
    ({'clusters2optim' : [0]}, {'ref_ligs' : ['mol_100']}, {'num_edges' : 'min' }, {}),
    ({'clusters2optim' : [0]}, {'ref_ligs' : ['mol_104']}, {'num_edges' : 'min' }, {'optim_types' : ['negA', 'negD']}),
    ({'clusters2optim' : [0]}, {'ref_ligs' : ['mol_108', 'mol_100']}, {'num_edges' : 'max' }, {'optim_types' : ['negA', 'negD']}),
    ({'clusters2optim' : 'w_ref_lig'}, {'ref_ligs' : ['mol_100']}, {}, {}),
])
def test_clusters2optimize(clustered_data, arg1, arg2, arg3, arg4):
    # Sets the random seed in optimization so that the results are deterministic.
    # Runs through multiple kwarg options for clustering to optimization.
    logging.basicConfig(format='%(message)s', level=logging.CRITICAL)
    
    sub_arr, sub_ID = clustered_data
    assert sub_arr
    assert sub_ID
    try:
        c = clustering.clusters2optimize(sub_arr, sub_ID, random_seed = 123458, **arg1, **arg2, **arg3, **arg4)
    except MyError:
        pytest.fail(f"Failure with parameters {arg1}, {arg2}, {arg3}, {arg4} ")

        
@pytest.mark.parametrize('arg1, arg2, arg3, arg4', [
    ({'clusters2optim' : [0]}, {}, {}, {}),
])
def test_output_df(clustered_data, arg1, arg2, arg3, arg4):
    '''
    Check for the numeric values of the optimized graphs and dataframe
    structure.
    '''
    logging.basicConfig(format='%(message)s', level=logging.CRITICAL)
    
    sub_arr, sub_ID = clustered_data
    assert sub_arr
    assert sub_ID
    
    # Generate what the data should output as.
    Y2 = [30.746343,
          162.430186,
          125.881077,
          30.746343,
          162.430186,
          -79.188656,
          125.881077,
          162.430186,
          -79.188656,
          162.430186,
          125.881077,
          58.816296,
          -107.159299,
          58.816296,
          45.066484,
          58.816296,
          202.138842,
          -107.159299,
          -32.560350,
          45.066484,
          202.138842,
          45.066484
         ]
         
    X1 = [63.118120,
          -40.398402,
          63.118120,
          -40.398402,
          -153.146151,
          -220.258824,
          -220.258824,
          -220.258824
         ]
    SIM_WEIGHT = [1.000000,
                  0.161665,
                  0.341043,
                  1.000000,
                  0.539553,
                  1.000000,
                  0.341043,
                  0.539553,
                  ]
                
    LIGAND_1 = ['mol_103',
                'mol_104',
                'mol_103',
                'mol_104',
                'mol_106',
                'mol_100',
                'mol_100',
                'mol_100'
                ]
         
    try:
        c = clustering.clusters2optimize(sub_arr, sub_ID, random_seed = 123458, **arg1, **arg2, **arg3, **arg4)
        # Check numeric values of the optimized arrays
        output_Y2 = c['Y2']
        output_X1 = c['X1'].iloc[0:7]
        output_SIM_WEIGHT = c['SIM_WEIGHT'].iloc[0:7]
        output_LIGAND_1 = c['LIGAND_1'].iloc[0:7]
        
        assert all([math.isclose(a, b, abs_tol=10**-5) for a, b in zip(Y2, output_Y2)])
        assert all([math.isclose(a, b, abs_tol=10**-5) for a, b in zip(X1, output_X1)])
        assert all([math.isclose(a, b, abs_tol=10**-5) for a, b in zip(SIM_WEIGHT, output_SIM_WEIGHT)])
        assert all([a == b for a, b in zip(LIGAND_1, output_LIGAND_1)])
 
    except MyError:
        pytest.fail(f"Failure with parameters {arg1}, {arg2}, {arg3}, {arg4} ")
