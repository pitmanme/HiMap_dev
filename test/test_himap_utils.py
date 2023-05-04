import pkg_resources
from himap import utils
import logging
import numpy as np
import pytest

'''
How to write unit tests with pytest:

When a file starts with test_ indicates it is a unit test to python
Test each function with test_(function name)()
every test must include an assert statement
If True, blank output

assert (input) == (ouput)

To run test files written with pytest:
pytest dir_of_test_file/test_*.py
'''

def _rf(fn):
    # get path to file from inside himap installation
    f = pkg_resources.resource_filename('himap', 'test/' + fn)
    return f.replace('/himap/test', '/test')


def test_clean_NaN():
    from numpy import nan
    output = utils.clean_NaN([nan, nan, nan])
    check = [0.0, 0.0, 0.0]
    for i in [0, 1, 2]:
        assert output[i] == check[i]
       

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


@pytest.mark.parametrize('input, expected', [
    ("1, 2, 3", "1  2  3"),
    ("(1)2 3", " 1 2 3"),
    ("12, 100, 1", "12  100  1"),
])
# Test string replacement with multidelimiter
def test_multi_delim(input, expected):
    logging.basicConfig(format='%(message)s', level=logging.CRITICAL)
    
    actual = utils.multi_delim(input)
    assert actual == expected


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


# Test is random similarity scores generated are symmetric
def test_rand_sim_scores():
    b = utils.rand_sim_scores(5)
    print(b)
    print(b.T)
    assert b.all() == b.T.all()
