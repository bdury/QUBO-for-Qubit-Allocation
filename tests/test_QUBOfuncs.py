'''
Contains the unit tests for the various functions used in this project.
'''

import pytest
import networkx as nx

import sys
sys.path.insert(0, 'C:/work/qubitallocation/src/') # change depending on home directory for fork
from QUBOfuncs import *
from matrix_funcs import *
from coef_matrix import *
from datalog import*

# Unit tests for functions, comments describe reason for tests when necessary

def test_qasmcleaner():
    # Correctly interprets double-digits
    assert qasmcleaner('q[10];') == 10
    # Correctly interprets triple-digits
    assert qasmcleaner('q[100];') == 100
    # Can handle numbers below 10
    assert qasmcleaner('q[1];') == 1
    # Ignores leading 0's
    assert qasmcleaner('q[01]') == 1

def test_mapind():
    # All just confirm that indices are being mapped to correct numbers
    assert mapind(2,2,3) == 8
    assert mapind(7,2,3) == 23
    assert mapind(0,2,30) == 2

def test_makepair():
    # Check basic functionality (needs to be able to do this)
    assert makepair([2,3,4,5,6]) == [(2,3),(3,4),(4,5),(5,6)]
    # Behaviour with negatives (doesn't care)
    assert makepair([2,-1,-3]) == [(2,-1),(-1,-3)]
    # Mixed data type (doesn't care)
    assert makepair([2,'lft',int(2.4), [3]]) == [(2,'lft'),('lft',2),(2,[3])]
    # Edge case - properly recreates single pairs
    assert makepair([2,3]) == [(2,3)]

# ****************************************************************************
graph = nx.Graph()
mel_edges = [[1,0], [1,2], [2,3], [4,3], [4,10], [5,4], [5,6], [5,9], [6,8],
             [7,8], [9,8], [9,10], [11,10], [11,3], [11,12], [12,2], [13,1],
             [13,12], [14,0], [14,13]
             ]
graph.add_edges_from(mel_edges)
paths = nx.shortest_path(graph)
# ****************************************************************************

def test_distance():
    # Neighbouring nodes should be distance of 1
    assert distance(paths, 0, 1) == 1
    # Single node specifications should point to 1 to ignore interaction
    assert distance(paths, 0, 0) == 1
    # Check a long path
    assert distance(paths, 0, 7) == 8

# ****************************************************************************

n_p = len(graph)
err_graph = 'C:/work/qubitallocation/IBM_qc_csv/ibmq_16_melbourne(march-30).csv'

# ****************************************************************************

err1, err2 = read_err(err_graph, n_p)

def test_read_err():
    # Grabs the right single qubit error
    assert round(err1[5], 5) == 5.01E-03
    # Grabs the right two-qubit error
    assert round(err2[(0,14)], 5) == 2.429e-2
    # Make sure err2 single qubit values are set to 1
    assert err2[(4,4)] == 1

# ****************************************************************************

loc = 'C:/work/qubitallocation/exp_circs/misex1_241-15_qbits_5k_gates_mix.qasm'

# ****************************************************************************

qbit, gate = circ_properties(loc)

def test_circ_properties():
    # Check if it works
    assert qbit == 15
    assert gate == 4817 - 4

# ****************************************************************************
graph = nx.Graph()
mel_edges = [[1,0], [1,2], [2,3], [4,3], [4,10], [5,4], [5,6], [5,9], [6,8],
             [7,8], [9,8], [9,10], [11,10], [11,3], [11,12], [12,2], [13,1],
             [13,12], [14,0], [14,13]
             ]
graph.add_edges_from(mel_edges)
n_p = len(graph)
err_graph = 'C:/work/qubitallocation/IBM_qc_csv/ibmq_16_melbourne(march-30).csv'
# ****************************************************************************

err1, err2 = read_err(err_graph, n_p)

def test_swap_err():
    # Test neighbour error
    assert swap_err(1, 2, graph, err2) == err2[(1,2)]
    # Check a longer swap case
    assert swap_err(3, 6, graph, err2) == err2[(3,4)]*3 + err2[(4,5)]*3 + err2[(5,6)]
    # Check reverse
    assert swap_err(6, 3, graph, err2) == err2[(6,5)]*3 + err2[(5,4)]*3 + err2[(4,3)]

# ****************************************************************************
t_dict = {0 : {'h' : 3, 't' : 11, 'z' : 0},
          1 : {'h' : 2, 't' : 0, 'z' : 3},
          (0,1) : {'cnot' : 15, 'swap' : 2}
         }
# ****************************************************************************

def test_gatecount():
    # test basic functionality (adds up vals in sub0-dict.)
    assert gatecount(t_dict) == {0 : 14, 1 : 5, (0,1) : 17}

# ****************************************************************************
t_dict2 = {(2,3) : 4,
           (2,6) : 2,
           (6,1) : 7,
           (1,6) : 9,
           (3,2) : 1,
           (6,2) : 0
          }
# ****************************************************************************

def test_makesymmetric():
    # check basic functionality - need to have flip of pair in dict
    # for function to work (can't just have (2,3), need (3,2) in there as well)
    assert makesymmetric(t_dict2) == {(2,3) : 5,
                                      (2,6) : 2,
                                      (6,1) : 16,
                                      (1,6) : 16,
                                      (3,2) : 5,
                                      (6,2) : 2
                                      }

# ****************************************************************************
