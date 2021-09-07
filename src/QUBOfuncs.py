"""
Contains functions for implementing qubit mapping as a QUBO problem.

"""
import numpy as np
import networkx as nx
import dimod
import neal as nl
import timeit
from datetime import datetime
from re import sub
from copy import deepcopy
from itertools import product
from pandas import read_csv

from os import listdir
from os.path import join

# from qiskit.transpiler import CouplingMap
# from qiskit.transpiler.passes import BasicSwap
from qiskit import QuantumCircuit
# from qiskit.compiler import transpile
# from qiskit.transpiler import PassManager

import matplotlib.colors
import matplotlib as mpl
import matplotlib.pyplot as plt

from numpy import random as rd
from random import sample
from pathlib import Path



def createcircuit(qbit, g1num, g2num):
    '''
    Function to create circuits for  arbitrary architectures, where we don't
    care about the functionality since we're just testing capability of QUBO.
    '''

    # Create quantum circuit object for qbit number of qubits
    qc = QuantumCircuit(qbit)

    # Generate random one & two qubit gate assignments
    g1 = rd.randint(0,qbit, g1num).tolist() # evenly (randomly) distr. single qubit gates
    g2 = []
    for _ in range(g2num):
        q1,q2 = sample(range(qbit), 2)
        g2.append((q1,q2))

    # Put random gate assignments into Quantum Circuit Object
    for gate in g1: qc.h(gate)
    for gate in g2: qc.cx(gate[0], gate[-1])

    print(qc.qasm())
    # Export circuit into qasm file and save to file
    loc = str(Path().resolve().parent / 'benchmarks' / '_newcirc')
    print(loc)
    with open(loc + "/{}QBT_{}g1_{}g2.txt".format(qbit,g1num,g2num), "w") as text_file:
        text_file.write(qc.qasm())


def grab_tket(arr, data=None):
    '''
    Given a tket data set, will extract a metric according to data string to a new array and return it.
    '''
    arr_cx = np.zeros(arr.size, dtype=int)

    for count, vals in enumerate(arr):
        if data == 'cx':
            arr_cx[count] = vals[0]
        elif data == 'swap':
            arr_cx[count] = vals[1]
        elif data == 'depth':
            arr_cx[count] = vals[2]
        else:
            print('Unrecognized data label, please check your input.')
    return arr_cx


def grab_qiskit(arr, data=None):
    '''
    Given a qiskit data set, will extract a metric according to data string to a new array and return it.
    '''
    if data == 'prob':
        arr_cx = np.zeros(arr.size, dtype=float)
    else:
        arr_cx = np.zeros(arr.size, dtype=int)

    for count, vals in enumerate(arr):
        if data == 'cx':
            arr_cx[count] = vals[0]
        elif data == 'swap':
            arr_cx[count] = vals[1]
        elif data == 'depth':
            arr_cx[count] = vals[2]
        elif data == 'prob':
            arr_cx[count] = vals[3]
        else:
            print('Unrecognized data label, please check your input.')

    return arr_cx


def trim(prop_arr, arr):
    '''
    Given a property array and either a qiskit/tket data set, trims the property array down
    to just contain the circuits used in arr
    '''

    # Sort the arrays just in case
    prop_arr.sort(order='circuit')
    arr.sort(order='circuit')

    ind = []
    for count, circ in enumerate(prop_arr['circuit']):
        if circ in arr['circuit']:
            ind.append(count)

    return ind


def qasmregfixer(loc, n_p):
    '''
    This function will change all the text files in a specified directory to
    remove the creg line and make the qreg # = # physical qubits that you
    specify.
    '''
    # Grab all files in specified location
    ls_circ = listdir(loc)

    # Remove anything that doesn't end with .qasm in our list
    ls_circ = [file for file in ls_circ if file.split('.')[-1] == 'qasm']

    # Loop over all the circuits
    for circ in ls_circ:

        # Load line data of circuit
        with open(join(loc, circ), 'r') as file:
            lines = file.readlines()

        # Get number of logical qubits in circuit
        n_c, g1, g2 = circ_properties(join(loc, circ))

        # Change qreg number and add creg
        lines[2] = 'qreg q[{}];\n'.format(n_c, n_p) # creg c[{}];\n
        if 'creg' in lines[3]:
            lines[3] = ''

        # Open file with write permissions
        with open(join(loc, circ), 'w') as file:
            # write changes to file
            file.writelines(lines)


def qasmcleaner(string):
    '''
    Expecting a line of qasm code that has been reduced per this example:

    1-qubit line: t q[11]; -> q[11];
    2-qubit line: cx q[9],q[12]; -> q[9] & q[12]; (call on each separately)
    '''
    # The 'reg' is added to deal with accidently labelling some file qubits
    # like 'qreg[2]' instead of 'q[2]'
    return int(string.translate({ord(i): None for i in 'q[];reg'}))


def mapind(i, j, n_p):
    '''
    Takes the indices (i,j) from a matrix (m x n) and maps them to a single
    (unique) index. We assume i,j are indexed from 0 and that the new index
    will start from 0. Note that we use n_p here because it is the column index
    for the assignments (n in the above dimension).

    General form: (0 indexing for (i,j) & z)

    map(i,j) = i*n + j
    '''
    return i*n_p + j


def makepair(lst):
    '''
    Given a list as input, returns a list of tuples where each tuple is of the form:

    (list[n], list[n+1])

    for all values in the input list.

    '''
    tmp = []

    for count, i in enumerate(lst):
        if count + 1 > len(lst) - 1: # could also do >= len(lst) for truncate
            return tmp

        tmp.append((i, lst[count + 1]))


def sym(array):
    '''
    Simple function to test whether or not a numpy array is symmetric. Returns
    True if the array is symmetric, False otherwise.
    '''
    return (array.T == array).all()


def distance(paths, v1, v2):
    '''
    Returns the graph distance between two vertexes. If multiple paths exist
    between nodes, the distance is given as the shortest path between the
    specified vertices.

    paths:
        An instance of networkx.shortest_path(your_quantum_architecture_graph),
        where the architecture graph was given as a bidirected graph.

    '''
    # Calc. distance
    dist = len(paths[v1][v2]) - 1

    # If 0, then paths was given the same node for both vertices, so we return
    # '1' as to ignore the 'interaction'
    if dist == 0:
        return 1
    # Otherwise we can just return the distance
    else:
        return dist


def assignment(response, n_c, n_p):
    '''
    Using the response object from the sampling of a neal simulated anneal,
    returns a list of tuples where each tuple has the following entries:

        1. a dictionary of allocation dictionaries that are formated as:

            { 'phys' : {hardware_qubit_n : logical_qubit_m},
              'circ' : {logical_qubit_m : hardware_qubit_n}
            }

            where you can specify which order you want with 'circ' or 'phys', referring
            to the type of qubit label for the dictionary keys.

        2. an allocation array:

            [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
            ]

            where '1' means that logical qubit i was allocated to physical qubit j.

        3. the energy of that allocation
        4. the number of times that specific allocation was found

    response:
        Must be a sampleset from a bqm sampler.

    n_c & n_p:
        Number of circuit and physical qubits respectively.
    '''
    assignments = []
    # loop over each anneal result in response
    for map, energy, num in response.aggregate().data():
        allocate_array = np.asarray(list(map.values()))
        allocate_array.resize(n_c, n_p)
        row_ind, col_ind = np.where(allocate_array == 1)

        circ_phys = {circ : phys for circ, phys in zip(row_ind,col_ind)}
        phys_circ = {phys : circ for circ, phys in zip(row_ind,col_ind)}

        assignments.append(({'circ':circ_phys,'phys':phys_circ}, allocate_array, energy, num))

    return assignments


def gatecount(gate_dic):
    '''
    Counts up the one/two qubit gate operations present in a dict. of the form
    returned by 'qasm_parse' function (see function for details on structure).

    Outputs dict. of the form:

    {qubit label : # of gate ops, ... , etc.}

    Works for both one_gates/two_gates dicts.
    '''
    gate_count = gate_dic.copy() # retain all initial labels (qubit #'s)
    for label in gate_dic:
        gate_count[label] = sum(gate_dic[label].values()) # update values

    return gate_count


def makesymmetric(gate2_dic):
    '''
    Taking a 2-gate dictionary as input, makes any 2-qubit interaction pair
    symmetric.

    i.e. (2,3) should be the same thing as (3,2), what you get with a bidirected
    architecture graph.
    '''
    gate_list = list(gate2_dic)
    for qbits in gate_list:
        flip = (qbits[1], qbits[0])

        if gate_list.index(qbits) > gate_list.index(flip): # reach pair a second time
            gate2_dic[qbits] = gate2_dic[flip]             # so just update to first value

        else: # first time seeing a pair
            gate2_dic[qbits] += gate2_dic[flip]

    return gate2_dic


def circ_properties(loc):
    '''
    Read over the specified qasm file and return some properties of it.
    '''
    # Initialize qubit and gate variables
    num_qbits = 0
    one_gates = 0
    two_gates = 0
    depth = QuantumCircuit.from_qasm_file(loc).depth()

    # Go over each line and update num_qbits to the highest number it finds
    with open(loc, newline='\n', mode='r+') as qasm_tmp:
        for line in qasm_tmp.readlines()[3:]: # first 3 lines shouldn't contain gate ops.

                # split string into gate operation and qubit specification
            if len(line.split()) > 2: # queko benchmarks have an extra space we need to deal with
                line_list = line.split()
                op, labels = line_list[0], line_list[1] + line_list[2]
            else:
                op, labels = line.split()

            if ',' in labels: # 2 qbit operation
                two_gates += 1 # count op

                qbits = labels.split(',')
                qbit1, qbit2 = qasmcleaner(qbits[0]), qasmcleaner(qbits[1]) # grab qubit # from strings

                if num_qbits < qbit1 or num_qbits < qbit2: # update condition
                    num_qbits = max(qbit1, qbit2)

            else: # 1 qbit operation:
                one_gates += 1

                qbit = qasmcleaner(labels)

                if num_qbits < qbit: # update condition
                    num_qbits = qbit

        # 0-indexed from loop, add 1 to fix
        num_qbits += 1

    # Print the properties
    # print('Circuit properties:', '\n',
    #       '# Qubits ', num_qbits, '\n',
    #       '# Gates ', two_gates+one_gates, '\n',
    #       )

    return num_qbits, one_gates, two_gates, depth


def qasm_parse(fname, n_c):
    '''
    Creates two dictionaries of the form:

            {qubit 0 : {op1 : # op1's, op2 : # op2's, ... , opN : # opN's },
             qubit 1 : {op1 : # op1's, op2 : # op2's, ... , opN : # opN's },
             .
             .
             .
             (qubit M : {op1 : # op1's, op2 : # op2's, ... , opN : # opN's }
            }

    where qubit 0 - > M is the logical qubit and op 1 - > N is the different
    types of gate operations it saw in the circuit, with a dict value of how
    many times it appeared in the circuit. The above example is for the 1-gate
    operations. The only difference between the 1-gate and 2-gate dict's is that
    we specify pairs of qubits for the 2-gate dict.

    fname:
        Filepath to qasm file.

    n_c:
        Number of circuit qubits used in the quantum circuit QASM file given
        in fname.

    '''
    # initalize the structure of the gate dictionaries with a base gate list
    # to be updated during parsing of qasm file

    # Add additional ops to these as new ones encountered (manually)
    one_gate_ops = {'h' : 0,
                    't' : 0,
                    'tdg' : 0,
                    'x' : 0,
                    'rz': 0,
                    's' : 0,
                    'z' : 0
                    }

    two_gate_ops = {'cx' : 0}
    its = [range(n_c)]*2

    one_gates = {i : one_gate_ops.copy() for i in range(n_c)}
    two_gates = {(i,j) : two_gate_ops.copy() for i,j in product(*its)}

    # read through qasmfile and append gate operations and how many of them
    # there were to the gate dictionaries
    with open(fname, newline='\n', mode='r+') as qasm_tmp:
        for line in qasm_tmp.readlines()[3:]: # first 3 lines shouldn't contain gate ops.

            # split string into gate operation and qubit specification
            if len(line.split()) > 2: # queko benchmarks have an extra space we need to deal with
                line_list = line.split()
                op, labels = line_list[0], line_list[1] + line_list[2]
            else:
                op, labels = line.split()

            if ',' in labels: # 2 qbit operation
                qbits = labels.split(',')
                qbit_left, qbit_right = qasmcleaner(qbits[0]), qasmcleaner(qbits[1]) # grab qubit # from strings
                two_gates[(qbit_left, qbit_right)][op] += 1

            else: # 1 qbit operation:
                qbit = qasmcleaner(labels)

                if '(' in op: # some gates specify rotation angle, ignore with this
                    op, misc = op.split('(')

                one_gates[qbit][op] += 1

    return one_gates, two_gates


def parse_transpile(qcirc, n_p):
    '''
    Does same thing as qasm_parse, but starts from a qasm string and assumes
    the circuit has been transpiled already by qiskit, meaning that the qubits
    labels refer to physical qubits rather than logical.

    qcirc:
        QuantumCircuit object after being run through pass manager.

    n_p:
        Number of qubits on the architecture graph.

    '''
    # initalize the structure of the gate dictionaries with a base gate list
    # to be updated during parsing of qasm string

    one_gate_ops = {'h' : 0,
                    't' : 0,
                    'tdg' : 0,
                    'x' : 0,
                    'rz': 0,
                    's' : 0,
                    'z' : 0
                    }

    two_gate_ops = {'cx' : 0,
                    'swap' : 0
                    }
    its = [range(n_p)]*2
    one_gates = {i : one_gate_ops.copy() for i in range(n_p)}
    two_gates = {(i,j) : two_gate_ops.copy() for i,j in product(*its)}

    # read through qasm string and append gate operations and how many of them
    # there were to the gate dictionaries
    for line in qcirc.qasm().splitlines()[4:]:

        op, labels = line.split() # split string into gate operation and qubit specification

        if ',' in labels: # 2 qbit operation
            qbits = labels.split(',')
            qbit_left, qbit_right = qasmcleaner(qbits[0]), qasmcleaner(qbits[1]) # grab qubit # from strings
            two_gates[ (qbit_left, qbit_right) ][op] += 1

        else: # 1 qbit operation:
            qbit = qasmcleaner(labels)

            if '(' in op: # some gates specify rotation angle, ignore with this
                op, misc = op.split('(')

            one_gates[qbit][op] += 1

    return one_gates, two_gates


def swap2cx(g2):
    '''
    Converts swap gates in a two-gate dictionary into cx's and updates the dict
    in place.
    '''
    for pair in g2.copy():
        g2[pair]['cx'] += 3*g2[pair]['swap']
        del g2[pair]['swap']


def anneal(r, Q, num_samples = 1000, seed = None):
    '''
    Preforms all the necessary operations to execute simulated annealing on a
    given configuration of the cost function (i.e. r & Q dicts.). Also records
    how long a given anneal job took (total time for all samples).

    r & Q:
        Must be the dictionaries returned by "bqm_dicts" method under the
        "Coefficient_Matrix" class.

    num_samples:
        Number of samples of the SimulatedAnnealingSampler to be run.
        Will default to 1000 if nothing is given.

    seed:
        If True will seed the SimulatedAnnealingSampler so results are
        repoducible.
    '''
    # Initialize the given coefficient matrix encoded by r & Q as a QUBO
    # and create a SimulatedAnnealingSampler object
    bqm = dimod.BinaryQuadraticModel(r, Q, 0, dimod.BINARY)
    sampler = nl.sampler.SimulatedAnnealingSampler()

    # Generate num_samples amount of samples from simulated annealer
    if seed != None:
        start = timeit.default_timer()
        response = sampler.sample(bqm, num_reads = num_samples, seed = seed)
        sampler_time = timeit.default_timer() - start

    else:
        start = timeit.default_timer()
        response = sampler.sample(bqm, num_reads = num_samples)
        sampler_time = timeit.default_timer() - start

    return response, sampler_time


def read_err(fname, n_p):
    '''
    Grab 1 & 2-qubit errors from an IBM QC calibration csv file.

    fname:
        Path to csv file.

    n_c:
        Number of physical qubits.
    '''
    data = read_csv(fname, usecols = [5,6])
    its = [range(n_p)]*2
    err1 = {i : 1 for i in range(n_p)}
    err2 = {(i,j) : 1 for i,j in product(*its)}

    # Parse the column containing the two-qubit error-rates
    # and append to dict.
    for line in data['CNOT error rate']:
        interactions = line.split(',') # split up edges

        for string in interactions: # for each edge
            label, error = string.split(':') # split error and qubit labels

            nums = label.split('_')
            qbits = (int(sub('cx','',nums[0])), int(nums[1])) # grab qubit labels as pair of ints
            err2[qbits] = float(error)

    # Parse column containing single qubit error-rates and append to dict.
    for count, error in enumerate(data['Single-qubit U2 error rate']):
        err1[count] = error

    return err1, err2


def swap_err(i,j, graph, err2):
    ###
    ######### Do not use for qc's with directed hardware graphs, will not give you
    ######### the correct answer!
    ###
    '''
    Calculates the error of an interaction for two qubits given their
    architecture graph, taking into account swaps that would have to be done.
    Doesn't take into account the error you'd incur from swapping back, only the
    swaps you need to realize any given CNOT.

    We add error rates due to the nature of error calculation for qubits.

    eg, you want (1,4) error on melbourne, this requires 2 swaps (1,2), (2,3)
        so you get error(1,4) = err(1,2)*3 + err(2,3)*3 + err(3,4)

    i & j:
        Qubits labels (i,j) that you'd like the error of.

    graph:
        Architecture graph, assumed bidirected for now.

    err2:
        Errors from hardware graph for 2 qbit interactions.
    '''
    # I thought paths chose the shortest path that was numerically ordered but
    # it does not. Not sure if this effects any of the code.
    paths = nx.shortest_path(graph)
    swap_pair = makepair(paths[i][j])

    # If no swaps are necessary, just return the error of a CNOT between qubits
    if len(paths[i][j]) == 2: # no swaps:
        return err2[(i,j)]

    # If swaps need to be considered, update error for each swap, adding the
    # actual interaction of the original qubits at the end.

    else: # need swaps
        err = 0

        for pair in swap_pair[:-1]:
            err += err2[pair]*3

        err += err2[swap_pair[-1]]
        return err


def swap_prob(i,j, graph, err2):
    ###
    ######### Do not use for qc's with directed hardware graphs, will not give you
    ######### the correct answer!
    ###
    '''
    Calculates an approximation of the success probability of an interaction
    for two qubits given their architecture graph, taking into account swaps
    that would have to be done. Doesn't take into account the error you'd incur
    from swapping back.

    i & j:
        Qubits labels (i,j) that you'd like the error of.

    graph:
        Architecture graph, assumed bidirected for now.

    err2:
        Errors from hardware graph for 2 qbit interactions.
    '''
    # I thought paths chose the shortest path that was numerically ordered but
    # it does not. Not sure if this effects any of the code.
    paths = nx.shortest_path(graph)
    swap_pair = makepair(paths[i][j])

    if len(paths[i][j]) == 2: # no swaps:
        return 1-err2[(i,j)]

    else: # need swaps
        prob = 1
        # For each swap, update success prob. by multiplying through the
        # individual success probs
        for pair in swap_pair[:-1]*2:
            prob *= (1-err2[pair])**3

        prob *= 1-err2[swap_pair[-1]]
        return prob


def successprob(assign_dic, prob_m, one_gates_count, two_gates_count, hardware=False):
    '''
    Calculates the success probability of a given assignment from the gatecounts
    of the circuit.

    assign_dic:
        A dictionary where the keys are logical qubits and the values the
        hardware qubits the logical qubits have been assigned to.

    prob_m:
        The probability matrix returned by "probability_matrix" in
        matrix_funcs.py.

    one & two_gates_count:
        The one and two gate dictionaries that have been passed through the
        "gatecount" function.


    *Note : Make sure you don't give a symmetric two_gate dict as this will
            over-count the interactions and end up calculating the wrong
            success probability.
    '''
    # Initialize probability to update as we parse probability matrix
    prob = 1

    # Loop through gate counts
    for i,j in two_gates_count:

        if hardware == False:
            # Convert the circuit qubit call (i,j) to a physical qubit call (n,m)
            # as the prob. matrix is indexed by physical qubit labels
            n,m = assign_dic[i], assign_dic[j]

        else:
            # When using gate dictionaries from transpile_parse(), we don't need
            # to convert the calls as they are already referencing hardware qubits
            n,m = i,j

        if i == j: # self interaction
            prob *= prob_m[(n,m)] ** one_gates_count[i]

        else: # pair interaction
            prob *= prob_m[(n,m)] ** two_gates_count[(i,j)]

    return prob


def swapcount(allo_dic, paths, gatecount_matrix):
    '''
    Calculates the numbers of swaps for a given allocation, assuming a naive
    swap-back after each set of swaps is added.

    allo_dic:
        A dictionary where the keys are logical qubits and the values the
        hardware qubits the logical qubits have been assigned to.

    paths:
        An instance of "networkx.shortest_path" for the hardware graph being
        used.

    gatecount_matrix:
        An instance of "gatecount_matrix" from matrix_funcs.py.
    '''
    ind = np.triu_indices(gatecount_matrix.shape[0], k=1) # Grab upper triangular indices

    num_swaps = 0 # Initialize swap variable
    for i,j in zip(ind[0], ind[1]):
        n, m = allo_dic[i], allo_dic[j] # Map logical call to physical

        if len(paths[n][m]) <= 2: # No swaps
            continue
        elif len(paths[n][m]) > 2: # Need swaps
            # Size of path - 2 gives the number of swaps for this interaction
            # The gate metric counts all the times you would have to do this swap
            num_swaps += (len(paths[n][m]) - 2)*gatecount_matrix[i,j]

    # Factor of 2 assumes you swap back after each set of swaps.
    return num_swaps*2


def heatmap(n_p, data, graph, fname, err1, err2, slice_type=None):
    '''
    Generates heatmap of a given anneal's allocations.

    n_p:
        Number of hardware qubits.

    data:
        Should be log array (not compare array from qiskit comparisons), so that
        it contains the occurences data type.

    graph:
        networkx instance of the hardware graph.

    fname:
        Name for the heatmap figure.

    err1 & err2:
        Error dictionaries as returned by "read_err".

    *slice_type parameter is used to specificy what region of energies
    are the allocations being pulled from - useful for generating heatmaps in
    succession for various energy levels.
    '''
    # loop over data to grab allocations
    heatnum = np.zeros(n_p)
    total_occ = 0
    for datum in data:
        # Grab physical allocations
        allo_arr = datum['allocation']
        occur = datum['occurences'] # Account for duplicate allocations
        row,col = np.where(allo_arr == 1)
        heatnum[col] += occur

        total_occ += occur # count the total number of anneals processed (including dupes)
    print(heatnum, total_occ)
    # Don't  have a better way of doing this right now, manually setting the
    # node positions for melbourne graph
    pos = {0:(0,1), 1:(1,1), 2:(2,1), 3:(3,1), 4:(4,1), 5:(5,1), 6:(6,1),
          8:(6,0), 7:(7,0), 9:(5,0), 10:(4,0), 11:(3,0), 12:(2,0), 13:(1,0), 14:(0,0)}

    # Need to specify the edges otherwise any size you specify is allocated unordered
    edge_list = [(0,1), (1,2), (2,3), (3,4), (4,5), (5,6), (7,8), (8,9), (9,10), (10,11), (11, 12), (12,13), (13,14),
                (0,14), (1,13), (2,12), (3,11), (4,10), (5,9), (6,8)]

    # Same idea for nodes
    node_list = [_ for _ in range(15)]

    # Scale node/edge sizes proportional to error rate
    elist = [_ for _ in err1.values()]
    err1_arr = np.asarray(elist)
    node_sizes = (-np.log(err1_arr))**6 / 145

    err2list = [err2[edge] for edge in edge_list]
    err2_arr = np.asarray(err2list)
    widths = (-np.log(err2_arr))**5 / 80

    # Set range of values and create colourmap/norm object
    cm = plt.cm.Reds # Change colourmap here
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    edge_colors = ['k'] * len(edge_list)

    # Draw the graph
    nx.draw(graph, pos, node_color=(heatnum / total_occ), edge_color=edge_colors,
            node_size=node_sizes, width=widths, nodelist=node_list,
            edgelist=edge_list, with_labels=True, linewidths=1, cmap=cm,
            vmin=0, vmax=1)

    # Get colourbar on plot
    sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
    plt.colorbar(sm)
    ax = plt.gca() # to get the current axis
    ax.collections[0].set_edgecolor("#000000")

    # Set the generated plot title according to what energy region you're looking
    # at - ignore if you just want a generic allocation heatmap.
    if slice_type == 'HIGH':
        plt.title('High Energy Slice', fontsize=18)
    elif slice_type == 'MID':
        plt.title('Mid Energy Slice', fontsize=18)
    elif slice_type == 'LOW':
        plt.title('Low Energy Slice', fontsize=18)
    else:
        plt.title('Allocation Heatmap', fontsize=18)

    plt.savefig(fname, bbox_inches='tight')
    plt.clf()


from sklearn.neighbors import KernelDensity
from scipy.stats import entropy

def kldiv(compare_arr1, compare_arr2, log_darr1, log_darr2, field='q_swap', plot=False, flip=False):
    '''
    Given input comparison and log arrays for two q_swap datasets,
    calculates the KL-divergence of the two distributions. Has option of plotting
    the Kernel Density Estimates for the q_swap distributions and calculating the
    flipped KL-div.

    --Updated: Can flag the field input for what data you want to calculate the
               KL-divergence between - can still use 'q_swap', but using a log
               array for both compare_arr / log_darr inputs is fine and lets you
               calculate the difference between two freshly annealed distributions.

    compare_arr1 & compare_arr2:  **
        The respective compare arrays for the two distributions you'd like to
        look at.

    log_darr1 & log_Darr2:  **
        The respective log arrays for the two distributions you'd like to
        look at.

    **: Make sure to use the actual data_arrays for each run as input and not the
        main arrays that contain all the data for your runs.
    '''
    # Force arrays to be ordered the same to ensure that our index calls will
    # align to the things we want
    compare_arr1.sort(order=['energy'])
    compare_arr2.sort(order=['energy'])
    log_darr1.sort(order=['energy'])
    log_darr2.sort(order=['energy'])

    # Grab where the orginal arrays have a solution which occurs more than once
    # so that we can add it to the q_swap array
    occ1 = log_darr1['occurences']
    occ2 = log_darr2['occurences']
    row1, col1 = np.where(occ1 > 1)
    row2, col2 = np.where(occ2 > 1)
    occ_num1 = occ1[row1] - 1 # Minus by one since we already have one swap accounted for
    occ_num2 = occ2[row2] - 1

    # Which swaps occur more than once
    q_swap_add1 = compare_arr1[field][row1]
    q_swap_add2 = compare_arr2[field][row2]
    # List where each swap that occurs more than once is repeated the number of times
    # it needs to be added (easy to append to main array)
    q_append1 = [swap for count, swap in enumerate(q_swap_add1) for occ in range(occ_num1[count][0])]
    q_append2 = [swap for count, swap in enumerate(q_swap_add2) for occ in range(occ_num2[count][0])]

    #Final swap arrays
    q_swaparr1 = np.append(compare_arr1[field],q_append1)
    q_swaparr2 = np.append(compare_arr2[field],q_append2)

    # this should always equal the number of samples for the anneal being used
    #print(len(q_swaparr1))
    #print(len(q_swaparr2))

    # Kernel Density estimate
    x_dom1 = np.linspace(np.min(q_swaparr1), np.max(q_swaparr1), len(q_swaparr1))[:, np.newaxis]
    x_dom2 = np.linspace(np.min(q_swaparr2), np.max(q_swaparr2), len(q_swaparr2))[:, np.newaxis]
    kde1 = KernelDensity(kernel='gaussian', bandwidth=4).fit(q_swaparr1[:, np.newaxis])
    kde2 = KernelDensity(kernel='gaussian', bandwidth=4).fit(q_swaparr2[:, np.newaxis])
    prob_dens1 = np.exp(kde1.score_samples(x_dom1))
    prob_dens2 = np.exp(kde2.score_samples(x_dom2))

    # Plot if the kernel density estimates if flapped by user
    if plot != False:
        label1 = input('Describe array1 for plot: ')
        label2 = input('Describe array2 for plot')
        plt.plot(x_dom1[:, 0], prob_dens1, color='r', lw=2,
                 linestyle='-', label=label1)
        plt.plot(x_dom2[:, 0], prob_dens2, color='b', lw=2,
                 linestyle='-', label=label2)

        plt.legend(loc='upper right')
        plt.title('Kernel Density Comparison', fontsize=14)
        plt.xlabel('Qiskit Swap Count', fontsize=14)
        plt.ylabel('Estimated Probability Density', fontsize=14)

        plt.savefig('Density_plot_'+ label1 + '_' + label2 + '.png', dpi=300, bbox_inches='tight')

    # Calculate the KL-divergences, with keyword to decide whether or not to
    # flip the distributions (since KL-div isn't symmetric)
    if flip == False:
        kldiv_kernel = entropy(pk=prob_dens1, qk=prob_dens2)
        kldiv_data = entropy(pk=q_swaparr1, qk=q_swaparr2)
    else:
        kldiv_kernel = entropy(pk=prob_dens2, qk=prob_dens1)
        kldiv_data = entropy(pk=q_swaparr2, qk=q_swaparr1)
    # Print out out the KL-divergences
    print('The KL-divergence using kernel estimate: ', kldiv_kernel)
    print('The KL-divergence using the data itself: ', kldiv_data)

###### Functions not currently in use:

# This function just doesn't work for some reason, but works in notebook, maybe
# some weird data rounding issue?

# def topsoln(data_arr, num=5, fields=['energy', 'occurences', 'swaps','success_prob']):
#     '''
#     Given an array of the type returned by datalog.data_array, will sort the
#     anneal results by swap count and print the first 'num' solutions.
#     '''
#     data_arr.sort(order=['swaps', 'energy'])
#     print('Showing the %d "best" solutions: '%(num), '\n', data_arr[fields][:num])


  ## Functions that have to do with grabbing success probability of qiskit
  ## transpiled circit
  # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼


  # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

# from qiskit.transpiler import PassManager, PassManagerConfig
# from qiskit.transpiler.passes import BasicSwap
# from qiskit.transpiler.preset_passmanagers import (level_0_pass_manager,
#                                                    level_1_pass_manager,
#                                                    level_2_pass_manager,
#                                                    level_3_pass_manager)
#
#
# def qiskit_preset_pm(initial_layout, coupling_map, routing_method, backend_properties, circuit, arr=None):
#     '''
#     Call the function with the properties you want to test on qiskits preset
#     passmanager levels and it will print out what each of them returns.
#
#     initial_layout:
#         A Layout() object from qiskit.transpiler.passes.
#
#     coupling_map:
#         A coupling_map object from qiskit.transpiler.
#
#     routing_method:
#         One of three strings --> 'basic' or 'lookahead' or 'stochastic'.
#
#     backend_properties:
#         A backend_properties object for one of IBM's backends.
#         Use provider.backends.[IBM_MACHINE_NAME].properties() to get the right
#         format for the backend_properties object.
#
#     circuit:
#         A QuantumCircuit object of the desired circuit you'd like to look at.
#
#     arr:
#         If given, the function will return the concatenated array of arr and the
#         plot_arr generated by running the func.
#     '''
#     #Generatee preset config and make list of passmanagers
#     config = PassManagerConfig(initial_layout=initial_layout, coupling_map=coupling_map, routing_method=routing_method, backend_properties=backend_properties)
#     pm_list = [level_0_pass_manager(config), level_1_pass_manager(config), level_2_pass_manager(config), level_3_pass_manager(config)]
#
#     # Initialize plot array
#     fields = [('lvl0', 'i8'), ('lvl1', 'i8'), ('lvl2', 'i8'), ('lvl3', 'i8')]
#     plot_arr = np.zeros(1, dtype=fields)
#     plot_arr.resize(1,1)
#
#     # Run through the preset passmanagers according to input config
#     for count, pm in enumerate(pm_list):
#         transpile_circ = pm.run(circuit)
#         plot_arr[0][0][count] = transpile_circ.count_ops()['swap']
#         #print('Swap # with lvl%d:'%(count), transpile_circ.count_ops()['swap'])
#
#     # Returns the plot array, but checks if you have a previous plot array you'd
#     # like to append to.
#     if arr is None:
#         return plot_arr
#     else:
#         return np.concatenate((arr,plot_arr), axis=0)
#
#     # # Control basic swap passmanager run
#     # bs = BasicSwap(coupling_map=coupling_map)
#     # pm_bs = PassManager(bs)
#     # basic_circ = pm_bs.run(circuit)
#     # print('Swap # with BasicSwap:', basic_circ.count_ops()['swap'])


# def swapper(edges, assign_dic, one_gates, two_gates, bidirected = False):
#     '''
#     Updates the original one and two gate dicts. with the gates that would need
#     to be added to actually execute the quantum circuit, given the qc's
#     architecture graph as well as the assignment of circuit to physical qubits.
#     Note that we're assuming n_c = n_p and naive swapping back of the circuit
#     after any swap operations.
#
#     di_edge:
#         Should be the edges from the directed architecture graph.
#
#     assign_dic:
#         A dictionary of the form returned by the assignment() function.
#
#     one_gates & two_gates:
#         Gate dictionaries returned by qasm_parse() function.
#     '''
#     # Need copies since we need the original dics for each assignment
#     one_gates_swaps = deepcopy(one_gates)
#     two_gates_swaps = deepcopy(two_gates)
#
#     # make bi-directional graph where labels for vertices are the assignments of
#     # logical qubits to their physical qubit, only works for correct assignments
#
#     allo_graph = nx.Graph()
#     for edge in edges:
#         allo_graph.add_edge(assign_dic[edge[0]], assign_dic[edge[1]])
#
#     graph_edges = [(assign_dic[edge[0]], assign_dic[edge[1]]) for edge in edges]
#     paths = nx.shortest_path(allo_graph)
#     swap_count = 0
#
#     for l_qbits in two_gates.keys():
#
#         for gate, count in zip(two_gates[l_qbits].keys(), two_gates[l_qbits].values()):
#             swap_pair = makepair(paths[l_qbits[0]][l_qbits[1]])
#             flip = (l_qbits[1], l_qbits[0])
#             ##############################################################################################
#             if count > 0:
#
#                 if len(paths[l_qbits[0]][l_qbits[1]]) == 1: # not valid, skip
#                     continue
#
#                 elif len(paths[l_qbits[0]][l_qbits[1]]) == 2: # no swaps necessary
#                     if bidirected == True: # if bidirected doesnt need any changes if nodes next to each other
#                         continue
#                     elif flip in graph_edges: # if can be executed w/ reverse direction (graph directed)
#                         one_gates_swaps[l_qbits[0]]['h'] += 2*count
#                         one_gates_swaps[l_qbits[1]]['h'] += 2*count
#                         two_gates_swaps[flip][gate] += count # add 2-gate operation to flip
#                         two_gates_swaps[l_qbits][gate] -= count # remove 2-gate operation from orginal pair
#                     else: # doesn't need swap or flip, leave alone
#                         continue
#
#                 else: # need swaps to execute gate
#                     for pair in swap_pair[:-1]: # skip last pair, each update term has *2 since we assume you swap back
#
#                         if bidirected == True: # don't need any hadimards for bidirected graph
#                             two_gates_swaps[pair]['cx'] += 2*3*count # wouldnt be correct if the fid of the flip of an edge wasnt the same
#                             continue                                 # but this is the case for ibm qc's so correct for now
#
#                         one_gates_swaps[pair[0]]['h'] += 2*2*count
#                         one_gates_swaps[pair[1]]['h'] += 2*2*count
#
#                         if pair in graph_edges: # checking if direction is right
#                             two_gates_swaps[pair]['cx'] += 2*3*count # decompose swap as 3 CNOTS
#
#                         else: # if direction was wrong, use flip
#                             two_gates_swaps[(pair[1],pair[0])]['cx'] += 2*3*count # decompose swap as 3 CNOTS
#
#                     original_qbits = swap_pair[-1] # the assigned qbits that actually interact for original gate op.
#
#                     if bidirected == True: #just to make sure bidirected updates correctly
#                         two_gates_swaps[original_qbits][gate] += count
#
#                     elif original_qbits in graph_edges: # right direction
#                         two_gates_swaps[original_qbits][gate] += count # add where original gate actually occured
#
#                     else: # wrong direction, use flip
#                         two_gates_swaps[ (original_qbits[1], original_qbits[0]) ][gate] += count
#
#                     two_gates_swaps[l_qbits][gate] -= count # remove original gate(s) that didn't occur
#
#                     swap_count += len(swap_pair[:-1])*count # don't assume swap back for swap_count
#             ##############################################################################################
#             else: # no gate operations between these qubits
#                 continue
#
#     return one_gates_swaps, two_gates_swaps, swap_count


# def findsoln(data): #Tested
#                     ### need to update description
#     '''
#     Given the sampling from the simulated annealer, spits out qubit allocations
#     that actually satisfied the constraint terms. Doesn't work when the cost
#     function is non-negative (i.e. when penalties are all 0), since no allocations
#     will be seen as satisfied.
#
#     data:
#         Output from data_array.
#
#     '''
#     valid = len(data['success_prob'][data['success_prob'] != 0]) / len(data)
#     print("Valid Solution %", valid*100, "\n")
#     for metric in data:
#         if metric['success_prob'][0] == 0: # invalid solution
#             print(colored(metric[0], 'red'))
#         else: # valid solution
#             print(colored(metric[0], 'green'))
