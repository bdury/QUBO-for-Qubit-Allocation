'''
Functions to calculate the various matrices we would like to use as different
qreg [16];

'''
import numpy as np
import networkx as nx
from QUBOfuncs import *


def distance_matrix(graph):
    '''
    Calculates the distance matrix for the architecture graph. Currently we
    assume the graph is bidirected.

    graph:
        A networkx graph of the quantum architecture graph.
    '''
    paths = nx.shortest_path(graph)
    n_p = graph.number_of_nodes()
    distancematrix = np.zeros(n_p*n_p) # initialize empty distance matrix

    its = [range(n_p)]*2

    # Iterate over all node combinations and use function to calculate the
    # distance between each node in the graph
    for count, i in enumerate(product(*its)):
        distancematrix[count] = distance(paths, i[0], i[1])

    distancematrix.resize(n_p,n_p)

    return distancematrix


def error_matrix(graph, err1, err2):
    '''
    Creates the error matrix, where each element (i,j) represents the
    error qubits i & j would have for an interaction, taking into account
    self-interactions (diagonal) and if swaps would have to be inserted. Does
    not calculate the incurred error from swapping back, just the error from the
    swaps you'd have to implement.

    graph:
        Architecture graph, assumed bidirected for now.

    err1 & err2:
        One and two qubit errors from the hardware graph.
    '''
    # Initialize matrix and resize
    size = len(graph)
    err_m = np.zeros(size**2)
    err_m.resize(size,size)

    its = [range(size)]*2

    # Loop over all node interactions and calculate error of interaction
    for i,j in product(*its):
        if i == j: # self-interaction
            err_m[i,j] = err1[i]
        else: # pair interaction
            err_m[i,j] = swap_err(i, j, graph, err2)

    return err_m


def probability_matrix(graph, err1, err2):
    '''
    Creates the probability matrix, where each element (i,j) represents the
    success probability qubits i & j would have for an interaction, taking into
    account self-interactions (diagonal) and if swaps would have to be inserted.

    graph:
        Architecture graph, assumed bidirected for now.

    err1 & err2:
        One and two qubit errors from the hardware graph.
    '''
    # Initialize matrix and resize
    size = len(graph)
    prob_m = np.zeros(size**2)
    prob_m.resize(size,size)

    its = [range(size)]*2

    for i,j in product(*its):
        if i == j: # self-interaction
            prob_m[i,j] = 1-err1[i]
        else: # pair interaction
            prob_m[i,j] = swap_prob(i, j, graph, err2)

    return prob_m


def gatecount_matrix(n_c, one_gates, two_gates):
    '''
    Creates a matrix where each element corresponds to the number of gates
    that a given qubit/pair of qubits saw. For now, assuming that it is
    symmetric (true for bidirected architecture graphs).
    '''
    # Initialize empty array
    gate_m = np.zeros(n_c*n_c)
    gate_m.resize(n_c,n_c)
    # Count up the gate operations for each dictionary
    one_gates_c = gatecount(one_gates)
    two_gates_c = makesymmetric(gatecount(two_gates))

    for q1, q2 in two_gates:
        if q1 == q2: # linear terms (diag.)
            gate_m[q1,q2] = one_gates_c[q1]

        else: # off-diagonal (quad.)
            gate_m[q1,q2] = two_gates_c[(q1,q2)]

    return gate_m

from networkx.algorithms.centrality import betweenness_centrality

def connectivity_matrix(graph):
    '''

    graph:
        Architecture graph, assumed bidirected for now.
    '''
    # Initialize matrix and resize
    size = len(graph)
    con_m = np.zeros(size**2)
    con_m.resize(size,size)

    # Generate betweeness centrality for graph
    betweenness = central = betweenness_centrality(graph, k=size)

    its = [range(size)]*2

    for i,j in product(*its):
        if i == j: # self-interaction
            con_m[i,j] = 1
        else: # pair interaction
            con_m[i,j] = (1-betweenness[i]) * (1-betweenness[j])

    return con_m


def gatescale_matrix(n_c, one_gates, two_gates, scale = 10):
    '''
    Matrix where the elements are the ratios of the gate counts w.r.t. the
    maximum count, scaled like the gain function (used in physics education
    research community):

        gain = (<post> - <pre>) / (1 - <pre>)

    where post and pre refer to the scores of students in a class before
    and after taking said class. The denominator's motivation is to put less
    weight on the students scores before they had taken the class. Motivation
    for using a scaled gate number is than you won't have differences in the
    behaviour of the QUBO model for large circuits, since if you use gate number
    and don't normalize it then large circuits have much larger coefficients.

    n_c:
        Number of circuit qubits.

    one & two_gates:
        Dictionaries returned by "qasm_parse" in QUBOfuncs.py.

    scale:
        The coefficient out front for the re-scaled gate counts - 10 seems to be
        fairly reasonable in this context.

    NOTE: Using log(gatecount) matrix seems to just a better implementation of
    this idea, but keeping this func. around for now.
    '''
    gate_m = gatecount_matrix(n_c, one_gates, two_gates)
    gatemax = np.max(gate_m)
    gatescale_m = scale*(gate_m / gatemax) / (1 - .99*(gate_m / gatemax))

    return gatescale_m
