"""
Contains the coefficient matrix class definition. Motivation for object oriented implementation is so that the heuristic used
to calculate the coefficient matrix can be easily changed by changing the initialization.

It is expected that if you want to change the number of heuristics being used, you will edit this function
to modify them.
"""
import numpy as np
from QUBOfuncs import *
from itertools import product


class Coefficient_Matrix():
    def __init__(self, n_c, n_p, heur_1, heur_2, heur_3):
        # Assign heuristic matrices
        self.arr1 = heur_1
        self.arr2 = heur_2
        self.arr3 = heur_3

        # Initialize useful variables
        self.n_p = n_p
        self.n_c = n_c

        alg = n_p * n_c
        self.alg = alg

        # Initialize empty coef. matrix w/ correct size
        matrix = np.zeros(alg**2)
        matrix.resize(alg,alg)
        self.matrix = matrix

        # this loop over the entries in the iterable output from product acts like
        # the cartesian product, giving the same result as 4 nested for loops
        # pairlist is then all possible pairs of pairs of values:
        #   ( ( [0,n_c],[0,n_c] ) , ( [0,n_p],[0,n_p] ) )
        pairlist = []
        its = [range(n_c), range(n_p)]*2

        for i,j,k,l in product(*its):
            pairlist.append([(i,j), (k,l)])

        self.pairlist = pairlist


    def calc_matrix(self, penalty = {'phi':0,'theta':0}):
        '''
        Calculates the coefficient matrix in place, according to the penalty
        coefficient values specified.

        penalty:
            Should match the format of the default penalty dictionary given,
            values for the penalty coefficients should always be >= 0.
        '''
        # Define penalty coefficients locally
        self.penalty = penalty
        phi = self.penalty['phi']
        theta = self.penalty['theta']

        # Coefficient matrix calculation. We do this by enumerating over all
        # combinations of allocations, encoded in pairlist.

        coef_matrix = self.matrix.ravel() # flattened view of the matrix

        # The exact combination of the arrays will change as we try testing
        # different things
        for count, pair in enumerate(self.pairlist):
            if pair[0] == pair[1]: # linear terms
                coef_matrix[count] = self.arr1[pair[0][0],pair[0][0]] * ( self.arr2[pair[0][1],pair[0][1]] \
                                     * self.arr3[pair[0][1],pair[0][1]] ) - phi - theta

            elif pair[0][0] == pair[1][0]: # penalize allocating within a row
                coef_matrix[count] = theta

            elif pair[0][1] == pair[1][1]: # penalize allocating within a column
                coef_matrix[count] = phi

            else: # All remaining entries (quadratic terms)
                coef_matrix[count] = self.arr1[pair[0][0],pair[1][0]] * ( self.arr2[pair[0][1],pair[1][1]] \
                                     * self.arr3[pair[0][1],pair[1][1]] )


    def bqm_dicts(self):
        '''
        Method that returns two dictionaries of the form:

            {variable_label : QUBO coefficient}

        for the linear and quadratic terms of the model. Make sure you've run
        the 'calc_matrix' method first if trying to anneal.
        '''
        # Linear terms are on the diagonal
        diag = np.diagonal(self.matrix)

        # assign coefficients to proper qubit label
        r = {count : diag_vals for count, diag_vals in enumerate(diag)}

        Q = {}
        for i in self.pairlist:
            if i[0] != i[1]: # everything but the linear terms
                new_ind1 = mapind(*i[0], self.n_p)
                new_ind2 = mapind(*i[1], self.n_p)
                Q[(new_ind1, new_ind2)] = self.matrix[new_ind1, new_ind2]
            # ignore linear cases
            else:
                continue

        return r, Q
