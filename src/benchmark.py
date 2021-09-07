from QUBOfuncs import *
from coef_matrix import *
from datalog import *
from matrix_funcs import *

import matplotlib.pyplot as plt
import networkx as nx

from tqdm import tqdm
from os import listdir
from os import mkdir
from os.path import join
from astropy.visualization import hist
from pathlib import Path

def calc_properties(path):
    '''
    Given a path to where circuit qasm files are located, uses "circ_properties"
    to grab properties of the circuits and load them into a record array which
    is saved in the benchmark/data/property array directory.
    '''
    # Location to save array to
    saveloc = Path().resolve().parent / 'benchmarks' / 'data' / 'property array'

    # Grab circuit names
    circ_name = listdir(path)

    # Define record array fields
    fields = [('circuit', 'U40'),
              ('logical_qubit', 'i8'),
              ('g1', 'i8'),
              ('g2', 'i8'),
              ('depth', 'i8')
             ]

    # Grab circuit properties and save to array
    for count, fname in tqdm(enumerate(circ_name)):
        # Grabs circuit properties
        qubit, g1, g2, depth = circ_properties( join(path, fname) )

        # If first run, initialize main array
        if count == 0:
            property_arr = np.array([(fname, qubit, g1, g2, depth)], dtype = fields)
        # After first run append to main array
        else:
            temp_arr = np.array([(fname, qubit, g1, g2, depth)], dtype = fields)
            property_arr = np.concatenate((property_arr, temp_arr), axis=0)

    # Save array to disk
    log_dump(property_arr,
             saveloc,
             'circ_prop'
            )


def benchmark(circuit_list, loc, n_p, graph, err1, err2, num_samples = 1000):
    '''
    Given a set of circuits, will create and save a log file containing the
    data for all the anneal runs of all the specified circuits.

    circuit_list:
        A list of the names of the circuits you would like to benchmark.

    loc:
        The location of the qasm files for the circuits you specified in
        'circuit_list'.

    n_p:
        The number of nodes in the hardware graph.

    graph:
        The networkx.Graph() object representing the hardware graph.

    err_graph:
        The location of the csv file containing the IBM calibration data.

    num_samples:
        How many times the annealer runs for each circuit.
        (eg. number of solutions)
    '''
    # Place to save data
    saveloc = Path().resolve().parent / 'benchmarks' / 'data' / 'benchmark array'

    # If something only needs to be calculated once do it before loop

    # Grab hardware error rates
    #err1, err2 = read_err(err_graph, n_p)

    # Define QUBO coefficients (these don't depend on circuit properties so we
    # keep outside the loop to save calculation time, the other coefficient does
    # so we keep it in the loop)
    heur2 = -np.log( probability_matrix(graph, err1, err2) )
    #heur2.fill(1)
    heur3 = distance_matrix(graph)**3

    for count, filename in tqdm(enumerate(circuit_list)):

        # Grab circuit proberties and write to dictionary
        n_c, g1, g2, depth = circ_properties( join(loc,filename) )
        one_gates, two_gates = qasm_parse( join(loc,filename), n_c)

        # Define the QUBO coefficients you want to use (put circuit dependent
        # heuristics here)
        heur1 = gatecount_matrix(n_c, one_gates, two_gates)

        # Initialize coefficient matrix object
        coef = Coefficient_Matrix(n_c, n_p, heur1, heur2, heur3)

        # Calculate the 'default' matrix (no penalty coefficients set)
        # so we can use it to set our penalty coefficients
        coef.calc_matrix()

        # If a circuit fails to run, re-anneal with larger penalties until it works
        loops = 1
        while True:
            # Set penalty coefficients
            penalty = {'phi' : np.max(coef.matrix)*loops,
                       'theta' : np.max(coef.matrix)*loops,
                      }

            # Calculate coefficient matrix with penalties set and turn into form the
            # simulated annealer expects
            coef.calc_matrix(penalty)
            r, Q = coef.bqm_dicts()

            # Get QUBO allocations using simulated annealer
            response, anneal_time = anneal(r, Q, num_samples)

            # Calculate metrics for the anneal and save to data_array
            data = data_array(n_c, n_p, response, graph, one_gates, two_gates, err_graph)

            # Check if any anneals failed
            if np.any(data['success_prob'] == 0) == True and np.any(data['swaps'] == 0) == True:
                loops += 1

            else:
                if loops != 1:
                    print("{} failed {} times before succeeding.".format(filename, loops-1))
                break

            if loops == 6: # prevent infinite loop if doesn't succeed within 5 tries
                print("{} couldn't reach 100% and failed in 5 tries.".format(filename))
                break
        # Need to initialize main data array if this is the first run, otherwise
        # create a temp_log and append to main one
        if count == 0:
            log = log_array(penalty,
                            filename,
                            data,
                            time = anneal_time,
                            benchmark = True)
        else:
            log_temp = log_array(penalty,
                            filename,
                            data,
                            time = anneal_time,
                            benchmark = True)
            log = np.concatenate((log, log_temp), axis = 0)

    # Save the log array to the benchmark folder
    log_dump(log, saveloc, 'benchmark-results')


def QUEKO_benchmark(circuit_list, loc, err_graph=None, num_samples = 1000):
    '''
    Given a set of circuits, will create and save a log file containing the
    data for all the anneal runs of all the specified circuits, assuming the
    circuits are from the QUEKO benchmarks.

    circuit_list:
        A list of the names of the circuits you would like to benchmark.

    loc:
        The location of the qasm files for the circuits you specified in
        'circuit_list'.

    num_samples:
        How many times the annealer runs for each circuit.
        (eg. number of solutions)
        Set to 100 for QUEKO since the hardware graphs are quite large.
    '''
    # Place to save data
    saveloc = Path().resolve().parent / 'benchmarks' / 'data' / 'benchmark array'

    # Loop over specified benchmark circuits
    for count, filename in tqdm(enumerate(circuit_list)):
        if '16QBT' in filename:
            edges = hardware_edges('16QBT')
        elif '20QBT' in filename:
            edges = hardware_edges('20QBT')
        elif '53QBT' in filename:
            edges = hardware_edges('53QBT')
        elif '54QBT' in filename:
            edges = hardware_edges('54QBT')
        else:
            print("Couldn't specify the hardware graph - please stop the benchmark")

        # Define hardware graph
        graph = nx.Graph()
        graph.add_edges_from(edges)

        # Get the number of physical qubits from graph
        n_p = len(graph)

        # Grab circuit proberties and write to dictionary
        n_c, g1, g2 = circ_properties( join(loc,filename) )
        one_gates, two_gates = qasm_parse(loc+filename, n_c)

        # Define the QUBO coefficients you want to use (put circuit dependent
        # heuristics here)
        heur1 = gatecount_matrix(n_c, one_gates, two_gates)
        heur3 = distance_matrix(graph)**3

        # We can't use our error-rate heuristic for the QUEKO benchmarks since we
        # lack that data for the specific hardware graphs used so we effectively
        # remove it by making it all 1
        heur2 = heur3.copy()
        heur2.fill(1)

        # Initialize coefficient matrix object
        coef = Coefficient_Matrix(n_c, n_p, heur1, heur2, heur3)

        # Calculate the 'default' matrix (no penalty coefficients set)
        # so we can use it to set our penalty coefficients
        coef.calc_matrix()

        # If a circuit fails to run, re-anneal with larger penalties until it works
        loops = 1
        while True:
            # Set penalty coefficients
            penalty = { 'phi' : np.max(coef.matrix)*loops,
                        'theta' : np.max(coef.matrix)*loops,
                      }

            # Calculate coefficient matrix with penalties set and turn into form the
            # simulated annealer expects
            coef.calc_matrix(penalty)
            r, Q = coef.bqm_dicts()

            # Get QUBO allocations using simulated annealer
            response, anneal_time = anneal(r, Q, num_samples)

            # Calculate metrics for the anneal and save to data_array
            data = data_array(n_c, n_p, response, graph, one_gates, two_gates, err_graph, QUEKO=True)

            # Check if any anneals failed
            if np.any(data['success_prob'] == 0) == True and np.any(data['swaps'] == 0) == True:
                loops += 1

            else:
                if loops != 1:
                    print("{} failed {} times before succeeding.".format(filename, loops-1))
                break

            if loops == 6: # prevent infinite loop if doesn't succeed within 5 tries
                print("{} couldn't reach 100% and failed in 5 tries.".format(filename))
                break
        # Need to initialize main data array if this is the first run, otherwise
        # create a temp_log and append to main one
        if count == 0:
            log = log_array(penalty,
                            filename,
                            data,
                            time = anneal_time,
                            benchmark = True)
        else:
            log_temp = log_array(penalty,
                            filename,
                            data,
                            time = anneal_time,
                            benchmark = True)
            log = np.concatenate((log, log_temp), axis = 0)

    # Save the log array to the benchmark folder
    log_dump(log, saveloc, 'QUEKO-benchmark-results')


def plot_times(bm_arr, prop_arr, save=False):
    '''
    Plots a graph of the anneal times vs gate number for benchmark run.

    bm_arr:
        Loaded array of the benchmark results.
    prop_arr:
        Loaded array of the properties of the benchmark circuits.
    '''
    # Grab anneal times & circuit names
    times = bm_arr['time']
    cnames = bm_arr['circuit']

    # Order both arrays just to make sure we match up
    bm_arr.sort(order=['circuit'])
    prop_arr.sort(order=['circuit'])


    # Grab gatenumber using cnames as filter
    ind = []
    for count, name in enumerate(prop_arr['circuit']):
        if name in cnames:
            ind.append(count)

    # Grab filtered gatenumbers and add them to get total gates per circuit
    g1 = prop_arr['g1'][ind]
    g2 = prop_arr['g2'][ind]
    g_tot = g1 + g2

    # # Make anneal-time plot
    plt.figure(1)
    plt.plot(g_tot, times, '.b')
    plt.title('Anneal time vs. Gate Number', fontsize=18)
    plt.ylabel('Anneal time  [s]', fontsize=14)
    plt.xlabel('Gate count', fontsize=14)
    plt.grid()

    plt.figure(2)
    plt.plot(prop_arr['logical_qubit'][ind], times, '.b')
    plt.title('Anneal time vs. Qubit Number', fontsize=18)
    plt.ylabel('Anneal time  [s]', fontsize=14)
    plt.xlabel('Qubit #', fontsize=14)
    plt.grid()

    if save != False:
        saveloc = Path().resolve().parent / 'benchmarks' / 'data'
        plt.savefig( join( str(saveloc), 'anneal_times.pdf') )


def plot_results(bm_arr, directory): ## need relative paths added
    '''
    Produces the set of histograms from the anneal distribution of a set of
    solutions for all the circuits the specified benchmark array.
    '''
    # Define directory to save plots
    path = Path().resolve().parent / 'benchmarks' / 'data' / 'plots'
    path = str(path) # to use with os.join

    # Make new folder to house all of this benchmark's data
    mkdir( join(str(path), directory) )

    for count, data in tqdm(enumerate(bm_arr['data_array'])):

        # Circuit name
        cname = bm_arr['circuit'][count]

        # Grab metrics
        swap = data['swaps']
        energy = data['energy']
        sp = data['success_prob']
        occ = data['occurences']

        # Make new folder for each circuit's anneal plots
        mkdir( join(path , directory , cname) )

        # Use for plot filenames
        str_penalty = ''
        for val in bm_arr['penalty'][count].values():
            str_penalty += str( round(val,2) ) + '_'

        # plot names
        saveloc = join(path, directory, cname)
        fname_swap = join(saveloc , str_penalty , cname , '_SWAP_HIST.pdf')
        fname_energy = join(saveloc , str_penalty , cname , '_ENERGY_HIST.pdf')

        # Loop to create a list of all swaps (we make the data array using response.aggregate
        # so if there are solutions that occur multiple times, we lose some swaps to plot in the
        # histograms)
        sw = []
        en = []
        for s, e, num in zip(swap, energy, occ):
            sw.extend([s]*num)
            en.extend([e]*num)

        # Plot and save the results

        # Sometimes the auto-bin method fails, so we catch it with an exception
        # and just make the bins a constant
        with np.errstate(divide='raise', invalid='raise'):
            try:
                plt.figure(1)
                hist(sw, bins='scott', rwidth=.85, facecolor='g', histtype="bar", alpha=.65)
                plt.title('Anneal Swap Histogram', fontsize=16)
                plt.grid(alpha=.6)
                plt.ylabel('Occurences', fontsize=14)
                plt.xlabel('Swap count', fontsize=14)
                plt.savefig(fname_swap)
                plt.clf()

                plt.figure(2)
                hist(en, bins='scott', rwidth=.85, facecolor='g', histtype="bar", alpha=.65)
                plt.title('Anneal Energy Histogram', fontsize=16)
                plt.grid(alpha=.6)
                plt.ylabel('Occurences', fontsize=14)
                plt.xlabel('Energy', fontsize=14)
                plt.savefig(fname_energy)
                plt.clf()
            except:
                plt.figure(1)
                hist(sw, bins=15, rwidth=.85, facecolor='g', histtype="bar", alpha=.65)
                plt.title('Anneal Swap Histogram', fontsize=16)
                plt.grid(alpha=.6)
                plt.ylabel('Occurences', fontsize=14)
                plt.xlabel('Swap count', fontsize=14)
                plt.savefig(fname_swap)
                plt.clf()

                plt.figure(2)
                hist(en, bins=15, rwidth=.85, facecolor='g', histtype="bar", alpha=.65)
                plt.title('Anneal Energy Histogram', fontsize=16)
                plt.grid(alpha=.6)
                plt.ylabel('Occurences', fontsize=14)
                plt.xlabel('Energy', fontsize=14)
                plt.savefig(fname_energy)
                plt.clf()

        # plot the metric comparisons and save results
        metric_plot_bm(swap, energy, sp, join(saveloc , str_penalty , cname) )


def metric_plot_bm(swap, energy, sp, saveloc):
    '''
    Plots the metric correlation plots assuming you're calling the function
    from the main benchmark plotter instead of with the bare data array.
    '''

    # Define filenames
    fig1name = join(saveloc , 'SWAP_VS_ENERGY.pdf')
    fig2name = join(saveloc , 'SP_VS_ENERGY.pdf')
    fig3name = join(saveloc , 'SP_VS_SWAP.pdf')

    # Plotting routines
    plt.figure(1)
    plt.plot(energy, swap, '.b', alpha=.35)
    plt.grid()
    plt.title('Swaps vs. Energy')
    plt.xlabel('Energy', fontsize=14)
    plt.ylabel('Swaps', fontsize=14)
    plt.savefig(fig1name)
    plt.clf()

    # For large circuits success probability will approach 0,
    # if that is the case, just don't plot the stuff
    with np.errstate(divide='raise'):
        try:
            log_sp = -np.log(sp)

            plt.figure(2)
            plt.plot(energy, log_sp, '.b', alpha=.35)
            plt.grid()
            plt.title('-ln(SP) vs. Energy')
            plt.xlabel('Energy', fontsize=14)
            plt.ylabel('-ln(SP)', fontsize=14)
            plt.savefig(fig2name)
            plt.clf()

            plt.figure(3)
            plt.plot(swap, log_sp, '.b', alpha=.35)
            plt.grid()
            plt.title('-ln(SP) vs. Swaps')
            plt.xlabel('Swaps', fontsize=14)
            plt.ylabel('-ln(SP)', fontsize=14)
            plt.savefig(fig3name)
            plt.clf()

        except:
            pass


def metric_correlation_plot(data_arr, circ_n, penalty, dir):
    '''
    Given a data_array type object, will check the correlation between the
    energy, swaps and success probability (3 total).
    '''

    # Grab data from array
    swap = data_arr['swaps']
    energy = data_arr['energy']
    sp = data_arr['success_prob']

    # For naming files
    str_penalty = ''
    for val in penalty.values():
        str_penalty += str( round(val,2) ) + '_'

    # Where to save files
    loc = Path().resolve().parent / 'benchmarks' / 'data' / 'plots' / dir
    fig1name = join( str(loc) , str_penalty , circ_n , 'SWAP_VS_ENERGY.png')
    fig2name = join( str(loc) , str_penalty , circ_n , 'SP_VS_ENERGY.png')
    fig3name = join( str(loc) , str_penalty , circ_n , 'SP_VS_SWAP.png')

    # Plotting routines
    plt.figure(1)
    plt.plot(energy, swap, '.b')
    plt.grid()
    plt.title('Swaps vs. Energy')
    plt.xlabel('Energy', fontsize=14)
    plt.ylabel('Swaps', fontsize=14)
    plt.savefig(fig1name, dpi=300)
    plt.clf()

    plt.figure(2)
    plt.plot(energy, -np.log(sp), '.b')
    plt.grid()
    plt.title('-ln(SP) vs. Energy')
    plt.xlabel('Energy', fontsize=14)
    plt.ylabel('-ln(SP)', fontsize=14)
    plt.savefig(fig2name, dpi=300)
    plt.clf()

    plt.figure(3)
    plt.plot(swap, -np.log(sp), '.b')
    plt.grid()
    plt.title('-ln(SP) vs. Swaps')
    plt.xlabel('Swaps', fontsize=14)
    plt.ylabel('-ln(SP)', fontsize=14)
    plt.savefig(fig3name, dpi=300)
    plt.clf()


def coef_judge(prop_arr, arr1, arr2, frac, flip=False, label1='arr1', label2='arr2'):
    '''
    Given two benchmark data arrays, will plot the %-difference of the mean of
    the top 'frac' number of solutions. Used to judge how well a given QUBO
    coefficient performs.

    prop_arr:
        Property array from a prop_arr file.

    arr1 & arr2:
        Data arrays of a benchmark run (eg. arr1=bench1['data_array'])

    frac:
        The fraction of top performing solutions to include in the average
        calculation.
    '''
    # Location to save data to
    saveloc = Path().resolve().parent / 'benchmarks' / 'data' / 'plots' / 'QUBO Coef. Comparisons'
    saveloc = str(saveloc) # to use with os.join

    # Set the fields we are going to use for plotting
    fields = ['energy', 'swaps', 'success_prob']

    # Initial swap/prob/energy arrays to fill
    swap = np.zeros(arr1.size)
    swap.resize(swap.size, 2)
    prob = np.zeros(arr1.size)
    prob.resize(swap.size, 2)

    # loop over data arrays
    for count, data in tqdm(enumerate(zip(arr1, arr2))):

        # Initialize data arrays to then fill with duplicate solutions that
        # occured
        data1 = np.copy(data[0])
        data2 = np.copy(data[1])

        # Loop over each run in this circuits anneal
        # and append the duplicate to the copied data array
        for run in data[0]:
            occ = run['occurences'] - 1
            for i in range(occ):
                data1 = np.append(data1, run)
        for run in data[1]:
            occ = run['occurences'] - 1
            for i in range(occ):
                data2 = np.append(data2, run)

        # remove fields that break the sorting for ties
        data1 = data1[fields]
        data2 = data2[fields]

        # sort and populate swap array
        data1.sort(kind='h', order=['swaps', 'energy'])
        data2.sort(kind='h', order=['swaps', 'energy'])
        swap[count] = (np.mean(data1['swaps'][:int(frac*data1.size)]),
                       np.mean(data2['swaps'][:int(frac*data2.size)])
                      )

        # sort and populate prob array (sort in reverse order since higher sp
        # is better)
        data1[::-1].sort(kind='h', order=['success_prob', 'energy'])
        data2[::-1].sort(kind='h', order=['success_prob', 'energy'])
        prob[count] = (np.mean(data1['success_prob'][:int(frac*data1.size)]),
                       np.mean(data2['success_prob'][:int(frac*data2.size)])
                      )

    # Deal with getting 0 for success prob. (since we need to log them)
    # Could combine these but rather just do separately for now
    for sp in prob:
        for col, val in enumerate(sp):
            if val == 0:
                sp[col] = 1
    prob = -np.log(prob)

    # Define % difference function for arrays
    def pdif(arr1,arr2, flip=False):
        if flip == True:
            ind = np.where(arr1 != 0)
            return (arr1[ind] - arr2[ind]) / arr1[ind] * 100, ind
        elif flip == False:
            ind = np.where(arr2 != 0)
            return (arr1[ind] - arr2[ind]) / arr2[ind] * 100, ind

    # Calculate the pdif arrays
    swapdiff, ind_s = pdif(swap[:,0], swap[:,1], flip)
    probdiff, ind_p = pdif(prob[:,0], prob[:,1], flip)

    # Plot the metric comparison for prob and swap
    print('# of circuits used for swap plot: ', len(ind_s[0]))
    print('# of circuits used for prob plot: ', len(ind_p[0]))

    plt.figure(1)
    plt.plot(prop_arr['logical_qubit'][ind_s], swapdiff, '.b') #label='{} - {}'.format(label1, label2))
    if frac == 1:
        plt.title('Full Solution Set SWAP Comparison')
    else:
        plt.title('Top {}% SWAP Comparison'.format(frac*100))
    plt.ylabel('Percent Difference Swap Count  [%]')
    plt.xlabel('Logical qubits')
    plt.grid()
    # plt.legend(loc='lower left')
    if flip == True:
        plt.savefig( join(saveloc , 'SwapDIFF_{}_{}-{}_FLIP.pdf'.format(frac, label1, label2)) )
    elif flip == False:
        plt.savefig( join(saveloc , 'SwapDIFF_{}_{}-{}.pdf'.format(frac, label1, label2)) )

    plt.figure(2)
    plt.plot(prop_arr['logical_qubit'][ind_p], probdiff, '.r') #label='{} - {}'.format(label1, label2))
    if frac == 1:
        plt.title('Full Solution Set SP Comparison')
    else:
        plt.title('Top {}% SP Comparison'.format(frac*100))
    plt.ylabel('Percent Difference ln(SP)  [%]')
    plt.xlabel('Logical qubits')
    plt.grid()
    # plt.legend(loc='lower left')
    if flip == True:
        plt.savefig( join(saveloc , 'ProbDIFF_{}_{}-{}_FLIP.pdf'.format(frac, label1, label2)) )
    elif flip == False:
        plt.savefig( join(saveloc , 'ProbDIFF_{}_{}-{}.pdf'.format(frac, label1, label2)) )



from itertools import combinations

def qiskit_judge(prop_arr, qiskit_arr, flip=False):
    '''
    Given two qiskit benchmark data arrays, will plot the %-difference for
    swaps, circuit depth and -log(success probability). You specify the labels
    to match the layout method.


    '''
    # Define where to save plots
    saveloc = Path().resolve().parent / 'benchmarks' / 'data' / 'plots' / 'Qiskit Metric Comparisons'
    saveloc = str(saveloc) # to use with os.join

    # Grab record array data names (skipping circuit)
    names = qiskit_arr.dtype.names[1:]

    # Define dtype for new record arrays
    fields = [('swap', 'i8'),
              ('depth', 'i8'),
              ('sp', 'f8')
             ]

    # Initialize new data arrays
    triv = np.zeros(qiskit_arr.size, dtype=fields)
    dense = np.zeros(qiskit_arr.size, dtype=fields)
    noise = np.zeros(qiskit_arr.size, dtype=fields)
    sabre = np.zeros(qiskit_arr.size, dtype=fields)
    qubo = np.zeros(qiskit_arr.size, dtype=fields)

    # Fiull data arrays with swap,circuit depth and sp of for all circuit runs

    # We first loop over each circuits data and then loop over the different
    # initial layout methods that were used (5 methods for each circuit)
    for count, data in enumerate(qiskit_arr):
        for method in names:

            metrics = data[method]

            if method == 'trivial':
                triv[count] = metrics
            elif method == 'dense':
                dense[count] = metrics
            elif method == 'noise':
                noise[count] = metrics
            elif method == 'sabre':
                sabre[count] = metrics
            elif method == 'qubo':
                qubo[count] = metrics
            else:
                print('Finding data broke.')

    # Turn sp into -log(sp)
    triv['sp'] = -np.log(triv['sp'])
    dense['sp'] = -np.log(dense['sp'])
    noise['sp'] = -np.log(noise['sp'])
    sabre['sp'] = -np.log(sabre['sp'])
    qubo['sp'] = -np.log(qubo['sp'])

    # Grab prop_arr indices, in order found in qiskit_arr
    # Do this to avoid mismatch in indexing prop_arr and qiskit_arr
    ind = []
    for circ in qiskit_arr['circuit']:
        ind.append(np.where(prop_arr['circuit']==circ)[0][0])

    # Set prop_arr to be ordered like indexing (and removes any circuits not
    # used in the data run)
    prop_arr = prop_arr[ind]

    # Define % difference function for arrays
    def pdif(arr1,arr2, flip=False):
        if flip == True:
            ind = np.where(arr1 != 0)
            return (arr1[ind] - arr2[ind]) / arr1[ind] * 100, ind
        elif flip == False:
            ind = np.where(arr2 != 0)
            return (arr1[ind] - arr2[ind]) / arr2[ind] * 100, ind

    # Make a list of all data arrays
    data_list = [triv,dense,noise,sabre,qubo]

    # Grab all combinations of data to plot and well as their names for labels
    plot_list = combinations(data_list, 2)
    name_list = combinations(names, 2)

    # Make all the plots
    for pair, label in zip(plot_list, name_list):

        # Define figure outside loop otherwise encounter warning
        plt.figure()
        for metric in triv.dtype.names:
            # Grab relative differences
            pdiff, ind = pdif(pair[0][metric], pair[1][metric], flip=flip)

            # Plot elements
            plt.plot(prop_arr['logical_qubit'][ind],
                     pdiff,
                     '.r',
                     label = '{} - {}'.format(*label),
                     alpha = .35
                     )
            plt.title('{} Relative Difference'.format(metric))
            plt.xlabel('Logical Qubit #', fontsize=14)
            plt.ylabel('Relative Difference', fontsize=14)
            plt.legend()
            plt.grid()
            plt.savefig( join( saveloc , '{}_Relative_{}-{}.pdf'.format(metric,*label)) )
            plt.clf()

from qiskit.transpiler import passes, PassManager, CouplingMap, Layout
from qiskit.transpiler.preset_passmanagers import level_0_pass_manager
from qiskit import QuantumCircuit

def qiskit_benchmark(bm_arr, n_p, graph, hardware_edges, backend, err_loc):
    '''
    Given a qubo benchmark run array, will return some metrics that describe how
    well QUBO/QISKIT/SABRE did at finding an initial allocation. Since we're
    comparing the algorithms from the point of view of initial allocation, we
    use the basic qiskit functionalities that don't further optimize the circuit.

    For now we take the lowest swap allocation as our choice of QUBO solution.
    '''
    # Define location of benchmark circuit qasm files
    loc = Path().resolve().parent / 'benchmarks' / 'circuits'
    loc = str(loc)

    # Define where to save results to
    saveloc = Path().resolve().parent / 'benchmarks' / 'data' / 'qiskit results'

    # Initialize array to store results in
    fields = [('circuit','U40'),
              ('trivial', 'object'),
              ('dense','object'),
              ('noise','object'),
              ('sabre','object'),
              ('qubo','object')
              ]
    qiskit_arr = np.zeros(len(bm_arr), dtype=fields)

    # Grab machine errors
    err1, err2 = read_err(err_loc, n_p)

    # Define probability matrix
    prob_m = probability_matrix(graph, err1, err2)

    # Define coupling map object
    coupling_map = CouplingMap(couplinglist=hardware_edges)

    # Define passmanger configs
    config_trivial = PassManagerConfig(coupling_map=coupling_map,
                     layout_method='trivial',
                     routing_method='basic',
                     backend_properties=backend.properties()
                                      )
    config_dense = PassManagerConfig(coupling_map=coupling_map,
                     layout_method='dense',
                     routing_method='basic',
                     backend_properties=backend.properties()
                                      )
    config_sabre = PassManagerConfig(coupling_map=coupling_map,
                     layout_method='sabre',
                     routing_method='basic',
                     backend_properties=backend.properties()
                                      )

    # # Define passmanagers
    pm_triv = level_0_pass_manager(config_trivial)
    pm_dense = level_0_pass_manager(config_dense)
    pm_sabre = level_0_pass_manager(config_sabre)

    # Loop over the circuits in benchmark array and log run data
    for count, data in enumerate(bm_arr):

        # Grab import bits in benchmark array
        circ_n = data['circuit']
        bm_metrics = data['data_array'][['swaps','energy','success_prob']]

        # add circuit name to qiskit_arr
        qiskit_arr['circuit'][count] = circ_n

        # Load the circuit
        circuit = QuantumCircuit.from_qasm_file( join(loc, circ_n) )

        # Grab the allocation for the lowest swap solution from QUBO
        bm_metrics.sort(kind='h', order=['swaps']) ##!!!!!!!!
        best_allo = data['data_array']['allocation'][0]
        ind = np.where(best_allo == 1)
        # Need allocation as a dic like this for layout method
        allo_dic = {circuit.qubits[i] : ind[1][i] for i in range(len(ind[1]))}

        # Define QUBO/noise config & passmanager (noise doesn't work when outside)
        # this loop for some reason
        config_noise = PassManagerConfig(coupling_map=coupling_map,
                             layout_method='noise_adaptive',
                             routing_method='basic',
                             backend_properties=backend.properties()
                                        )
        config_qubo = PassManagerConfig(initial_layout=Layout(allo_dic),
                         coupling_map=coupling_map,
                         routing_method='basic',
                         backend_properties=backend.properties()
                                        )
        pm_noise = level_0_pass_manager(config_noise)
        pm_qubo = level_0_pass_manager(config_qubo)

        # Create list of passmanagers
        pm_list = [pm_triv, pm_dense, pm_noise, pm_sabre, pm_qubo]

        # Run all passmanagers on circuit and extract useful metrics
        for num, pm in tqdm(enumerate(pm_list)):

            # Run circuit through pm and grab # swaps and depth
            pass_circ = pm.run(circuit)

            # If circuit didn't need swaps, won't have in dictionary so catch
            # with try/except
            try:
                swap = pass_circ.count_ops()['swap']
            except KeyError:
                swap= 0

            # Parse the transpiled circuit to grab which gate ops. act on what
            # qubits to calculate the success prob. of the circuit
            g1, g2 = parse_transpile(pass_circ, n_p)
            swap2cx(g2) # converts swaps to cx's

            # Calculate success probability
            prob = successprob(assign_dic = None,
                               prob_m = prob_m,
                               one_gates_count = gatecount(g1),
                               two_gates_count = gatecount(g2),
                               hardware = True
                               )

            # Decompose  swaps to CNOT's and grab depth
            unroll_pm = PassManager(Unroller(['u1', 'u2', 'u3', 'cx']))
            pass_circ = unroll_pm.run(pass_circ)
            depth = pass_circ.depth()

            # Grab CX count
            cx = pass_circ.count_ops()['cx']

            # Save data to correct place in qiskit_arr
            if num == 0: # Trivial pm
                qiskit_arr['trivial'][count] = (cx, swap, depth, prob)
            elif num == 1: # Dense pm
                qiskit_arr['dense'][count] = (cx, swap, depth, prob)
            elif num == 2: # noise pm
                qiskit_arr['noise'][count] = (cx, swap, depth, prob)
            elif num == 3: # sabre pm
                qiskit_arr['sabre'][count] = (cx, swap, depth, prob)
            elif num == 4: # QUBO pm
                qiskit_arr['qubo'][count] = (cx, swap, depth, prob)
            else:
                print('Was not able to save some data.')

    # Dump the array containing the data from runs
    log_dump(qiskit_arr, saveloc, 'qiskit_results')


def hardware_edges(hardware_graph):
    '''
    Given a string which is the name of a certain achitecture graph, will return
    the edge list for that hardware.
    '''
    sycamore_edges = [(0, 6), (1, 6), (1, 7), (2, 7), (2, 8), (3, 8), (3, 9), (4, 9), (4, 10), (5, 10), (5, 11),
                      (6, 12), (6, 13), (7, 13), (7, 14), (8, 14), (8, 15), (9, 15), (9, 16), (10, 16), (10, 17), (11, 17),
                      (12, 18), (13, 18), (13, 19), (14, 19), (14, 20), (15, 20), (15, 21), (16, 21), (16, 22), (17, 22), (17, 23),
                      (18, 24), (18, 25), (19, 25), (19, 26), (20, 26), (20, 27), (21, 27), (21, 28), (22, 28), (22, 29), (23, 29),
                      (24, 30), (25, 30), (25, 31), (26, 31), (26, 32), (27, 32), (27, 33), (28, 33), (28, 34), (29, 34), (29, 35),
                      (30, 36), (30, 37), (31, 37), (31, 38), (32, 38), (32, 39), (33, 39), (33, 40), (34, 40), (34, 41), (35, 41),
                      (36, 42), (37, 42), (37, 43), (38, 43), (38, 44), (39, 44), (39, 45), (40, 45), (40, 46), (41, 46), (41, 47),
                      (42, 48), (42, 49), (43, 49), (43, 50), (44, 50), (44, 51), (45, 51), (45, 52), (46, 52), (46, 53), (47, 53)
                      ]
    aspen_edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7),
                   (0, 8), (3, 11), (4, 12), (7, 15), (8, 9), (9, 10),
                   (10, 11), (11, 12), (12, 13), (13, 14), (14, 15)
                  ]
    tokyo_edges = [(0, 1), (1, 2), (2, 3), (3, 4),
                   (0, 5), (1, 6), (1, 7), (2, 6), (2, 7), (3, 8), (3, 9), (4, 8), (4, 9),
                   (5, 6), (6, 7), (7, 8), (8, 9),
                   (5, 10), (5, 11), (6, 10), (6, 11), (7, 12), (7, 13), (8, 12), (8, 13), (9, 14),
                   (10, 11), (11, 12), (12, 13), (13, 14),
                   (10, 15), (11, 16), (11, 17), (12, 16), (12, 17), (13, 18), (13, 19), (14, 18), (14, 19),
                   (15, 16), (16, 17), (17, 18), (18, 19)
                  ]
    rochester_edges = [(0, 1), (1, 2), (2, 3), (3, 4),
                       (0, 5), (4, 6), (5, 9), (6, 13),
                       (7, 8), (8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15),
                       (7, 16), (11, 17), (15, 18), (16, 19), (17, 23), (18, 27),
                       (19, 20), (20, 21), (21, 22), (22, 23), (23, 24), (24, 25), (25, 26), (26, 27),
                       (21, 28), (25, 29), (28, 32), (29, 36),
                       (30, 31), (31, 32), (32, 33), (33, 34), (34, 35), (35, 36), (36, 37), (37, 38),
                       (30, 39), (34, 40), (38, 41), (39, 42), (40, 46), (41, 50),
                       (42, 43), (43, 44), (44, 45), (45, 46), (46, 47), (47, 48), (48, 49), (49, 50),
                       (44, 51), (48, 52)
                      ]
    mel_edges = [[1,0], [1,2], [2,3], [4,3], [4,10], [5,4], [5,6], [5,9], [6,8], [7,8], [9,8], [9,10],
		         [11,10], [11,3], [11,12], [12,2], [13,1], [13,12], [14,0], [14,13]
		        ]
    qbit100_edges = [(0, 10), (0, 1), (1, 11), (1, 2), (2, 12), (2, 3), (3, 13), (3, 4), (4, 14), (4, 5), (5, 15), (5, 6), (6, 16), (6, 7), (7, 17), (7, 8), (8, 18), (8, 9), (9, 19), (10, 20), (10, 11), (11, 21), (11, 12), (12, 22), (12, 13), (13, 23), (13, 14), (14, 24), (14, 15), (15, 25), (15, 16), (16, 26), (16, 17), (17, 27), (17, 18), (18, 28), (18, 19), (19, 29), (20, 30), (20, 21), (21, 31), (21, 22), (22, 32), (22, 23), (23, 33), (23, 24), (24, 34), (24, 25), (25, 35), (25, 26), (26,36), (26, 27), (27, 37), (27, 28), (28, 38), (28, 29), (29, 39), (30, 40), (30, 31), (31, 41), (31, 32), (32, 42), (32, 33), (33, 43), (33, 34), (34, 44), (34, 35), (35, 45), (35, 36), (36, 46), (36, 37), (37, 47), (37, 38), (38, 48), (38, 39), (39, 49), (40, 50), (40, 41), (41, 51), (41, 42), (42, 52), (42, 43), (43, 53), (43, 44), (44, 54), (44, 45), (45, 55), (45, 46), (46, 56), (46, 47), (47, 57), (47, 48), (48, 58), (48, 49), (49, 59), (50, 60), (50, 51), (51, 61), (51, 52), (52, 62), (52,53), (53, 63), (53, 54), (54, 64), (54, 55), (55, 65), (55, 56), (56, 66), (56, 57), (57, 67), (57, 58), (58, 68), (58, 59), (59, 69), (60, 70), (60, 61), (61, 71), (61, 62), (62, 72), (62, 63), (63, 73), (63, 64), (64, 74), (64, 65), (65, 75), (65, 66), (66, 76), (66, 67), (67, 77), (67, 68), (68, 78), (68, 69), (69, 79), (70, 80), (70, 71), (71, 81), (71, 72), (72, 82), (72, 73), (73, 83), (73, 74), (74, 84), (74, 75), (75, 85), (75, 76), (76, 86), (76, 77), (77, 87), (77, 78), (78,88), (78, 79), (79, 89), (80, 90), (80, 81), (81, 91), (81, 82), (82, 92), (82, 83), (83, 93), (83, 84), (84, 94), (84, 85), (85, 95), (85, 86), (86, 96), (86, 87), (87, 97), (87, 88), (88, 98), (88, 89), (89, 99), (90, 91), (91, 92), (92, 93), (93, 94), (94, 95), (95, 96), (96, 97), (97, 98), (98, 99)
    ]
    if hardware_graph == 'aspen' or hardware_graph == '16QBT':
        return aspen_edges
    elif hardware_graph == 'tokyo' or hardware_graph == '20QBT':
        return tokyo_edges
    elif hardware_graph == 'rochester' or hardware_graph == '53QBT':
        return rochester_edges
    elif hardware_graph == 'sycamore' or hardware_graph == '54QBT':
        return sycamore_edges
    elif hardware_graph == 'melbourne':
        return mel_edges
    elif hardware_graph == '100qbit':
        return qbit100_edges
    else:
        print('Unexpected input, please try again.')


from qiskit.transpiler.passes import Unroller
def queko_qiskit_bm(queko_bm):
    '''
    Given the benchmark arrays for using QUBO on QUEKO benchmark circuits,
    will use qiskit to compile the circuit using the QUBO initial allocation
    and dump a log of the results of that process.
    '''
    # Define location of benchmark circuit qasm files
    loc = Path().resolve().parent / 'benchmarks' / '_QUEKO' / 'BSS'
    loc = str(loc)
    # Define where to save results to
    saveloc = Path().resolve().parent / 'benchmarks' / 'data' / 'queko results'

    # Initialize queko results array
    fields = [('circuit','U40'),
              ('hardware', 'U40'),
              ('swaps','i8'),
              ('optimal_depth', 'i8'),
              ('depth','i8')
              ]
    queko_arr = np.zeros(queko_bm.size, dtype=fields)

    for count, data in tqdm(enumerate(queko_bm)):

        # Grab relevant data from bm_array
        circ_n = data['circuit']
        bm_metrics = data['data_array'][['swaps','energy','success_prob']]

        # Grab useful data from QUEKO circuit name
        hardware, optimal_depth, ratio, label = circ_n.split('_')
        optimal_depth = int(optimal_depth.strip('CYC'))

        # Load the circuit
        circuit = QuantumCircuit.from_qasm_file( join(loc, circ_n) )

        # Grab the hardware graph edges and define couplingmap object
        edges = hardware_edges(hardware)
        # We need to generate a list also containing the reverse edges
        # as CouplingMap treats the edges you give it as directed
        edge_list = []
        edge_list += edges
        for edge in edges:
            edge_list.append( (edge[-1], edge[0]) )

        coupling_map = CouplingMap(edge_list)

        # Grab the allocation for the lowest swap solution from QUBO
        bm_metrics.sort(kind='h', order=['swaps']) ##!!!!!
        best_allo = data['data_array']['allocation'][0]
        ind = np.where(best_allo == 1)
        # Need allocation as a dic like this for layout method
        allo_dic = {circuit.qubits[i] : ind[1][i] for i in range(len(ind[1]))}

        # Define QUBO/noise config & passmanager
        # We use stochastic routing because that's what QUEKO paper used
        config_qubo = PassManagerConfig(initial_layout=Layout(allo_dic),
                         coupling_map=coupling_map,
                         routing_method='stochastic'
                                        )

        pm_qubo = level_0_pass_manager(config_qubo)

        # Run circuit through pm and grab # swaps
        pass_circ = pm_qubo.run(circuit)
        try:
            swap = pass_circ.count_ops()['swap']
        except KeyError:
            swap = 0

        # Decompose  swaps to CNOT's and grab depth
        unroll_pm = PassManager(Unroller(['u1', 'u2', 'u3', 'cx']))
        pass_circ = unroll_pm.run(pass_circ)
        depth = pass_circ.depth()

        queko_arr[count] = (circ_n, hardware, swap, optimal_depth, depth)

    log_dump(queko_arr, saveloc, 'qiskit-QUEKO-results')


def queko_plot(bntf_qiskit, bss_qiskit, bntf_tket, bss_tket):
    '''
    Given the data from a queko compilation run (both through tket and qiskit) will recreate the plots from the queko
    paper, using QUBO initial mappings.
    '''
    # Define location to save results
    saveloc = Path().resolve().parent / 'benchmarks '/ 'data' / 'plots' / 'QUEKO plots'
    saveloc = str(saveloc)

    # Define domains to group data
    dom_bntf = np.arange(0,180,10)
    dom_bss = np.arange(0,360,10)

    # Get depth ratio data
    qiskit_ratio_bntf = bntf_qiskit['depth'] / bntf_qiskit['optimal_depth']
    qiskit_ratio_bss = bss_qiskit['depth'] / bss_qiskit['optimal_depth']
    tket_ratio_bntf = bntf_tket['depth'] / bntf_tket['optimal_depth']
    tket_ratio_bss = bss_tket['depth'] / bss_tket['optimal_depth']

    # Grab average of every 10 data points
    mean_qiskit_bntf = []
    mean_qiskit_bss = []
    mean_tket_bntf = []
    mean_tket_bss = []

    for bntf, arr in zip([mean_qiskit_bntf, mean_tket_bntf], [qiskit_ratio_bntf, tket_ratio_bntf]):
        for step in dom_bntf:
            mean = np.mean(arr[step:step+10])
            bntf.append(mean)

    for bntf, arr in zip([mean_qiskit_bss, mean_tket_bss], [qiskit_ratio_bss, tket_ratio_bss]):
        for step in dom_bss:
            mean = np.mean(arr[step:step+10])
            bntf.append(mean)

    # Define optimal depth plot domain for bntf/bss
    plot_dom_bntf = np.arange(5,50,5)
    plot_dom_bss = np.arange(100,1000,100)

    # Plot tket and qiskit data
    # BNTF plots grouped by hardware graph

    # Aspen data
    plt.figure()
    plt.plot(bntf_qiskit['optimal_depth'][:90], qiskit_ratio_bntf[:90], linestyle='',
             marker='+', color='magenta', alpha=.65, label='Qiskit')
    plt.plot(plot_dom_bntf, mean_qiskit_bntf[:9],
             color='magenta', linestyle='-')

    plt.plot(bntf_tket['optimal_depth'][:90], tket_ratio_bntf[:90], linestyle='',
             marker='x', color='lime', alpha=.65, label='t|ket>')
    plt.plot(plot_dom_bntf, mean_tket_bntf[:9],
             color='lime', linestyle='-')

    plt.ylim(1,10)
    plt.xlim(0,50)
    plt.ylabel('Depth Ratio')
    plt.xlabel('Optimal Depth')
    plt.grid()
    plt.legend(loc='upper right')
    plt.savefig(join(saveloc,'aspen.pdf'), dpi=900)
    plt.clf()

    # Sycamore data
    plt.figure()
    plt.plot(bntf_qiskit['optimal_depth'][90:], qiskit_ratio_bntf[90:], linestyle='',
             marker='+', color='magenta', alpha=.65, label='Qiskit')
    plt.plot(plot_dom_bntf, mean_qiskit_bntf[9:],
             color='magenta', linestyle='-')

    plt.plot(bntf_tket['optimal_depth'][90:], tket_ratio_bntf[90:], linestyle='',
             marker='x', color='lime', alpha=.65, label='t|ket>')
    plt.plot(plot_dom_bntf, mean_tket_bntf[9:],
             color='lime', linestyle='-')

    plt.ylim(1,16)
    plt.xlim(0,50)
    plt.ylabel('Depth Ratio')
    plt.xlabel('Optimal Depth')
    plt.grid()
    plt.legend(loc='upper right')
    plt.savefig(join(saveloc,'sycamore.pdf'), dpi=900)
    plt.clf()

    # BSS plots grouped by tket/qiskit data

    # tket BSS
    plt.figure()

    plt.plot(bss_tket['optimal_depth'][180:270], tket_ratio_bss[180:270], linestyle='',
             marker='x', color='blue', alpha=.75, label='Rochester')
    plt.plot(plot_dom_bss, mean_tket_bss[18:27],
             color='blue', linestyle='-')

    plt.plot(bss_tket['optimal_depth'][270:360], tket_ratio_bss[270:360], linestyle='',
             marker='+', color='red', alpha=.75, label='Sycamore')
    plt.plot(plot_dom_bss, mean_tket_bss[27:36],
             color='red', linestyle='-')

    plt.plot(bss_tket['optimal_depth'][90:180], tket_ratio_bss[90:180], linestyle='',
             marker='o', markeredgecolor='lime', markerfacecolor='none', alpha=.75, label='Tokyo')
    plt.plot(plot_dom_bss, mean_tket_bss[9:18],
             color='lime', linestyle='-')

    plt.plot(bss_tket['optimal_depth'][:90], tket_ratio_bss[:90], linestyle='',
             marker='d', markeredgecolor='chocolate', markerfacecolor='none', alpha=.75, label='Aspen-4')
    plt.plot(plot_dom_bss, mean_tket_bss[:9],
             color='chocolate', linestyle='-')

    plt.ylim(1,16)
    plt.xlim(0,1000)
    plt.ylabel('Depth Ratio')
    plt.xlabel('Optimal Depth')
    plt.grid()
    plt.legend(loc='upper right')
    plt.savefig(join(saveloc,'tket.pdf'), dpi=900)
    plt.clf()

    # qiskit BSS
    plt.figure()

    plt.plot(bss_qiskit['optimal_depth'][180:270], qiskit_ratio_bss[180:270], linestyle='',
             marker='x', color='blue', alpha=.75, label='Rochester')
    plt.plot(plot_dom_bss, mean_qiskit_bss[18:27],
             color='blue', linestyle='-')

    plt.plot(bss_qiskit['optimal_depth'][270:360], qiskit_ratio_bss[270:360], linestyle='',
             marker='+', color='red', alpha=.75, label='Sycamore')
    plt.plot(plot_dom_bss, mean_qiskit_bss[27:36],
             color='red', linestyle='-')

    plt.plot(bss_qiskit['optimal_depth'][90:180], qiskit_ratio_bss[90:180], linestyle='',
             marker='o', markeredgecolor='lime', markerfacecolor='none', alpha=.75, label='Tokyo')
    plt.plot(plot_dom_bss, mean_qiskit_bss[9:18],
             color='lime', linestyle='-')

    plt.plot(bss_qiskit['optimal_depth'][:90], qiskit_ratio_bss[:90], linestyle='',
             marker='d', markeredgecolor='chocolate', markerfacecolor='none', alpha=.75, label='Aspen-4')
    plt.plot(plot_dom_bss, mean_qiskit_bss[:9],
             color='chocolate', linestyle='-')

    plt.ylim(1,16)
    plt.xlim(0,1000)
    plt.ylabel('Depth Ratio')
    plt.xlabel('Optimal Depth')
    plt.grid()
    plt.legend(loc='upper right')
    plt.savefig(join(saveloc,'qiskit.pdf'), dpi=900)
    plt.clf()


def qiskit_full_anneal_compile(bm_data, circ_n, method='basic'):
    '''
    For the specific circuit anneal run, will feed every allocation through
    qiskit and grab the compiled circuits swap count and depth.

    '''
    # Location where to save data to
    saveloc = Path().resolve().parent / 'benchmarks' / 'data' / 'qiskit results'

    # Location of benchmark circuits
    loc = Path().resolve().parent / 'benchmarks' / 'circuits'
    loc = str(loc)

    # Initialize record array
    fields = [('allocation', 'object'),
              ('energy', 'f8'),
              ('success_prob', 'f8'),
              ('naive_swap', 'i8'),
              ('compiled_CX','i8'),
              ('compiled_swap', 'i8'),
              ('compiled_depth', 'i8')
             ]
    array = np.zeros(len(bm_data), dtype=fields)

    # Loop over each anneal in data and compile using lvl-0 optimization
    # using the allocation from each anneal as the initial layout
    for count, datum in tqdm(enumerate(bm_data)):
        # Grab our metrics from data
        allo = datum['allocation']
        energy = datum['energy']
        naive_swap = datum['swaps']
        prob = datum['success_prob']

        # Load the circuit
        circuit = QuantumCircuit.from_qasm_file(join(loc, circ_n))

        # Grab the hardware graph edges and define couplingmap object
        edges = hardware_edges('melbourne')
        # We need to generate a list also containing the reverse edges
        # as CouplingMap treats the edges you give it as directed
        edge_list = []
        edge_list += edges
        for edge in edges:
            edge_list.append( (edge[-1], edge[0]) )

        coupling_map = CouplingMap(edge_list)

        # Grab where logical qubits got allocated to
        ind = np.where(allo == 1)
        # Need allocation as a dic like this for layout method
        allo_dic = {circuit.qubits[i] : ind[1][i] for i in range(len(ind[1]))}

        # Define QUBO config & passmanager
        # Specify the routing method in the argument of the function
        config_qubo = PassManagerConfig(initial_layout=Layout(allo_dic),
                                        coupling_map=coupling_map,
                                        routing_method=method
                                       )
        pm_qubo = level_0_pass_manager(config_qubo)

        # Run circuit through pm and grab # swaps
        pass_circ = pm_qubo.run(circuit)
        try:
            swap = pass_circ.count_ops()['swap']
        except KeyError:
            swap = 0

        # Decompose  swaps to CNOT's and grab depth
        unroll_pm = PassManager(Unroller(['u1', 'u2', 'u3', 'cx']))
        pass_circ = unroll_pm.run(pass_circ)
        depth = pass_circ.depth()
        cx = pass_circ.count_ops()['cx']

        # Populate record array
        array[count] = (allo, energy, prob, naive_swap, cx, swap, depth)


    log_dump(array, saveloc, circ_n + '-full-anneal-compiled-with-' + method)


## Note pytket requires macOS or linux (not windows)
## so keep the below commented out on windows or you'll
## get an error when importing from this script

# from pytket.circuit import Qubit, Node, OpType
# from pytket.routing import QubitMap, place_with_map, Architecture, route
# from pytket.qasm import circuit_from_qasm
# from pytket.transform import Transform
#
# from pytket.routing import GraphPlacement, LinePlacement
# from pytket.device import Device
#
# def tket_queko_bm(queko_bm):
#     '''
#     Given a QUEKO benchmark run from QUBO, will route the circuits using the QUBO lowest swap allocation
#     as the initial allocation, using the tket router.
#     '''
#     # Define local paths to find / save data
#     loc = '/home/bdury/qubitallocation/benchmarks/_QUEKO/BSS/'
#     saveloc = '/home/bdury/qubitallocation/benchmarks/data/queko results/'
#
#     # Initialize queko results array
#     fields = [('circuit','U40'),
#               ('hardware', 'U40'),
#               ('swaps','i8'),
#               ('optimal_depth', 'i8'),
#               ('depth','i8')
#               ]
#     queko_arr = np.zeros(queko_bm.size, dtype=fields)
#
#     for count, data in tqdm(enumerate(queko_bm)):
#
#         # Grab relevant data from bm_array
#         circ_n = data['circuit']
#         bm_metrics = data['data_array'][['swaps','energy','success_prob']]
#
#         # Grab useful data from QUEKO circuit name
#         hardware, optimal_depth, ratio, label = circ_n.split('_')
#         optimal_depth = int(optimal_depth.strip('CYC'))
#
#         # Load the circuit
#         circuit = circuit_from_qasm(join(loc, circ_n))
#
#         # Define architecture for circuit
#         edges = hardware_edges(hardware)
#         hardware_graph = Architecture(edges)
#
#         # Grab the allocation for the lowest swap solution from QUBO
#         bm_metrics.sort(kind='h', order=['swaps']) ##!!!!!
#         best_allo = data['data_array']['allocation'][0]
#         logical, phys = np.where(best_allo == 1)
#         # Put into form tket expects
#         qubo_map = QubitMap()
#         for q_i, n_i in zip(logical, phys):
#             qubo_map[Qubit(q_i)] = Node(n_i)
#
# #         # Try out graph placement
# #         device = Device(architecture=hardware_graph)
# #         graph_map = GraphPlacement(device)
# #         graph_map.place(circuit)
#         # Assign mapping to tket circuit
#         place_with_map(circuit, qubo_map)
#
#         # Route the circuit using tket's built-in routing
#         routed_circ = route(circuit, hardware_graph)
#
#         # Grab swaps from circuit before decomposing to CNOT's
#         swap = routed_circ.n_gates_of_type(OpType.SWAP)
#
#         # Transform swaps to CNOT before grabbing circuit depth
#         Transform.DecomposeSWAPtoCX().apply(routed_circ)
#         depth = routed_circ.depth()
#
#         # Save run data to array
#         queko_arr[count] = (circ_n, hardware, swap, optimal_depth, depth)
#
#     log_dump(queko_arr, saveloc, 'tket-QUEKO-results')
#
#
# def tket_bm(qubo_bm):
#     '''
#     Given a QUBO benchmark run, will benchmark the QUBO allocations against tkets initial mappers using tkets
#     default router.
#     '''
#     # Define local paths to find / save data
#     loc = '/home/bdury/qubitallocation/benchmarks/circuits/'
#     saveloc = '/home/bdury/qubitallocation/benchmarks/data/tket results/'
#
#     # Initialize queko results array
#     fields = [('circuit','U40'),
#               ('line', 'object'),
#               ('qubo', 'object'),
#               ('graph', 'object')
#               ]
#     tket_arr = np.zeros(qubo_bm.size, dtype=fields)
#
#     # Define architecture for circuits
#     edges = hardware_edges('melbourne')
#     hardware_graph = Architecture(edges)
#
#     # Define routing parameters
#     basic_parameters = dict(bridge_lookahead = 0, bridge_interactions = 0, swap_lookahead = 0)
#
#     for count, data in tqdm(enumerate(qubo_bm)):
#
#         # Grab relevant data from bm_array
#         circ_n = data['circuit']
#         bm_metrics = data['data_array'][['swaps','energy','success_prob']]
#
#         # Add circuit to tket array
#         tket_arr[count]['circuit'] = circ_n
#
#         # Load the circuit
#         circuit = circuit_from_qasm(join(loc, circ_n))
#
#         # Grab the allocation for the lowest swap solution from QUBO
#         bm_metrics.sort(kind='h', order=['swaps']) ##!!!!!
#         best_allo = data['data_array']['allocation'][0]
#         logical, phys = np.where(best_allo == 1)
#         # Put into form tket expects
#         qubo_map = QubitMap()
#         for q_i, n_i in zip(logical, phys):
#             qubo_map[Qubit(q_i)] = Node(n_i)
#
#         # Define tket initial allocation algs to be used
#         device = Device(architecture=hardware_graph)
#         line_map = LinePlacement(device)
#         graph_map = GraphPlacement(device)
#
#         # Route the circuit for each initial allocation method specified
#         for method in tket_arr.dtype.names[1:]: # !!!!!!!!!!!!!! Switch from this to '1:'
#
#             circ = circuit.copy()
#
#             # Map qubits according to method
#             if method == 'qubo':
#                 place_with_map(circ, qubo_map)
#             elif method == 'line':
#                 line_map.place(circ)
#             elif method == 'graph':
#                 graph_map.place(circ)
#             else:
#                 print('Invalid method encountered - investigate your tket array dtype.')
#
#             # Route the circuit with basic parameters
#             routed = route(circ, hardware_graph, **basic_parameters)
#
#             # Grab data from circuit
#             swap = routed.n_gates_of_type(OpType.SWAP)
#
#             Transform.DecomposeSWAPtoCX().apply(routed)
#             CX = routed.n_gates_of_type(OpType.CX)
#             depth = routed.depth()
#
#             # Append data to appropriate method
#             tket_arr[count][method] = (CX, swap, depth)
#
#     log_dump(tket_arr, saveloc, 'tket-results')
