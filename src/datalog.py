'''
Contains functions used to scructure data returned from the main code and
qreg [16];

'''
import numpy as np
import matplotlib.pyplot as plt
from re import sub
from os.path import join
from qiskit.transpiler import PassManagerConfig, CouplingMap, Layout
from qiskit.transpiler.preset_passmanagers import level_1_pass_manager
from QUBOfuncs import *
from matrix_funcs import *

from tqdm import tqdm


def data_array(n_c, n_p, response, graph, one_gates, two_gates, fname, QUEKO=False):
    '''
    Creates a structured array containing the relevant data for a given run.

    [[(allo_arr_1, energy_1,
    num_oc_1, num_swaps_1, succprob_1)]
     .
     .
     .
     [[(allo_arr_N, energy_N, num_oc_N, num_swaps_N, succprob_N)]]

     with N = # unique assignments and the columns ordered according to:

     [('allocation', 'object'), ('energy', 'f8'), ('occurences', 'i8'), ('swaps', 'i8'), ('success_prob', 'f8')]

    Gate errors are assumed to be distributed uniform from r_start -> r_end.
    At some point these will be replaced with the real error.

    n_c & n_p:
        Number of circuit and physical qubits respectively.

    response:
        The set of samples generated from "nl.sampler.SimulatedAnnealingSampler".

    graph:
        networkx graph instance for the hardware graph being used.

    one_gates & two_gates:
        Gate dictionaries returned by "qasm_parse" from QUBOfuncs.py.

    fname:
        File location of the IBM QC calibration csv.

    '''
    # Very lazy if-else to avoid calculating useless things for a use-case
    if QUEKO==False:
        # Grab the hardware qubit error rates
        err1, err2 = read_err(fname, n_p)

        # Calculate necessary matrices, shortest paths and assignments
        err_m = error_matrix(graph, err1, err2)
        prob_m = probability_matrix(graph, err1, err2)
    else:
        pass

    paths = nx.shortest_path(graph)
    assign = assignment(response, n_c, n_p)

    # Initialize data array as a structured array.
    arraysize = len(assign)
    fields = [('allocation', 'object'),
              ('energy', 'f8'),
              ('occurences', 'i8'),
              ('swaps', 'i8'),
              ('success_prob', 'f8')
             ]
    data_array = np.zeros(arraysize, dtype = fields)

    # Loop over all assignments to fill data array with appropriate values.
    # In the process of the loop the swap_num & success_prob are calculated
    # for a given assignment.

    for count, assigned in enumerate(assign):
        assign_dic = assigned[0]
        assign_graph = assigned[1]
        energy = assigned[2]
        num_oc = assigned[3]

        circ_phys = assign_dic['circ'] # Grab {Circ:Phys} assignment dictionary
        phys_circ = assign_dic['phys'] # Grab {Phys:Circ} assignment dictionary

        # Multiple logical assigned to same physical
        # Multiple physical assigned to same logical
        # or didn't assign the right amount of qbits

        if len(circ_phys.values()) != len(set(circ_phys.values())) \
        or len(phys_circ.values()) != len(set(phys_circ.values())) \
        or len(circ_phys) != n_c or len(phys_circ) != n_c:         # Both should be n_c otherwise implying they're not the same dict.

        # If the assignment was unphysical, we define the success probability to be 0
        # as well as the number of swaps necessary (circuit can never execute anyways)
            succprob = 0
            num_swaps = 0

        else: # correctly assigned
            # When doing QUEKO benchmarks, don't have error-rates for hardware
            # graphs it's using so just make probabilities 1
            if QUEKO == False:
                succprob = successprob(circ_phys, prob_m, gatecount(one_gates), gatecount(two_gates))
            else:
                succprob = 1

            num_swaps = swapcount(circ_phys, paths, gatecount_matrix(n_c, one_gates, two_gates))
        # Fill the structured array with relevant data
        data_array[count] = (assign_graph, energy, num_oc, num_swaps, succprob)

    return data_array


def log_array(penalty, qasmfile, data_array, time = None, benchmark = False):
    '''
    Creates a structured array in a format convenient to dump data from anneal
    runs.

    penalty:
        Penalty coefficient dictionary.

    qasmfile:
        Name of QASM file being used to represent quantum circuit.

    data_array:
        Output of data_array.
    '''
    fields = [('penalty', 'object'),
              ('circuit', 'U40'),
              ('data_array', 'object')
             ]

    bm_fields = [('penalty', 'object'),
                 ('time', 'f8'),
                 ('circuit', 'U40'),
                 ('data_array', 'object')
                ]

    if benchmark == True:
        log_array = np.array([(penalty, time, qasmfile, data_array)], dtype = bm_fields)

    if benchmark == False:
        log_array = np.array([(penalty, qasmfile, data_array)], dtype = fields)

    return log_array


def log_dump(log, directory, prefix):
    '''
    Dumps the given log array in the specified directory.

    log:
        Array to be dumped.

    Directory:
        Place to save the log as a .npy file. Expected format is a Pathlib Path
        object.

    prefix:
        Prefix to write infront of datetime string for the file name.

    '''
    # Grab current date & time (second accuracy) and convert any ':' --> '.'
    # since the colons are treated weirdly by np.save().
    datetimestring = sub(':', '.', datetime.now().strftime("%Y-%m-%d-%X"))

    # Set filename structure
    fname = prefix + '_' + datetimestring
    # Use os.path.join to link directory and name
    np.save(join( str(directory), fname), log)


# def plot_qiskit(compare_arr, pen, directory, circ, basic_swap):
#     '''
#     Make a bunch of plots to visualize the qiskit comparison. Will save to the
#     given directory.
#
#     compare_arr:
#
#     pen:
#         Penalty dictionary for this comparison run.
#
#     directory:
#         Directory where you want the plots saved, should be a string.
#
#     circ:
#         Circuit used for this comparison, should be a string
#     '''
#     # Grab penlaty values from penalty dic and turn into string
#     str_penalty = ''
#     for val in pen.values():
#         str_penalty += str(val) + '_'
#
#     # Set filepaths and names for each figure
#     saveloc = 'C:/work/qubitallocation/heuristic_tests/' + directory # Change the base location if on diff pc
#     fname_swap_h = saveloc + str_penalty + circ + '_SWAP_HIST.png'
#     fname_energy = saveloc + str_penalty + circ + '_ENERGY_SCATTER.png'
#     fname_swap = saveloc + str_penalty + circ + '_SWAP_SCATTER.png'
#     fname_prob = saveloc + str_penalty + circ + '_PROB_SCATTER.png'
#     fname_colour = saveloc + str_penalty + circ + '_COLOUR_SCATTER_BY-LOG-PROB.png'
#
#     # Grab each of the data columns from the compare_array for use in plotting
#     prob = compare_arr['success_prob']
#     en = compare_arr['energy']
#     sw = compare_arr['swap']
#     q_sw = compare_arr['q_swap']
#     #phys = compare_arr['phys_tuple']
#
#     # all of the desired plots, as well as calls to save them.
#
#     plt.figure(1)
#     num, bin, patch = plt.hist(q_sw, bins=20, rwidth=.85, facecolor='g', histtype="bar", alpha=.7)
#     plt.axvline(x=basic_swap, color='r', linestyle='--', label='Trivial Layout')
#     plt.title('Qiskit Swap Histogram', fontsize=16)
#     plt.grid(alpha=.6)
#     plt.ylabel('Occurences', fontsize=14)
#     plt.xlabel('Swap count', fontsize=14)
#     plt.legend(loc='upper right')
#     plt.savefig(fname_swap_h, dpi=300)
#
#     plt.figure(2)
#     plt.plot(en, q_sw, 'g+', markersize=2)
#     plt.title('Energy Scatterplot', fontsize=16)
#     plt.grid(alpha=.6)
#     plt.ylabel('Qiskit Swap', fontsize=14)
#     plt.xlabel('Energy', fontsize=14)
#     plt.savefig(fname_energy, dpi=300)
#
#     plt.figure(3)
#     plt.plot(sw, q_sw, 'g+', markersize=2)
#     plt.title('Swap Scatterplot', fontsize=16)
#     plt.grid(alpha=.6)
#     plt.ylabel('Qiskit Swap', fontsize=14)
#     plt.xlabel('Swap count', fontsize=14)
#     plt.savefig(fname_swap, dpi=300)
#
#     plt.figure(4)
#     plt.plot(-np.log(prob), q_sw, 'g+', markersize=2)
#     plt.title('Probability Scatterplot', fontsize=16)
#     plt.grid(alpha=.6)
#     plt.ylabel('Qiskit Swap', fontsize=14)
#     plt.xlabel('-ln(success_prob)', fontsize=14)
#     plt.savefig(fname_prob, dpi=300)
#
#     # # Iterate over the physical allocations in 'phys' and calculate how similar
#     # # each allocation is to the lowest energy allocation and create a colour
#     # # plot from those similarity %'s
#     #
#     # c_list = []
#     # compare_arr['energy'].sort(axis=0)
#     # ref_allo = compare_arr['phys_tuple'][0]
#     #
#     # for phys_allo in phys:
#     #     simularity = np.sum(phys_allo[0] == ref_allo[0]) / len(ref_allo[0])
#     #     c_list.append(simularity)
#
#     # Instead of doing above we just use success probability, more information
#     # than the similarity of allocations, since they're all very similar to
#     # each other
#
#     # Sort like this just to make sure that all the arrays are plotted in the
#     # same ordering, so that the colouring is applied to the right value pairs
#     compare_arr['success_prob'] = -compare_arr['success_prob']
#     compare_arr.sort(kind='h', order=['success_prob', 'energy'])
#     compare_arr['success_prob'] = -compare_arr['success_prob']
#     c_list = -np.log(prob.ravel())
#
#     # Plot the coloured energy histogram and save it.
#     plt.figure(5)
#     cm = plt.cm.get_cmap('Reds')
#     sc = plt.scatter(en.ravel(), q_sw.ravel(), c=c_list, cmap=cm, marker='.', s=25, alpha =.5)
#     plt.title('Energy Scatterplot', fontsize=16)
#     plt.grid(alpha=.6)
#     plt.ylabel('Qiskit Swap', fontsize=14)
#     plt.xlabel('Energy', fontsize=14)
#     plt.colorbar(sc)
#     plt.savefig(fname_colour, dpi=300)
#
#
# def compare_qiskit_prob(n_c, graph_edges, backend_prop, circuit, data, err1, err2, n_p):
#     '''
#     Same as the above compare qiskit function, but returns a calculation of the
#     success probability for the compiled qiskit circuit. Was used for testing
#     purposes, might be needed again so keeping for now.
#     '''
#     # Initialize record array
#     fields = [('allocation', 'object'), ('energy', 'f8'), ('success_prob', 'f8'), ('swap', 'i8'), ('q_swap', 'i8'), ('phys_tuple', 'object')]
#     array = np.zeros(len(data), dtype=fields)
#
#     # Loop over each allocation in data and run the lvl1 preset passmanager
#     # using that allocation as the initial layout
#     q_prob = []
#     for count, datum in tqdm(enumerate(data)):
#         # Grab metrics from data
#         allo = datum['allocation'][0]
#         energy = datum['energy']
#         swap = datum['swaps']
#         prob = datum['success_prob']
#         phys = np.where(allo == 1)[1]
#
#         # Convert allocation to the dictionary form qiskit expects for transpilation
#         ind = np.where(allo == 1)
#         allo_dic = {circuit.qubits[i] : ind[1][i] for i in range(n_c)}
#
#         # Create object definitions needed for configuring pass manager
#         layout = Layout(allo_dic)
#         coupling_map = CouplingMap(couplinglist=graph_edges)
#
#         # Create level 1 pass manager with the settings in config
#         # and run the input circuit on it
#         config = PassManagerConfig(initial_layout=layout, coupling_map=coupling_map, routing_method='basic', backend_properties=backend_prop)
#
#         pm = level_1_pass_manager(config)
#         transpile_circ = pm.run(circuit)
#         q_swap = transpile_circ.count_ops()['swap']
#
#         # Make a list and populate with qiskit success probabilities based on
#         # the circuit it transpiled
#
#         g1, g2 = parse_transpile(transpile_circ, n_p) # read gates into dicts
#         swap2cx(g2) # convert swaps to cx
#
#         prob_q = np.zeros(n_p*n_p) # initialize prob matrix
#         prob_q.resize(n_p, n_p)
#
#         g1count = gatecount(g1) # count up gate interactions
#         g2count = gatecount(g2)
#
#         # loop over all possible qbit interactions (1 or 2)
#         # and calculate probability for each one for all gates
#         for i,j in g2count:
#
#             if i == j: # single qubit interaction
#                 prob_q[i,j] = (1-err1[i])**g1count[i]
#
#             else: # two qubit interaction
#                 prob_q[i,j] = (1-err2[i,j])**g2count[(i,j)]
#
#         q_prob.append(np.prod(prob_q))
#
#         # Populate record array
#         array[count] = (allo, energy, prob, swap, q_swap, phys)
#
#     return array, q_prob


# stuff not in use
    # def permute_colours(allo):
    #     '''
    #     Creates % difference from reference lowest energy and returns a list of those
    #     %'s for each allocation and returns this as a list to use for colour plots.
    #     '''
    #
    #     data['energy'].sort(axis=0)
    #     ref_allo = np.where(data['allocation'][0][0] == 1)[1]
    #
    #     perc_list = []
    #     for datum in data:
    #         allo = datum['allocation'][0]
    #         allo_set = np.where(allo == 1)[1]
    #
    #         percentage = np.sum(allo_set == ref_allo) / len(ref_allo)
    #
    #         perc_list.append(percentage)
    #
    #     return perc_list
