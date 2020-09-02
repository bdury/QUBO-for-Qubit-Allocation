# QUBO-for-Qubit-Allocation
Uses the QUBO formalism to generate a cost function whose solutions are allocations
of logical qubits from a quantum circuit to the physical qubits of a
hardware graph. Solutions from the cost function are obtained using simulated
annealing, as implemented in D-wave's `neal`. An investigation of how
effective this method is at generating initial allocations is available here (https://arxiv.org/pdf/2009.00140.pdf).
Code was written by Bryan Dury ([@bdury](https://github.com/bdury)), but the paper is a product of both [@bdury](https://github.com/bdury)
and [@glassnotes](https://github.com/glassnotes).

## Installation
The specifications of an Anaconda environment are available in the `environment.yml` file.
To reproduce the environment install Anaconda and run the following command:

```
  conda env create --file environment.yml
```

Note that the project specifies Qiskit v.0.20.0, but uses an unstable build of
qiskit-terra for generating the data used in the paper. If you don't want to try
and replicate our results, just follow the above instructions. Otherwise, if the current
available version of qiskit-terra is > v.0.15.1, use the most recent
version available. If this is not the case please follow the guide [here](https://qiskit.org/documentation/contributing_to_qiskit.html#building-from-source)
to install qiskit-terra from source.

Also, there are some functions under `benchmark.py` that use `pytket` which is
only available on macOS or Linux distributions, and I developed this mainly on
Windows, so it is not included in the environment file. If you want to use those
functions please install `pytket` using pip.

## Working with the Code
There is a Jupyter Notebook with a minimal working example that leads you through
how to generate QUBO initial allocations under the folder `examples`. The source
code itself contains many other useful functions to work with the data that is
produced from each simulated annealing run for each set of circuits - please have
a look if you are interested in working with the data.
