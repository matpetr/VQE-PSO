# VQE-PSO

**This repository provides the necessary files to reproduce experiments with Variational Quantum Eigensolver (VQE) utilizing Particle Swarm Optimization (PSO).**

## About the files

* `MT_ansatz.py` contains the necessary functions to create a quantum circuit and calculate the energy.
* `MT_pso.py` contains the necessary functions to run optimization using VQE with PSO.
* `MT_main.py` contains additional functions that can be used to calculate precise parameters, perform multiple runs of optimization, and save the results to a file.
* `MT_vqe.py` is an almost independent package that contains everything necessary to run VQE with different optimizers from qiskit, with the only missing feature in the  package being the recalculation of the true energy when saving data.
* `requirements.txt` contains the versions of the libraries that were used to obtain the original data.

## Examples

**Example using PSO:**

```python
from MT_main import pso_w, multi_run, save_results

# Set PSO parameters
x = 4.3
w = pso_w(x)
c1 = c2 = x / 2 * w
particles = 14
iterations = 20

# Set shots
shots = 1785
recalculation = 25010

# Perform 10 runs of optimization with VQE using PSO on a noisy device.
# Save the results to output.csv and two additional files with more detailed data
tmp = multi_run(10, particles, iterations, shots, ("Global", {'c1': c1, 'c2': c2, 'w': w}), improve=recalculation, noisy=True, cores=4)
save_results("output.csv", tmp, particles, iterations, shots, "PSO", improve=recalculation, noisy=True, details=True, cores=4)
```

**Example using SPSA:**

```python
from MT_vqe import save_results, multi_run, get_backend
from qiskit.algorithms.optimizers import SPSA

shots = 3984

# Use SPSA optimizer and save the results to output.csv
save_results("output.csv", multi_run(10, get_backend(False), SPSA(maxiter=100), shots), 1024, shots, "SPSA")
```

## Notes

* It is no longer possible to use `ibm_lagos`. If you choose to use a quantum computer, `ibm_kyoto` will be used instead.

* If you are able to reserve a quantum computer and want to run this code, it is recommended to put the `service` definition in the `MT_ansatz` file outside of the function. Each service definition can take a few seconds, which can significantly slow down the evaluation.
