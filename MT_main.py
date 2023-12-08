import numpy
import math
from joblib import Parallel, delayed
from MT_ansatz import calculate_energy
from MT_pso import run_parallel

# The following lines are used to solve spam of INFO messages
import logging
loggers = ['qiskit.transpiler.runningpassmanager', 'qiskit.transpiler.passes.basis.basis_translator',
           'qiskit.compiler.transpiler', 'qiskit.compiler.assembler', 'qiskit.providers.basicaer.statevector_simulator']
for logger in loggers:
    logging.getLogger(logger).setLevel(logging.WARNING)


def get_cost(par, ite, shots, imp):
    """
    Return a total number of shots required for evaluating PSO with recalculation for provided arguments.
    This follows formula: (par*shots+imp)*ite*2

    :param par: int, Number of particle
    :param ite: int, Number of iterations
    :param shots: int, Number of shots per particle
    :param imp: int, Number of shots per recalculation at the end of each iteration

    :return: int, Total number of shots used by the algorithm.
    """
    return (par*shots+imp)*ite*2


def get_imp(par, ite, shots, budget):
    """
     Return the maximum number of shots that can be used for recalculation at the end of each iteration.
     This follows formula: (budget/ite - par*shots)/2

    :param par: int, Number of particle
    :param ite: int, Number of iterations
    :param shots: int, Number of shots per particle
    :param budget: int, Maximum number of shots the algorithm can use

    :return: int, Maximum number of shots that can be used for recalculation at the end of each iteration with the
    provided arguments.

    """
    return math.floor((budget/ite - par*shots)/2)


def get_shots(par, ite, imp, budget):
    """
    Return the maximum number of shots that can be used per particle.
    This follows formula: (budget/ite - imp)/par/2

    :param par: int, Number of particle
    :param ite: int, Number of iterations
    :param imp: int, Number of shots per recalculation at the end of each iteration
    :param budget: int, Maximum number of shots the algorithm can use

    :return: int, Maximum number of shots that can be used per particle with the provided arguments.
    """
    return math.floor((budget/ite - imp)/par/2)


def pso_w(x):
    """
    Return an inertia weight that corresponds to the constriction factor with psi=x.

    :param x: int, Corresponding to the psi in constriction factor version of PSO.

    :return: int, Corresponding to the inertia weight for PSO based on the provided x.
    """
    return 2/(x-2+math.sqrt(x*x-4*x))


# This is hell of a function
def get_opt_statevector(results, cores):
    """
    Recalculate the energy with statevector using multiple cores.

    :param results: list, Two lists of tuples containing results from multi_run.
    :param cores: int, Number of cores that will be used for calculations of true energies

    :return: list, Array with recalculated energies based on detailed data from results.
    """
    tmp = []
    for i in range(len(results[1])):
        for j in range(len(results[1][i])):
            flag = True
            for k in range(len(results[1][i][j][1])):
                if results[1][i][j][0] == results[1][i][j][2][k] and flag:
                    tmp.append(results[1][i][j][1][k])
                    flag = False
    tmp2 = Parallel(n_jobs=cores)(delayed(calculate_energy)(tmp[i], 1, True, False, True) for i in range(len(tmp)))
    return tmp2


def save_results(file_name, results, particles, iterations, shots, optimiser, improve=0, simulator=True, noisy=False,
                 details=False, cores=1):
    """
    This function saves the results from a multi-run simulation.  It does not perform the multi-run simulation
     itself. If `details=True`, three files will be created; otherwise, only the first file described below will be
     created. The first file will be named as file_name, name of second one will have _all_opt inserted before first
     comma, name of third one will have _all_opt_detailed inserted before first comma. The content of the first file is:
     true_energy,best_energy,angle1,angle2,angle3,angle4,shots,iterations,particles,optimiser,simulator,noisy,improve
     for each run. The content of second file is best_cost,iteration,run,true_cost for each iteration*run. The content
     of the third file is angle1,angle2,angle3,angle4,value,particle_number,iteration,run for each
     particle*iteration*run

    :param file_name: str, Name of the file to save results to.
    :param results: list, Contains two lists of tuples with results of multi_run. For details refer to multi_run.
    :param particles: int, Number of particles
    :param iterations: int, Number of iterations
    :param shots: int, Number of shots per particle
    :param optimiser: str, Name of optimiser
    :param improve: int, Number of shots per recalculation at the end of each iteration, default value = 0
    :param simulator: bool, Whether a simulator was used, default value = True
    :param noisy: bool,  Whether a noisy simulator was used, default value = False
    :param details: bool, Whether to create additional files with more details about each run. If True, then two
    additional files will be created. If False, then only the first file will be created. default value = False
    :param cores: int, Number of cores used for parallelization, only have effect if details=True
    """
    # Open the file for writing
    f = open(file_name, "w")

    # Write the headers for the data
    f.write("true_energy,best_energy,angle1,angle2,angle3,angle4,"
            "shots,iterations,particles,optimiser,simulator,noisy,improve\n")

    # Initialize an empty string to store the data
    item_str = ""

    # Iterate through the results
    correct_result = []
    for item in results[0]:
        # Extract the angles from the current item
        angles = []
        for element in item:
            if isinstance(element, (list, tuple, numpy.ndarray)):
                angles = element
                for element2 in element:
                    item_str += str(element2) + ","
            else:
                item_str += str(element) + ","

        # Calculate the true energy for the current set of angles
        true_value = calculate_energy(angles, 1, True, False, True)
        correct_result.append(true_value)

        # Write the data for the current item to the file
        f.write(str(true_value) + ",")
        f.write(item_str + str(shots) + "," + str(iterations) + "," + str(particles) + "," + optimiser +
                "," + str(simulator) + "," + str(noisy) + "," + str(improve) + "\n")
        item_str = ""

    # Close the file
    f.close()

    # Check if user want to save optional data
    if details:

        # Create the filenames for the optional results files
        file_all_opt = file_name.split(".")[0] + "_all_opt." + file_name.split(".")[1]
        file_all_opt_detail = file_name.split(".")[0] + "_all_opt_detailed." + file_name.split(".")[1]

        # Get the statevector for results, this is can use multiple cores if provided
        opt = get_opt_statevector(results, cores)

        # Open the files for writing
        f = open(file_all_opt, "w")
        f2 = open(file_all_opt_detail, "w")

        # Write the headers
        f.write("best_cost,iteration,run,true_cost\n")
        f2.write("angle1,angle2,angle3,angle4,value,particle_number,iteration,run\n")

        for i in range(len(results[1])):
            for j in range(len(results[1][i])):
                # Write the data for the current result to the first file
                f.write(str(results[1][i][j][0]) + "," + str(j + 1) + "," + str(i + 1) + "," +
                        str(opt[i*len(results[1][i])+j]) + "\n")

                # Write the detailed information for the current result to the second file
                for k in range(len(results[1][i][j][1])):
                    for m in range(len(results[1][i][j][1][k])):
                        f2.write(str(results[1][i][j][1][k][m]) + ",")
                    f2.write(str(results[1][i][j][2][k]) + ",")
                    f2.write(str(k+1) + "," + str(j + 1) + "," + str(i + 1) + "\n")

        # Close the files
        f.close()
        f2.close()


def multi_run(runs, particles, iterations, shots, optimiser, improve=0, simulator=True, noisy=False, statevector=False,
              cores=1):
    """
    Run the VQE algorithm a specified number of times, using the given parameters for each run.

    :param runs: int, Number of runs
    :param particles: int, Number of particles
    :param iterations: int, Number of iterations
    :param shots: int, Number of shots per particle
    :param optimiser: (str, dict), Supported strings are Global, Local, VonNeumann, Pyramid, Random.
    Dictionary contains options for the optimiser, minimally c1, c2, w.
    :param improve: int, Number of shots per recalculation at the end of each iteration, default value = 0
    :param simulator: bool, If True then simulator will be used, else ibm_kyoto will be used, default True
    :param noisy: bool, If True then FakeLagosV2 will be used, else qasm_simulator will be used, default False
    :param statevector: Boolean if True then statevector will be used instead of qasm simulator
    :param cores: int, The number of cores to use for parallelism.

    :return: (list, list), A tuple of two arrays. The first array contains the best_cost and best_pos for each VQE run.
     The second array has the following form: array[run][iteration] = (minimal energy of this iteration, array with
     position of particles, array with energy of particles)
    """
    result = []
    intermediate = []
    for _ in range(runs):
        best_cost, best_pos, data = run_parallel(particles, iterations, shots, optimiser, improve, simulator, noisy,
                                                 statevector, cores)
        result.append((best_cost, best_pos))
        intermediate.append(data)
    return result, intermediate

# Uncomment following lines to run the example
# x=4.3; w = pso_w(x); c1 = c2 = x / 2 * w
# tmp = multi_run(5, 14, 20, 1785, ("Global", {'c1': c1, 'c2': c2, 'w': w}), improve=25010, noisy=True, cores=5)
# save_results("test.csv", tmp, 14, 20, 1785, "PSO", improve=25010, noisy=True, details=True, cores=6)
