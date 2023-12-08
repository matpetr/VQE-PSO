from MT_ansatz import calculate_energy
import math
import pyswarms as ps
from joblib import Parallel, delayed


def create_optimiser(optimiser, particles):
    """
    Create an optimizer object based on the specified settings.

    :param optimiser: (str, dict), str: The topology to be used for the PSO optimizer. Supported topologies are: Global,
     Local, VonNeumann, Pyramid, Random; dict: Dictionary containing additional options for the PSO optimizer. Necessary
     options are: c1, c2, w
    :param particles: int, The number of particles to use in the PSO optimizer.

    :return: obj, An instance of the pyswarms.single optimizer based on the provided settings.
    """
    optimiser_str = optimiser[0]
    options = optimiser[1]
    center = [math.pi / 2, math.pi / 2, math.pi / 2, math.pi / 2]
    if optimiser_str == "Global":
        return ps.single.GlobalBestPSO(n_particles=particles, dimensions=4, options=options, center=center)
    if optimiser_str == "Local":
        return ps.single.LocalBestPSO(n_particles=particles, dimensions=4, options=options, center=center)
    if optimiser_str == "VonNeumann":
        tp = ps.backend.topology.VonNeumann(static=False)
        return ps.single.GeneralOptimizerPSO(n_particles=particles, dimensions=4, options=options, topology=tp,
                                             center=center)
    if optimiser_str == "Pyramid":
        tp = ps.backend.topology.Pyramid(static=False)
        return ps.single.GeneralOptimizerPSO(n_particles=particles, dimensions=4, options=options, topology=tp,
                                             center=center)
    if optimiser_str == "Random":
        tp = ps.backend.topology.Random(static=False)
        return ps.single.GeneralOptimizerPSO(n_particles=particles, dimensions=4, options=options, topology=tp,
                                             center=center)


def pso_parallel_one_ite(particles, shots, improve, data, simulator, noisy, statevector, cores):
    """
    Run one iteration of the PSO optimization algorithm using parallelism and obtain intermediate results.

    :param particles: list, An array of arrays containing the angles for each particle.
    :param shots: int, The number of shots to use for each particle when evaluating the energy.
    :param improve: int, The number of shots to use for recalculating the energy after each iteration.
    :param simulator: bool, Whether to use the simulator or the ibm_kyoto
    :param noisy: bool, Whether to use the FakeLagosV2 backend or the qasm_simulator backend.
    :param data: list, A list in which intermediate results will be stored.
    :param statevector: bool, Whether to use the statevector simulator or the qasm_simulator.
    :param cores: int, The number of cores to use for parallelism.

    :return: list, An array containing the best cost for each particle.
    """
    result = Parallel(n_jobs=cores)(delayed(calculate_energy)(particles[i], shots, simulator, noisy, statevector)
                                    for i in range(len(particles)))
    if improve and not statevector:
        minimum = min(result)
        index = result.index(minimum)
        improved_value = calculate_energy(particles[index], improve, simulator, noisy, statevector)
        inc = 0
        while minimum + inc == 0 or (improved_value + inc) / (minimum + inc) <= 0:
            inc += 1
        rate = (improved_value + inc) / (minimum + inc)
        result = [rate * (j + inc) - inc for j in result]

    data.append((min(result), particles, result))
    return result


def run_parallel(particles, iterations, shots, optimiser, improve=0, simulator=True, noisy=False, statevector=False,
                 cores=1):
    """
    Run a single instance of optimization with VQE utilizing PSO which is using parallelism on particles.

    :param particles: int, The number of particles to use in the PSO optimizer.
    :param iterations: int, The number of iterations for the optimization process.
    :param shots: int, The number of shots to use for each particle when evaluating the energy.
    :param optimiser: (str, dict), See the `create_pso_optimizer` function for details.
    :param improve: int, The number of shots to use for recalculating the energy after each iteration.
    :param simulator: bool, Whether to use the simulator or the ibm_kyoto
    :param noisy: bool, Whether to use the FakeLagosV2 backend or the qasm_simulator backend.
    :param statevector: bool, Whether to use the statevector_simulator or the qasm_simulator.
    :param cores: int, The number of cores to use for parallelism.

    :return: tuple, A tuple containing the following:
        best_cost: The lowest energy found during the optimization process.
        best_pos: The angles that archived the lowest energy.
        data: A dictionary containing detailed information about the optimization process.
    """
    opt = create_optimiser(optimiser, particles)
    data = []
    kwargs = {"shots": shots, "data": data, "improve": improve, "simulator": simulator, "noisy": noisy,
              "statevector": statevector, "cores": cores}
    best_cost, best_pos = opt.optimize(objective_func=pso_parallel_one_ite, iters=iterations, verbose=False, **kwargs)
    return best_cost, best_pos, data
