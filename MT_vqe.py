from qiskit import QuantumCircuit, BasicAer
from qiskit.utils import QuantumInstance
from qiskit.algorithms import VQE
from qiskit.circuit import Parameter
from qiskit.opflow import I, X, Z
from qiskit.providers.fake_provider import FakeLagosV2
from qiskit.algorithms.optimizers import SPSA, NFT, GradientDescent, ADAM
from MT_ansatz import calculate_energy
import numpy

# Following import is used to prevent spam of DeprecationWarning
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def hamiltonian():
    """
    Returns the Hamiltonian of the Hydrogen molecule. The Hamiltonian is expressed in OperatorBase necessary for VQE.

    :return: OperatorBase, Hydrogen Hamiltonian
    """
    return (-1.0501604336972703 * I ^ I) + \
           (0.4042146605457886 * I ^ Z) + \
           (0.4042146605457886 * Z ^ I) + \
           (0.011346884397300416 * Z ^ Z) + \
           (0.18037524720542222 * X ^ X)


def circuit():
    """
    Create quantum 2-qubit circuit with 4 parameters, which define space expressive enough to contain ground state of
    Hydrogen

    :return: QuantumCircuit, circuit with 2 quantum registers and 4 parameters
    """
    qc = QuantumCircuit(2, 0)
    a = Parameter('a')
    b = Parameter('b')
    c = Parameter('c')
    d = Parameter('d')
    qc.ry(a, 0)
    qc.ry(b, 1)
    qc.cx(0, 1)
    qc.ry(c, 0)
    qc.ry(d, 1)
    return qc


def get_backend(noisy):
    """ Create backend, if noisy=True returns FakeLagosV2, else return qasm_simulator from BasicAer
    Create backend, if noisy returns FakeLagosV2, else return qasm_simulator from BasicAer

    :param noisy: bool, Specify if a noisy simulator should be used

    :return: qiskit.backend instance, Backend that can be used for vqe function.
    """
    if noisy:
        return FakeLagosV2()
    return BasicAer.get_backend('qasm_simulator')


def vqe(backend, opt, shots):
    """
     Run one instance of vqe optimization using provided setting

    :param backend: qiskit.backend instance, Backend which will be used to simulate circuit
    :param opt: qiskit.algorithms.optimizers instance, Optimizer which will be used in vqe
    :param shots: int, Number of shots per iteration
    :return: tuple, Results of optimization (energy, list of counts, list of values, list of parameters). Lists contain
     values used in each iteration of vqe optimization.
    """
    qi = QuantumInstance(backend=backend, shots=shots)
    ansatz = circuit()

    counts = []
    values = []
    parameter = []

    def store_intermediate_result(eval_count, parameters, mean, std):
        counts.append(eval_count)
        values.append(mean)
        parameter.append(parameters)

    vqe_alg = VQE(ansatz, optimizer=opt, quantum_instance=qi, callback=store_intermediate_result)
    result = vqe_alg.compute_minimum_eigenvalue(operator=hamiltonian())
    return result.eigenvalue.real, counts, values, parameter


def save_results(file_name, results, shots, noisy, opt):
    """
    Save result provided by multi_run in file_name, each line contains true_energy,best_energy,angle1,angle2,angle3,
    angle4,eval,shots,noisy,opt.

    :param file_name: str, File with corresponding name will be created and used to save results
    :param results: list,  List of tuples with results of multi_run
    :param shots: int, Number of shots used per iteration.
    :param noisy: bool, Specify if a noisy simulator should be used
    :param opt: str, Name of optimizer.
    """
    f = open(file_name, "w")
    f.write("true_energy,best_energy,angle1,angle2,angle3,angle4,eval,shots,noisy,opt\n")
    for item in results:
        item_str = ""
        for element in item:
            if type(element) in (list, tuple, numpy.ndarray):
                for element2 in element:
                    item_str += str(element2) + ","
            else:
                item_str += str(element) + ","
        f.write(item_str + str(shots) + "," + str(noisy) + "," + opt + "\n")
    f.close()


def multi_run(runs, backend, opt, shots):
    """
    Run vqe runs number of times, each time using provided arguments.

    :param runs: int, Number of times vqe will be run.
    :param backend: qiskit.backend instance, Backend which will be used to simulate circuit
    :param opt: qiskit.algorithms.optimizers instance, Optimizer which will be used in vqe
    :param shots: int, Number of shots used per iteration

    :return: list, Array of tuples with results of vqe function.
    """
    result = []
    for i in range(runs):
        tmp = vqe(backend, opt, shots)
        for j in range(len(tmp[2]) - 1, -1, -1):
            if tmp[2][-1] == tmp[0]:
                break
            print(j, tmp[2][j], tmp[0])
        true_value = calculate_energy(tmp[3][-1], 1, True, False, True)
        result.append((true_value, tmp[0], tmp[3][-1], tmp[1][-1]))
    return result


# Uncomment following line to run example
# save_results("test.csv", multi_run(10, get_backend(False), SPSA(maxiter=100), 1024), 1024, False, "SPSA")
