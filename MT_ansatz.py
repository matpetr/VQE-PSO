from qiskit import QuantumCircuit, transpile, BasicAer
from qiskit.providers.fake_provider import FakeLagosV2
from qiskit_ibm_runtime import QiskitRuntimeService
import math


def second_power(c):
    """
    Calculate the second power of complex number, return only the real part.

    :param c: complex, Complex number

    :return: int, the real part of the squared complex number
    """
    return (c * c.conjugate()).real


def complete_circuit(angles, statevector=False):
    """
    Create a 4-qubit circuit with 4 parameters, which is expressive enough to cover the ground state of the Hydrogen
    Hamiltonian. For easier simulation in the case of the statevector are created two circuits with 2 qubits each.

    :param angles: list, List of angles of the circuit
    :param statevector: bool, Indicate whether the backend is a statevector simulator

    :return: list, Array of QuantumCircuits, containing either a single 4-qubit circuit or two 2-qubit circuits in case
    of the statevector
    """
    if statevector:
        qc = [QuantumCircuit(2, 2), QuantumCircuit(2, 2)]
        qc[0].ry(angles[0], 0)
        qc[0].ry(angles[1], 1)
        qc[0].cx(0, 1)
        qc[0].ry(angles[2], 0)
        qc[0].ry(angles[3], 1)
        qc[1].ry(angles[0], 0)
        qc[1].ry(angles[1], 1)
        qc[1].cx(0, 1)
        qc[1].ry(angles[2], 0)
        qc[1].ry(angles[3], 1)
        qc[1].ry(math.pi / 2, 0)
        qc[1].ry(math.pi / 2, 1)
        return qc

    qc = QuantumCircuit(4, 4)
    qc.ry(angles[0], 0)
    qc.ry(angles[1], 1)
    qc.cx(0, 1)
    qc.ry(angles[2], 0)
    qc.ry(angles[3], 1)
    qc.ry(angles[0], 2)
    qc.ry(angles[1], 3)
    qc.cx(2, 3)
    qc.ry(angles[2], 2)
    qc.ry(angles[3], 3)
    qc.ry(math.pi / 2, 2)
    qc.ry(math.pi / 2, 3)
    qc.measure([0, 1, 2, 3], [0, 1, 2, 3])
    return [qc]


def tensor_diagonal(diagonal1, diagonal2):
    """
    Compute the tensor product of two vectors. This is possible because values outside the main diagonal are zero.

    :param diagonal1: list, Array representing the diagonal of the first matrix
    :param diagonal2: list, Array representing the diagonal of the second matrix

    :return: list, Array representing the tensor product of the provided vectors
    """
    diagonal = []
    for n1 in diagonal1:
        for n2 in diagonal2:
            diagonal.append(n1 * n2)
    return diagonal


def evaluate_count(counts, diagonal, shots, statevector):
    """
    Calculate the value based on the measurement result and matrix diagonal.

    :param counts: dict or list, Measurement result, either a list in the statevector case or a dictionary
    :param diagonal: list, represents the diagonal of the matrix which specifies which values should be used to
    calculate the final result
    :param shots: int, Number of shots used during the measurement
    :param statevector: bool, indicates whether the evaluation method for statevector should be used

    :return: int, calculated value based on the measurement result and matrix diagonal
    """
    if statevector:
        result = 0
        for i in range(len(counts)):
            result += second_power(counts[i]) * diagonal[i]
        return result
    result = 0
    possible = ["00", "01", "10", "11"]
    for i in range(len(possible)):
        if possible[i] in counts:
            result += counts[possible[i]] * diagonal[i] / shots
    return result


def split_counts(counts):
    """
    Splits the counts dictionary into two dictionaries, one for the first two qubits and one for the last two qubits.

    :param counts: dict, Dictionary containing the measurement results. Keys represent the measurement outcomes and
    values are the counts for each outcome.

    :return: list, Array containing two dictionaries, one for the first two qubits and one for the last two qubits.
    """
    counts1 = {"00": 0,
               "01": 0,
               "10": 0,
               "11": 0}
    counts2 = {"00": 0,
               "01": 0,
               "10": 0,
               "11": 0}
    for key in counts:
        counts1[key[2:]] += counts[key]
        counts2[key[:2]] += counts[key]
    return [counts1, counts2]


def create_backend(simulator=True, noisy=False, statevector=False):
    """
    Creates a backend instance that will be used for running the simulation. If a simulator is used, returns an instance
     of the specified simulator. If a quantum computer is used, returns the string ibm_kyoto.

    :param simulator: bool, Specify if a simulator should be used. If True, a simulator is returned. If False, a quantum
    computer is returned. This argument have priority over the noisy argument. Default value = True
    :param noisy: bool, Specify if a noisy simulator should be used. If True, an instance of FakeLagosV2 is returned. If
     False, an instance of qasm_simulator, is returned. Default value = False
    :param statevector: bool, Specify if a statevector simulator should be used. If True, a statevector_simulator is
    returned. This argument have priority over other arguments. Default value = False

    :return: String or qiskit.backend instance, Backend that can be used for calculate_energy function.
    """
    if statevector:
        return BasicAer.get_backend('statevector_simulator')
    if not simulator:
        return 'ibm_kyoto'
    if noisy:
        return FakeLagosV2()
    else:
        return BasicAer.get_backend('qasm_simulator')


def calculate_energy(angles, shots, simulator, noisy, statevector):
    """
    Constructs a circuit using the provided angles and runs it with the specified backend. Calculates the energy based
    on the measurement results.


    :param angles: list, Contains the angles of the circuit
    :param simulator: bool, Specify if a simulator should be used. If True then ibm_kyoto is used instead
    :param noisy: bool, Specify if a noisy simulator should be used
    :param shots: int, Number of shots used for the measurement
    :param statevector: bool, Specify if a statevector_simulator should be used., have priority over simulator and noisy
     arguments

    :return: int, The Energy achieved by the measurement of the circuit calculated by the Hydrogen Hamiltonian
    """

    qc = complete_circuit(angles, statevector)
    backend = create_backend(simulator, noisy, statevector)

    if statevector:
        job = backend.run(transpile(qc, backend))
    elif not simulator:
        service = QiskitRuntimeService()
        inputs = {"shots": shots,
                  "circuits": qc}
        options = {"max_execution_time": 300,
                   "backend": backend}
        job = service.run(program_id="circuit-runner", options=options, inputs=inputs)
    else:
        job = backend.run(transpile(qc, backend), shots=shots)

    if statevector:
        counts = [job.result().get_statevector(experiment=0), job.result().get_statevector(experiment=1)]
    else:
        counts = job.result().get_counts()
        counts = split_counts(counts)

    I = [1, 1]
    Z = [1, -1]

    II = tensor_diagonal(I, I)
    IZ = tensor_diagonal(I, Z)
    ZI = tensor_diagonal(Z, I)
    ZZ = tensor_diagonal(Z, Z)

    Id = evaluate_count(counts[0], II, shots, statevector)
    Z0 = evaluate_count(counts[0], IZ, shots, statevector)
    Z1 = evaluate_count(counts[0], ZI, shots, statevector)
    Z1Z0 = evaluate_count(counts[0], ZZ, shots, statevector)
    X1X0 = evaluate_count(counts[1], ZZ, shots, statevector)

    c0 = -1.0501604336972703
    c1 = 0.4042146605457886
    c2 = 0.011346884397300416
    c3 = 0.18037524720542222

    H = c0 * Id + c1 * Z0 + c1 * Z1 + c2 * Z1Z0 + c3 * X1X0
    return H
