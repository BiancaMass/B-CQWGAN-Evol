import numpy as np
from qiskit import QuantumCircuit
import pennylane as qml

from qiskit_qml_conversion.personalized_gates import RXX, RYY, RZZ


class ConversionQiskitPenny:
    """
    Converts a Qiskit quantum circuit into a PennyLane quantum circuit. It supports RX, RY, RZ,
    RXX, RYY, RZZ, and H gates. Additional gates can be added with modifications.

    Attributes:
        n_qubits (int): Number of qubits in the quantum circuit.
        dev (qml.Device): PennyLane device for simulation.
        # TODO might change  to tensor ??
        latent_vector (np.ndarray): Vector to encode into the circuit's initial state.
    """
    def __init__(self, quantum_circuit: QuantumCircuit, latent_vector):
        """
        Initializes the class with a quantum circuit and a latent vector for encoding.

        Parameters:
            quantum_circuit (QuantumCircuit): The Qiskit circuit to convert.
            # TODO: might change to tensor ??
            latent_vector (np.ndarray): The latent vector to encode into the quantum state.
        """
        self.qc = quantum_circuit
        self.n_qubits = quantum_circuit.num_qubits
        self.latent_vector = latent_vector
        # TODO maybe better as an argument to input, so it can be defined outside the class
        self.dev = qml.device("default.qubit", wires=self.n_qubits) # Initialize PennyLane device

    def variational_block(self):
        """"
        Converts the Qiskit circuit to a PennyLane circuit, applying Qiskit gates as their
        PennyLane equivalents, including any custom gates.

        Returns:
            function: A PennyLane qnode function representing the converted circuit.
        """
        # TODO: delete after finishing development
        # # Extract parameters from the Qiskit circuit
        # initial_params = []
        # for instr, _, _ in self.qc.data:
        #     if instr.name.lower() in ["rx", "ry", "rz", "rxx", "ryy", "rzz"]:
        #         initial_params.extend([param for param in instr.params])

        # print(f'Initial parameters: {initial_params}')
        @qml.qnode(self.dev)
        def circuit(latent_vector):
            # Apply initial encoding from the latent vector
            for i, angle in enumerate(latent_vector):
                qml.RY(angle, wires=i)

            # Map Qiskit gates to PennyLane gates
            for instr, qubits, _ in self.qc.data:  # Instructions, qubits, empty
                name = instr.name.lower()  # gate names all lower case
                wires = [q._index for q in qubits]  # wires for each single and double gate

                if name in ["rx", "ry", "rz"]:
                    # print(f'rx,ry,rz. Found gate {name} with param {instr.params} on qubit {wires}')
                    getattr(qml, name.upper())(instr.params[0], wires=wires)
                elif name == "rxx":
                    # print(f'RXX. Found gate {name} with param {instr.params} on qubit '
                    #       f'{wires[0]},{wires[1]}')
                    RXX(instr.params[0], wires=[wires[0], wires[1]])
                elif name == "ryy":
                    # print(f'RYY. Found gate {name} with param {instr.params} on qubit '
                    #       f'{wires[0]},{wires[1]}')
                    RYY(instr.params[0], wires=[wires[0], wires[1]])
                elif name == "rzz":
                    # print(f'RZZ. Found gate {name} with param {instr.params} on qubit '
                    #       f'{wires[0]},{wires[1]}')
                    RZZ(instr.params[0], wires=[wires[0], wires[1]])
                elif name == "h":
                    # print(f'h. Found gate {name} on qubit {wires}')
                    qml.Hadamard(wires=wires[0])  # hadamard has no parameters

            return qml.probs(wires=range(self.n_qubits))

        return circuit

    def get_probability_vector(self):
        """
        Executes the converted PennyLane circuit with the encoded latent vector, returning
        the probability vector for each computational basis state.

        Returns:
            np.ndarray: Probability vector of length 2**n_qubits.
        """
        circuit = self.variational_block()
        probs = circuit(self.latent_vector)
        return probs

