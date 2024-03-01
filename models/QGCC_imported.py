import pennylane as qml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from qiskit import QuantumCircuit

from qiskit_qml_conversion.personalized_gates import RXX, RYY, RZZ

"""
This script contains the class PQWGAN_CC for a Quantum Generator and Classical Discriminator
CC = Classical Critic, with the Quantum Generator imported from a qiskit file and obtained with 
an evolutionary algorithm.
"""

#################################################
# Quantum Generator and Classical Discriminator #
#         WITH IMPORTED QUANTUM CIRCUIT         #
#################################################


class PQWGAN_CC_imported():
    def __init__(self, image_size, channels, n_ancillas, qasm_file_path):
        """
        Initializes the PQWGAN_CC_imported class with a quantum generator (imported from a .qasm
        file) and classical discriminator. Translates the file from qiskit to pennylane and sets
        up the class for training of the generator (quantum) and discriminator (classical).
        Args:
            image_size (int): The size of the input image (which is assumed to be squared)
            n_ancillas (int): The number of ancilla qubits in the quantum generator.
            patch_shape (tuple): The shape of the patch in the output image.
            qasm_file_path (str): The file path to the QASM file.
        """
        # Initialize the PQWGAN_CC class: create the discriminator and generator entities.
        self.image_shape = (channels, image_size, image_size)
        self.critic = self.ClassicalCritic(self.image_shape)
        self.qasm_file_path = qasm_file_path
        self.generator = self.QuantumGeneratorImported(self.image_shape, self.qasm_file_path, n_ancillas)

    class ClassicalCritic(nn.Module):  # class torch.nn.Module because classical
        def __init__(self, image_shape):
            """
            Initializes the ClassicalCritic class with pytorch class Module.
            :param image_shape: tuple. The shape of the input image.
           """
            super().__init__()
            self.image_shape = image_shape

            self.fc1 = nn.Linear(int(np.prod(self.image_shape)), 512)
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, 1)

        def forward(self, x):
            """
            Performs forward pass through the ClassicalCritic (a classical neural network).
            :param x: torch.Tensor. The input tensor (image to be evaluated by critic).

            :returns: torch.Tensor. The prediction of the critic.
            """
            x = x.view(x.shape[0], -1)  # Flatten input image
            x = F.leaky_relu(self.fc1(x), 0.2)  # Apply leaky ReLU activation to the 1st fully connected layer output
            x = F.leaky_relu(self.fc2(x), 0.2)  # Apply leaky ReLU activation to the 2nd fully connected layer output
            return self.fc3(x)  # Return the output of the 3rd fully connected layer



    class QuantumGeneratorImported(nn.Module):
        def __init__(self, image_shape, qasm_file_path, n_ancillas):
            """Initialize the QuantumGenerator class.

            Args:
                image_shape (tuple): The shape of the output image. A tuple (channels, height, width).
                qasm_file_path (str): The path to the QASM file that contains the quantum circuit.
                n_ancillas (int): The number of ancillary qubits included in the quantum circuit.
            Returns:
                None
            """
            super().__init__()
            self.image_shape = image_shape
            self.qasm_file_path = qasm_file_path

            # Import and convert the circuit
            self.qiskit_circuit, self.n_qubits, initial_params = self.importing_circuit()
            self.q_device = qml.device("default.qubit", wires=self.n_qubits)
            self.n_ancillas = n_ancillas

            # self.params = nn.ParameterList(
            #     [nn.Parameter(torch.tensor(value, dtype=torch.float32, requires_grad=True))
            #      for value in initial_params])

            self.params = nn.ParameterList(
                [nn.Parameter(torch.tensor(initial_params, dtype=torch.float32, requires_grad=True))
                 for _ in range(4)]
            )

            self.qnode = qml.QNode(func=self.pennylane_circuit,  # defined below
                                   device=self.q_device,  # the pennylane device initialized above
                                   interface="torch")  # The interface for classical backpropagation


        def importing_circuit(self):
            """
            Imports a circuit from a QASM file and preprocesses it for use with Pennylane.

            Returns:
                QuantumCircuit: The imported Qiskit quantum circuit.
                int: The number of qubits in the circuit.
                list: A list of initial parameters for the quantum gates present in the circuit.
            """
            with open(self.qasm_file_path, 'r') as file:
                qasm_str = file.read()
            qiskit_circuit = QuantumCircuit.from_qasm_str(qasm_str)
            # Delete the encoding layer  # Note: assumption that there is one
            del qiskit_circuit.data[0:qiskit_circuit.num_qubits]

            n_qubits = qiskit_circuit.num_qubits

            initial_params = []
            for instr, _, _ in qiskit_circuit.data:
                if instr.name.lower() in ["rx", "ry", "rz", "rxx", "ryy", "rzz"]:
                    initial_params.extend([param for param in instr.params])

            return qiskit_circuit, n_qubits, initial_params


        def pennylane_circuit(self, latent_vector, params):
            """
            Converts and executes Qiskit circuit in PennyLane. Maps Qiskit gates to PennyLane
            equivalents, applies them with given parameters, and returns the probabilities of the
            computational basis states, of length 2^n_tot_qubits.

            Args:
                latent_vector (torch.Tensor): Input latent vector.
                params (list): Parameters for quantum gates.

            Returns:
                torch.Tensor: Probabilities of quantum states (computational basis states vector).
            """
            # Encode the latent vector
            for i, angle in enumerate(latent_vector):
                qml.RY(angle, wires=i)

            # Map Qiskit gates to PennyLane gates
            for layer in range(4):
                param_idx = 0  # Initialize parameter index
                for instr, qubits, _ in self.qiskit_circuit.data:  # Instructions, qubits, empty
                    name = instr.name.lower()  # gate names all lower case
                    wires = [q._index for q in qubits]  # wires for each single and double gate

                    if name in ["rx", "ry", "rz"]:
                        # print(f'rx,ry,rz. Found gate {name} with param {instr.params} on qubit {wires}')
                        getattr(qml, name.upper())(params[layer][param_idx], wires=wires)
                        param_idx += 1
                    elif name == "rxx":
                        # print(f'RXX. Found gate {name} with param {instr.params} on qubit '
                        #       f'{wires[0]},{wires[1]}')
                        RXX(params[layer][param_idx], wires=[wires[0], wires[1]])
                        param_idx += 1
                    elif name == "ryy":
                        # print(f'RYY. Found gate {name} with param {instr.params} on qubit '
                        #       f'{wires[0]},{wires[1]}')
                        RYY(params[layer][param_idx], wires=[wires[0], wires[1]])
                        param_idx += 1
                    elif name == "rzz":
                        # print(f'RZZ. Found gate {name} with param {instr.params} on qubit '
                        #       f'{wires[0]},{wires[1]}')
                        RZZ(params[layer][param_idx], wires=[wires[0], wires[1]])
                        param_idx += 1
                    elif name == "h":
                        # print(f'h. Found gate {name} on qubit {wires}')
                        qml.Hadamard(wires=wires[0])  # hadamard has no parameters

            return qml.probs(wires=range(self.n_qubits))

        def forward(self, x):
            """
            Perform a forward pass through the QuantumGenerator. Generates one image (tensor)
            given a latent vector.

            :param x: torch.Tensor. Input tensor (latent vector).

            :returns: torch.Tensor. Output tensor (image or image patch).
            """
            # Assuming x is a batch of latent vectors
            images = []
            for latent_vector in x:
                image = self.partial_trace_and_postprocess(latent_vector, self.params)
                images.append(image)
            output_images = torch.stack(images).view(-1, *self.image_shape)
            return output_images.float()



        def partial_trace_and_postprocess(self, latent_vector, weights):
            """
            Performs partial trace and post-processing operations on the quantum circuit outputs.

            :param latent_vector: torch.Tensor. Latent vector, the input to the circuit.
            :param weights: torch.Tensor. Tensor containing the weights of the quantum circuit.

            :return: torch.Tensor. Post-processed patch obtained from the circuit outputs.
            """
            probs = self.qnode(latent_vector, weights)
            # Introduce non-linearity
            probs_given_ancilla_0 = probs[:2**(self.n_qubits - self.n_ancillas)]

            # Normalize the probabilities by their sum
            post_measurement_probs = probs_given_ancilla_0 / torch.sum(probs_given_ancilla_0)

            # Normalise image between [-1, 1] (why not 0,1??)
            post_processed_patch = ((post_measurement_probs / torch.max(post_measurement_probs)) - 0.5) * 2
            total_pixels = self.image_shape[1] * self.image_shape[2]
            post_processed_patch = post_processed_patch[:total_pixels]
            return post_processed_patch  #.to(latent_vector.device)


if __name__ == "__main__":
    gen = PQWGAN_CC_imported(image_size=16, channels=1, n_ancillas=1).generator
    print(qml.draw(gen.qnode)(torch.rand(5), torch.rand(1, 5, 3)))
