import numpy as np
from qiskit import QuantumCircuit
import pennylane as qml
import torch
import torch.nn as nn

from qiskit_qml_conversion.personalized_gates import RXX, RYY, RZZ


# qasm_file_path = "/Users/bmassacci/main_folder/maastricht/academics/quantum_thesis/scripts/B-WGAN-Evol/input/final_best_ciruit_U_CNOT.qasm"
#
# with open(qasm_file_path, 'r') as file:
#     qasm_str = file.read()
# qiskit_circuit = QuantumCircuit.from_qasm_str(qasm_str)
# # Delete the encoding layer  # Note: assumption that there is one
# del qiskit_circuit.data[0:qiskit_circuit.num_qubits]

qiskit_circuit = QuantumCircuit(2)

# Apply gates
qiskit_circuit.rx(0.5, 0)
qiskit_circuit.ry(0.8, 1)
qiskit_circuit.rz(0.3, 0)
qiskit_circuit.rxx(0.4, 0, 1)
qiskit_circuit.ryy(0.6, 0, 1)
qiskit_circuit.u(0.1, 0.2, 0.3, 0)

# Add Hadamard gate to qubit 0
qiskit_circuit.h(0)

# Add CNOT gate with control qubit 0 and target qubit 1
qiskit_circuit.cx(0, 1)

# Draw the circuit
print(qiskit_circuit.draw())


n_qubits = qiskit_circuit.num_qubits

initial_params = []
for instr, _, _ in qiskit_circuit.data:
    if instr.name.lower() in ["rx", "ry", "rz", "rxx", "ryy", "rzz"]:
        initial_params.append(instr.params[0])
    elif instr.name.lower() in ["u"]:
        initial_params.append(instr.params)


params = nn.ParameterList([
            nn.Parameter(torch.tensor(param, dtype=torch.float32, requires_grad=True))
            if isinstance(param, list) else
            nn.Parameter(torch.tensor([param], dtype=torch.float32, requires_grad=True))
            for param in initial_params
        ])

print(params)

param_idx = 0  # Initialize parameter index
for instr, qubits, _ in qiskit_circuit.data:  # Instructions, qubits, empty
    name = instr.name.lower()  # gate names all lower case
    wires = [q._index for q in qubits]  # wires for each single and double gate

    if name in ["rx", "ry", "rz"]:
        print(f'rx,ry,rz. Found gate {name} with param {instr.params} on qubit {wires}')
        getattr(qml, name.upper())(params[param_idx], wires=wires)
        param_idx += 1
    elif name == "rxx":
        print(f'RXX. Found gate {name} with param {instr.params} on qubit '
              f'{wires[0]},{wires[1]}')
        RXX(params[param_idx], wires=[wires[0], wires[1]])
        param_idx += 1
    elif name == "ryy":
        print(f'RYY. Found gate {name} with param {instr.params} on qubit '
              f'{wires[0]},{wires[1]}')
        RYY(params[param_idx], wires=[wires[0], wires[1]])
        param_idx += 1
    elif name == "rzz":
        print(f'RZZ. Found gate {name} with param {instr.params} on qubit '
              f'{wires[0]},{wires[1]}')
        RZZ(params[param_idx], wires=[wires[0], wires[1]])
        param_idx += 1
    elif name == "u":  # todo: check this works
        qml.Rot(params[param_idx][0],
                params[param_idx][1],
                params[param_idx][2],
                wires=wires)
        param_idx += 1
    elif name == "cx":
        qml.CNOT(wires=[wires[0], wires[1]])
    elif name == "h":
        # print(f'h. Found gate {name} on qubit {wires}')
        qml.Hadamard(wires=wires[0])  # hadamard has no parameters