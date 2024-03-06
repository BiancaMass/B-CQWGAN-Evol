"""
Issue with conflicting libraries and library versions (of qiskit). Temporarily gave up.
"""


# import numpy as np
#
# from qiskit import QuantumCircuit
# from qiskit import Aer, transpile
# from qiskit import transpile
# from qiskit import QuantumCircuit
#
#
#
# # def compare_probs_vectors(qiskit_circuit, pennylane_circuit):
# #     # Assuming qiskit_circuit is already defined and filled with gates
# #     n_qubits = qiskit_circuit.num_qubits
# #
# #     circ = QuantumCircuit(2)
# #     circ.h(0)
# #     circ.cx(0, 1)
# #     circ.save_statevector()
# #     # Measure all qubits
# #     qiskit_circuit.measure_all()
# #     # Get the Aer simulator backend for a quantum circuit
# #     simulator = Aer.get_backend('aer_simulator')
# #     circ = transpile(circ, simulator)
# #     # Execute the transpiled circuit on the simulator
# #     job = sim.run(circ, shots=1000)
# #     # Wait for the job to complete and fetch the results
# #     # Get the counts of the qubit measurement outcomes
# #     result = job.result()  # Retrieves the result of the execution
# #     statevector = result.get_statevector(circ)
# #
# #     return statevector
#
#
# # qasm_file_path = '/Users/bmassacci/main_folder/maastricht/academics/quantum_thesis/scripts/PQWGAN/input/final_best_ciruit_0.qasm'
# #
# # # Read the QASM file
# # with open(qasm_file_path, 'r') as file:
# #     qasm_str = file.read()
# #
# # # Create a QuantumCircuit from QASM string
# # qiskit_circuit = QuantumCircuit.from_qasm_str(qasm_str)
# #
# # probs = compare_probs_vectors(qiskit_circuit, 2)
# #
# # print(probs)
#
#
# circ = QuantumCircuit(2)
# circ.h(0)
# circ.cx(0, 1)
# # Get the Aer simulator backend for a quantum circuit
# simulator = Aer.get_backend('aer_simulator')
# circ.save_statevector()
# circ = transpile(circ, simulator)
# # Execute the transpiled circuit on the simulator
# job = simulator.run(circ, shots=1000)
# # Wait for the job to complete and fetch the results
# # Get the counts of the qubit measurement outcomes
# result = job.result()  # Retrieves the result of the execution
# statevector = result.get_statevector(circ)