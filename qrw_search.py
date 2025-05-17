# import libraries
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit.library import UnitaryGate
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
from numpy import pi, array
import numpy as np
from helper import print_table_from_dict

# Define the variables
n = 3 # dimension of graph
n_d = int(np.ceil(np.log2(n))) # dimension of coin

# declare the shift operator
ed = np.array([[0, 0, 0],
               [0, 0, 1],
               [0, 1, 0],
               [1, 0, 0]])

S = np.zeros((2**(n+n_d), 2**(n+n_d)))
for d in range(2**n_d):
    for x in range(2**n):
        vd = np.zeros((1, 2**n_d))
        vd[0, d] = 1
        vx = np.zeros((1, 2**n), dtype=np.int64)
        vx[0, x] = 1
        ved = np.zeros((1, 2**n), dtype=np.int64)
        ed_decimal = int("".join(map(str, ed[d])), 2)
        ved[0, np.bitwise_xor(ed_decimal, x)] = 1
        col  = np.kron(ved, vd).reshape(2**(n+n_d), 1)
        row  = np.kron(vx, vd).reshape(1, 2**(n+n_d))
        S += np.dot(col, row)
np.set_printoptions(threshold=np.inf)
S_gate = UnitaryGate(S, label='shift\noperator')

# Declare the coin operator
C = UnitaryGate(2/2**n_d*np.ones((2**n_d, 2**n_d)) - np.eye(2**n_d), label='coin')
marked_state = 3
marked_state_mat = np.zeros((2**n, 2**n))
marked_state_mat[marked_state, marked_state] = 1
C1 = np.kron(np.eye(2**n), C.to_matrix()) - np.kron(marked_state_mat, (C-np.eye(2**n_d)))
C1_gate = UnitaryGate(C1,  label='marked\ncoin')

# Create the quantum circuit 
qc_search  = QuantumCircuit()
steps = 5
# Add quantum registers to the circuit
shift = QuantumRegister(n, 's') # qubits associated with the graph nodes
coin = QuantumRegister(n_d, 'c') # qubit associated with the coin
qc_search.add_register(coin)
qc_search.add_register(shift)
# Create equal superposition of coin and walker
[qc_search.h(coin[i]) for i in range(n_d)]
qc_search.barrier()
for _ in range(steps):
    qc_search.append(C1_gate, range(qc_search.num_qubits))
    # Apply the walk
    qc_search.append(S_gate, range(qc_search.num_qubits))
    qc_search.barrier()
# Measure the state of the walker
qc_search.measure_all()
# Display the quantum circuit
qc_search.draw('mpl',  filename=f"qrw_search_{steps}.png")

# Run the simulation
sim_ideal = AerSimulator()
tqc_search = transpile(qc_search, sim_ideal)
result_ideal = sim_ideal.run(tqc_search).result()
search_counts =  result_ideal.get_counts()
# Display the results
plot_histogram(search_counts, filename=f"qrw_search_{steps}")
# Print the simulation data
print_table_from_dict(search_counts, n, n_d)
# Print the marked element
max_key = max(search_counts, key=search_counts.get)
print("Marked element:", int(max_key[0:3], 2))

