from vqengine import *
import sys

N = int(sys.argv[1])
r = 1.5
Name = f"LiH_{N}"
dt = 0.1
total_t = 10.0

# Define geometry e.g. N=2 Li-H, N=3 H-Li-H, N=4 H-Li-H-Li
geometry = [("Li" if n % 2 == 0 else "H", (0., 0., n*r)) for n in range(N)]

# Initialize Molecule
mol = Molecule(geometry, freeze=2)
H, hf = mol.get_hamiltonian(transform='jw')

# Generate Pools
qubit_pool, qubit_v_pool = generate_qubit_pool(mol.n_qubits, mol.n_electrons)

# Run
qubit_E, qubit_thetas, qubit_gates, qubit_gates_t, qubit_L2_t, qubit_variance_t = driver.avqite(
    qubit_pool, qubit_v_pool, dt=dt, grad_tol=5e-4, t=total_t
)

# Post-processing
qubit_errors = [abs(e - mol.fci_energy) for e in qubit_E]
t_vals = [dt * n for n in range(len(qubit_E))]
qubit_L2, qubit_t_L2 = get_t_L2(qubit_L2_t)
qubit_variance, _ = get_t_variance(qubit_variance_t)
qubit_t_L2 = [m*dt for m in qubit_t_L2]
qubit_ncnot, qubit_nparams = count_cn(qubit_gates_t)