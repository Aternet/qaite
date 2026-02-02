import openfermion as of
from openfermion import MolecularData, FermionOperator, QubitOperator, transforms,get_fermion_operator, jw_configuration_state, hermitian_conjugated, normal_ordered
from openfermion.transforms import binary_code_transform, parity_code
from openfermion.linalg import get_sparse_operator # type: ignore
from openfermionpyscf import run_pyscf
from copy import copy
import scipy
import scipy.sparse
import numpy as np
from tqdm import tqdm
import re
import os
from joblib import Parallel, delayed
from time import time
import csv
import warnings
warnings.filterwarnings("ignore")


class Molecule:
    """
    A class to represent a molecule and generate its Hamiltonian and excitation pools
    for quantum simulations using OpenFermion.
    """

    def __init__(
        self,
        geometry: list,
        basis: str = 'sto-3g',
        multiplicity: int = 1,
        charge: int = 0,
        freeze: int = 0
    ):
        """
        Initialize the Molecule.

        Args:
            geometry (list): List of tuples representing atomic coordinates, e.g., [('H', (0, 0, 0)), ...].
            basis (str): Basis set to use (default: 'sto-3g').
            multiplicity (int): Spin multiplicity (default: 1).
            charge (int): Molecular charge (default: 0).
            freeze (int): Number of orbitals to freeze (default: 0).
        """
        self.mol = MolecularData(geometry=geometry, basis=basis, multiplicity=multiplicity, charge=charge)
        self.mol = run_pyscf(self.mol, run_scf=True, run_fci=True)

        self.H = get_fermion_operator(self.mol.get_molecular_hamiltonian())
        
        # Freeze core orbitals if requested
        self.H = transforms.freeze_orbitals(self.H, [n for n in range(freeze)], None, prune=True)

        self.n_electrons: int = self.mol.n_electrons - freeze
        self.n_qubits = self.mol.n_qubits - freeze
        self.hf_energy = self.mol.hf_energy
        self.fci_energy = self.mol.fci_energy
        self.basis = basis
        self.multiplicity = multiplicity
        self.charge = charge
        self.freeze = freeze

    def get_hamiltonian(self, transform: str = 'jw'):
        """
        Get the Qubit Hamiltonian and reference state.

        Args:
            transform (str): Mapping transform to use. Supported: 'jw', 'parity'.

        Returns:
            tuple: (H_csr, ref) where H_csr is the sparse Hamiltonian matrix and ref is the reference state vector.
        """
        if transform == 'jw':
            self.qubit_H = transforms.jordan_wigner(self.H)
            # Create reference state |11...100...0> corresponding to filled orbitals
            ref = scipy.sparse.csc_matrix(jw_configuration_state(list(range(0, self.n_electrons)), self.n_qubits)).T
        elif transform == 'parity':
            self.qubit_H = binary_code_transform(self.H, parity_code(self.n_qubits))
            
            # For Parity: q_i = sum_{j=0}^i n_j (mod 2)
            # Reference: First n_electrons orbitals are filled (1), rest are empty (0).
            # n = [1, 1, ..., 1, 0, 0, ...]
            # q_0 = n_0 = 1
            # q_1 = n_0 + n_1 = 1 + 1 = 0 (mod 2)
            # q_2 = n_0 + n_1 + n_2 = 1 + 1 + 1 = 1 (mod 2) ...
            # Pattern: 1, 0, 1, 0, ... for the filled part.
            # Once we hit unfilled (0), the parity sum stays constant.
            
            ref_occupations = [0] * self.n_qubits
            current_parity = 0
            for i in range(self.n_qubits):
                if i < self.n_electrons:
                    occ = 1 
                else:
                    occ = 0
                current_parity = (current_parity + occ) % 2
                ref_occupations[i] = current_parity
            
            # Create computational basis state from this 0/1 list
            # Reverse order because typical "binary string" interpretation vs qubit index can vary, 
            # but usually qubit 0 is LSB or MSB. OpenFermion 'jw_configuration_state' takes list of indices occupied.
            # Here we just want a specific basis state.
            
            # Let's construct the integer index directly.
            # Qubit 0 is usually the "rightmost" in binary string printing, but in array indexing commonly n-1.
            # However, OpenFermion's `jw_configuration_state` maps `[0, 1]` to |1100...>.
            
            # Let's use a cleaner way:
            val = 0
            for i, bit in enumerate(ref_occupations):
                if bit:
                    val += 2**(self.n_qubits - 1 - i) # Qubit 0 is MSB
            
            ref = scipy.sparse.csc_matrix((2**self.n_qubits, 1), dtype=complex)
            ref[val, 0] = 1.0
            
        else:
            raise ValueError(f"Transform '{transform}' is not supported. Choose 'jw' or 'parity'.")

        H_csr = get_sparse_operator(self.qubit_H, n_qubits=self.n_qubits).tocsr().real.astype(np.float64)
        # Clean up small numerical noise
        H_csr.data[np.abs(H_csr.data) < 2.2e-16] = 0
        H_csr.eliminate_zeros()

        return H_csr, ref

    def print_info(self, pretty: bool = False):
        """
        Print basic information about the molecule: number of qubits, electrons, HF and FCI energies.
        
        Args:
            pretty (bool): If True, display a Unicode box table with aligned keys and values.
        """
        if not pretty:
            print("Molecule Information:")
            print(f"  Number of qubits: {self.n_qubits}")
            print(f"  Number of electrons: {self.n_electrons}")
            print(f"  Hartree-Fock (HF) energy: {self.hf_energy:.8f}")
            print(f"  Full Configuration Interaction (FCI) energy: {self.fci_energy:.8f}")
            return

        # Prepare data for table
        properties = [
            ("Multiplicity", f"{int(self.multiplicity)}"),
            ("Charge", f"{int(self.charge)}"),
            ("Freeze", f"{int(self.freeze)}"),
            ("Basis", f"{self.basis}"),
            ("Number of qubits", f"{int(self.n_qubits)}"),
            ("Number of electrons", f"{int(self.n_electrons)}"),
            ("Hartree-Fock (HF) energy", f"{self.hf_energy:.8f} Ha"),
            ("Full Configuration Interaction (FCI) energy", f"{self.fci_energy:.8f} Ha"),
        ]

        # Determine column widths
        key_width = max(len("Property"), max(len(k) for k, _ in properties))
        val_width = max(len("Value"), max(len(v) for _, v in properties))

        # Unicode box drawing characters
        h_line = "─"
        v_line = "│"
        
        # Construct borders
        def make_separator(left, mid, right):
            return left + h_line * (key_width + 2) + mid + h_line * (val_width + 2) + right

        top_border = make_separator("┌", "┬", "┐")
        header_sep = make_separator("├", "┼", "┤")
        bottom_border = make_separator("└", "┴", "┘")

        # Print the table
        print(top_border)
        print(f"{v_line} {'Property'.ljust(key_width)} {v_line} {'Value'.ljust(val_width)} {v_line}")
        print(header_sep)
        for k, v in properties:
            print(f"{v_line} {k.ljust(key_width)} {v_line} {v.ljust(val_width)} {v_line}")
        print(bottom_border)

    def format_xyz(self) -> str:
        """
        Return the current geometry in standard XYZ format as a string.
        Units: Angstrom.
        """
        lines = [str(len(self.mol.geometry)), "Generated by vqengine.Molecule"]
        for sym, (x, y, z) in self.mol.geometry:
            lines.append(f"{sym} {x:.8f} {y:.8f} {z:.8f}")
        return "\n".join(lines) + "\n"

    def save_xyz(self, path: str):
        """Save the geometry to an XYZ file."""
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.format_xyz())

    def print_structure(self, pretty: bool = False):
        """
        Print atomic coordinates.
        
        Args:
            pretty (bool): If True, use a Unicode box table for atoms.
        """
        geom = self.mol.geometry

        # Prepare atom rows
        atom_rows = []
        for idx, (sym, (x, y, z)) in enumerate(geom):
            atom_rows.append((idx, sym, f"{x:.6f}", f"{y:.6f}", f"{z:.6f}"))

        if not pretty:
            print("Atoms (index, symbol, x, y, z) [Å]:")
            for r in atom_rows:
                print(f"  {r[0]:>3}  {r[1]:>2}  {r[2]:>10}  {r[3]:>10}  {r[4]:>10}")
            return

        # Pretty table (Unicode box drawing)
        def box_table(headers, rows):
            col_w = [max(len(h), *(len(str(r[c])) for r in rows)) for c, h in enumerate(headers)]
            
            def make_sep(left, mid, right, line="─"):
                return left + mid.join(line * (w + 2) for w in col_w) + right

            top = make_sep("┌", "┬", "┐")
            mid = make_sep("├", "┼", "┤")
            bot = make_sep("└", "┴", "┘")

            def fmt_row(row):
                return "│" + "│".join(f" {str(val).ljust(w)} " for val, w in zip(row, col_w)) + "│"

            print(top)
            print(fmt_row(headers))
            print(mid)
            for row in rows:
                print(fmt_row(row))
            print(bot)

        print("Atoms [Å]:")
        box_table(["idx", "el", "x", "y", "z"], atom_rows)


def generate_uccsd_pool(n_qubits: int, n_electrons: int, transform: str = 'jw', approach: str = 'spin_adapt'):
    """
    Generate the UCCSD excitation pool.

    Args:
        n_qubits (int): Number of qubits.
        n_electrons (int): Number of electrons.
        transform (str): Mapping transform (default: 'jw').
        approach (str): Approach for pool generation (default: 'spin_adapt').

    Returns:
        tuple: (csr_qubit_pool, v_pool) where csr_qubit_pool is a list of sparse matrices and v_pool is a list of QubitOperators.
    """
    print("Starting UCCSD excitation pool generation...")
    
    def spin_adapted_t1(i, j):
        ia = i * 2 + 0
        ib = i * 2 + 1
        ja = j * 2 + 0
        jb = j * 2 + 1
        term1 = FermionOperator(((ia, 1), (ja, 0)), 1.0)
        term2 = FermionOperator(((ib, 1), (jb, 0)), 1.0)
        tpq_list = [term1 + term2]
        return tpq_list
    
    def Pij(i: int, j: int):
        ia = i * 2 + 0
        ib = i * 2 + 1
        ja = j * 2 + 0
        jb = j * 2 + 1
        term1 = FermionOperator(((ja, 0), (ib, 0)), 1.0)
        term2 = FermionOperator(((ia, 0), (jb, 0)), 1.0)
        return np.sqrt(0.5) * (term1 + term2)

    def Pij_dagger(i: int, j: int):
        return hermitian_conjugated(Pij(i, j))

    def Qij_plus(i: int, j: int):
        ia = i * 2 + 0
        ja = j * 2 + 0
        term = FermionOperator(((ja, 0), (ia, 0)), 1.0)
        return term

    def Qij_minus(i: int, j: int):
        ib = i * 2 + 1
        jb = j * 2 + 1
        term = FermionOperator(((jb, 0), (ib, 0)), 1.0)
        return term

    def Qij_0(i: int, j: int):
        ia = i * 2 + 0
        ib = i * 2 + 1
        ja = j * 2 + 0
        jb = j * 2 + 1
        term1 = FermionOperator(((ja, 0), (ib, 0)), 1.0)
        term2 = FermionOperator(((ia, 0), (jb, 0)), 1.0)
        return np.sqrt(0.5) * (term1 - term2)

    def Qij_vec(i: int, j: int):
        return [Qij_plus(i, j), Qij_minus(i, j), Qij_0(i, j)]

    def Qij_vec_dagger(i: int, j: int):
        return [hermitian_conjugated(x) for x in Qij_vec(i, j)]

    def Qij_vec_inner(a: int, b: int, i: int, j: int):
        vec_dagger = Qij_vec_dagger(a, b)
        vec = Qij_vec(i, j)
        return sum([vec[k] * vec_dagger[k] for k in range(len(vec))])

    def spin_adapted_t2(creation_list, annihilation_list):
        p = creation_list[0]
        r = annihilation_list[0]
        q = creation_list[1]
        s = annihilation_list[1]
        tpqrs1 = Pij_dagger(p, q) * Pij(r, s)
        tpqrs2 = Qij_vec_inner(p, q, r, s)
        tpqrs_list = [tpqrs1, tpqrs2]
        return tpqrs_list

    if approach == 'spin_adapt':
        n_spatial = n_qubits // 2
        n_occ = n_electrons // 2
        vir_indices = list(range(n_occ, n_spatial))
        occ_indices = list(range(n_occ))
        fermionic_pool = []

        # Singles
        for p in tqdm(vir_indices, desc="Generating Singles Excitations"):
            for q in occ_indices:
                tpq_list = spin_adapted_t1(p, q)
                for tpq in tpq_list:
                    tpq = tpq - hermitian_conjugated(tpq)
                    tpq = normal_ordered(tpq)
                    if tpq.many_body_order() > 0:
                        fermionic_pool.append(tpq)
        # Doubles
        for p_idx, p in enumerate(tqdm(vir_indices, desc="Generating Doubles Excitations")):
            for q_idx, q in enumerate(vir_indices[p_idx:], start=p_idx):
                for r in occ_indices:
                    for s in occ_indices[occ_indices.index(r):]:
                        tpqrs_list = spin_adapted_t2([p, q], [r, s])
                        for tpqrs in tpqrs_list:
                            tpqrs = tpqrs - hermitian_conjugated(tpqrs)
                            tpqrs = normal_ordered(tpqrs)
                            if tpqrs.many_body_order() > 0:
                                fermionic_pool.append(tpqrs)
    else:
        raise ValueError("approach must be 'spin_adapt'")

    # Transform to QubitOperator
    csr_qubit_pool = []
    v_pool = []
    for ferm_op in fermionic_pool:
        if transform == 'jw':
            qubit_op = transforms.jordan_wigner(ferm_op)
        elif transform == 'parity':
            qubit_op = binary_code_transform(ferm_op, parity_code(n_qubits))
        else:
            raise ValueError(f"Transform '{transform}' is not supported. Choose 'jw' or 'parity'.")
        qubit_op_antiherm = 0.5 * (qubit_op - hermitian_conjugated(qubit_op))
        v_pool.append(qubit_op_antiherm)
        h = get_sparse_operator(qubit_op_antiherm, n_qubits=n_qubits).tocsr().real.astype(np.float64)
        csr_qubit_pool.append(h)

    print(f"Generated {len(v_pool)} UCCSD Excitations")
    return csr_qubit_pool, v_pool

def generate_uccgsd_pool(n_qubits: int, transform: str = 'jw', approach: str = 'spin_adapt'):
    """
    Generate the UCCGSD excitation pool (Generalized Singles and Doubles).
    
    Args:
        n_qubits (int): Number of qubits.
        transform (str): Mapping transform (default: 'jw').
        approach (str): Approach for pool generation (default: 'spin_adapt').

    Returns:
        tuple: (csr_qubit_pool, v_pool) where csr_qubit_pool is a list of sparse matrices and v_pool is a list of QubitOperators.
    """
    print("Starting UCCGSD excitation pool generation...")
    
    def spin_adapted_t1(i, j):
        ia = i * 2 + 0
        ib = i * 2 + 1
        ja = j * 2 + 0
        jb = j * 2 + 1
        term1 = FermionOperator(((ia, 1), (ja, 0)), 1.0)
        term2 = FermionOperator(((ib, 1), (jb, 0)), 1.0)
        tpq_list = [term1 + term2]
        return tpq_list
    
    def Pij(i: int, j: int):
        ia = i * 2 + 0
        ib = i * 2 + 1
        ja = j * 2 + 0
        jb = j * 2 + 1
        term1 = FermionOperator(((ja, 0), (ib, 0)), 1.0)
        term2 = FermionOperator(((ia, 0), (jb, 0)), 1.0)
        return np.sqrt(0.5) * (term1 + term2)

    def Pij_dagger(i: int, j: int):
        return hermitian_conjugated(Pij(i, j))

    def Qij_plus(i: int, j: int):
        ia = i * 2 + 0
        ja = j * 2 + 0
        term = FermionOperator(((ja, 0), (ia, 0)), 1.0)
        return term

    def Qij_minus(i: int, j: int):
        ib = i * 2 + 1
        jb = j * 2 + 1
        term = FermionOperator(((jb, 0), (ib, 0)), 1.0)
        return term

    def Qij_0(i: int, j: int):
        ia = i * 2 + 0
        ib = i * 2 + 1
        ja = j * 2 + 0
        jb = j * 2 + 1
        term1 = FermionOperator(((ja, 0), (ib, 0)), 1.0)
        term2 = FermionOperator(((ia, 0), (jb, 0)), 1.0)
        return np.sqrt(0.5) * (term1 - term2)

    def Qij_vec(i: int, j: int):
        return [Qij_plus(i, j), Qij_minus(i, j), Qij_0(i, j)]

    def Qij_vec_dagger(i: int, j: int):
        return [hermitian_conjugated(x) for x in Qij_vec(i, j)]

    def Qij_vec_inner(a: int, b: int, i: int, j: int):
        vec_dagger = Qij_vec_dagger(a, b)
        vec = Qij_vec(i, j)
        return sum([vec[k] * vec_dagger[k] for k in range(len(vec))])

    def spin_adapted_t2(creation_list, annihilation_list):
        p = creation_list[0]
        r = annihilation_list[0]
        q = creation_list[1]
        s = annihilation_list[1]
        tpqrs1 = Pij_dagger(p, q) * Pij(r, s)
        tpqrs2 = Qij_vec_inner(p, q, r, s)
        tpqrs_list = [tpqrs1, tpqrs2]
        return tpqrs_list
        

    if approach == 'spin_adapt':
        n_spatial = n_qubits // 2
        indices = list(range(n_spatial))
        fermionic_pool = []
        # Singles
        for p in tqdm(indices, desc="Generating Singles Excitations (GSD)"):
            for q in indices:
                if p == q:
                    continue
                tpq_list = spin_adapted_t1(p, q)
                for tpq in tpq_list:
                    tpq = tpq - hermitian_conjugated(tpq)
                    tpq = normal_ordered(tpq)
                    if tpq.many_body_order() > 0:
                        fermionic_pool.append(tpq)
        # Doubles
        pq = -1
        for p_idx in tqdm(range(len(indices)), desc="Generating Doubles Excitations (GSD)"):
            p = indices[p_idx]
            for q_idx in range(p_idx, len(indices)):
                q = indices[q_idx]
                pq += 1
                rs = -1
                for r_idx in range(len(indices)):
                    r = indices[r_idx]
                    for s_idx in range(r_idx, len(indices)):
                        s = indices[s_idx]
                        rs += 1
                        if pq > rs:
                            continue
                        tpqrs_list = spin_adapted_t2([p, q], [r, s])
                        for tpqrs in tpqrs_list:
                            tpqrs = tpqrs - hermitian_conjugated(tpqrs)
                            tpqrs = normal_ordered(tpqrs)
                            if tpqrs.many_body_order() > 0:
                                fermionic_pool.append(tpqrs)
    else:
        raise ValueError("approach must be 'spin_adapt'")

    # Convert to QubitOperator
    csr_qubit_pool = []
    v_pool = []
    for ferm_op in fermionic_pool:
        if transform == 'jw':
            qubit_op = transforms.jordan_wigner(ferm_op)
        elif transform == 'parity':
            qubit_op = binary_code_transform(ferm_op, parity_code(n_qubits))
        else:
            raise ValueError(f"Transform '{transform}' is not supported. Choose 'jw' or 'parity'.")
        qubit_op_antiherm = 0.5 * (qubit_op - hermitian_conjugated(qubit_op))
        v_pool.append(qubit_op_antiherm)
        h = get_sparse_operator(qubit_op_antiherm, n_qubits=n_qubits).tocsr().real.astype(np.float64)
        csr_qubit_pool.append(h)
    print(f"Generated {len(v_pool)} UCCGSD Excitations")
    return csr_qubit_pool , v_pool

def _build_fermionic_gsd_pool(n_qubits: int, n_electrons: int):
    """
    Build the Spin-adapted Generalized Singles and Doubles (GSD) fermionic operator pool.
    
    Args:
        n_qubits (int): Number of qubits.
        n_electrons (int): Number of electrons.
        
    Returns:
        list: List of FermionOperators.
    """
    M = int(n_qubits / 2)
    # N = int(n_electrons / 2) # Unused
    sq_pool = []
    
    # Singles
    for p in range(0, M):
        pa = 2 * p
        pb = 2 * p + 1
        for q in range(p + 1, M):
            qa = 2 * q
            qb = 2 * q + 1
            term = (1/np.sqrt(2) * of.ops.FermionOperator(((pa, 1), (qa, 0))) + 
                    1/np.sqrt(2) * of.ops.FermionOperator(((pb, 1), (qb, 0))))
            term -= of.utils.hermitian_conjugated(term)
            sq_pool.append(term)
    
    # Doubles
    pq = -1
    for p in range(0, M):
        pa = 2 * p
        pb = 2 * p + 1
        for q in range(p, M):
            qa = 2 * q
            qb = 2 * q + 1
            pq += 1
            rs = -1
            for r in range(0, M):
                ra = 2 * r
                rb = 2 * r + 1
                for s in range(r, M):
                    sa = 2 * s
                    sb = 2 * s + 1
                    rs += 1
                    
                    if pq > rs: 
                        continue
                    
                    # Term A
                    termA = of.ops.FermionOperator(((ra, 1), (pa, 0), (sa, 1), (qa, 0)), 2 / np.sqrt(12))
                    termA += of.ops.FermionOperator(((rb, 1), (pb, 0), (sb, 1), (qb, 0)), 2 / np.sqrt(12))
                    termA += of.ops.FermionOperator(((ra, 1), (pa, 0), (sb, 1), (qb, 0)), 1 / np.sqrt(12))
                    termA += of.ops.FermionOperator(((rb, 1), (pb, 0), (sa, 1), (qa, 0)), 1 / np.sqrt(12))
                    termA += of.ops.FermionOperator(((ra, 1), (pb, 0), (sb, 1), (qa, 0)), 1 / np.sqrt(12))
                    termA += of.ops.FermionOperator(((rb, 1), (pa, 0), (sa, 1), (qb, 0)), 1 / np.sqrt(12))

                    # Term B
                    termB = of.ops.FermionOperator(((ra, 1), (pa, 0), (sb, 1), (qb, 0)), 1 / 2.0)
                    termB += of.ops.FermionOperator(((rb, 1), (pb, 0), (sa, 1), (qa, 0)), 1 / 2.0)
                    termB += of.ops.FermionOperator(((ra, 1), (pb, 0), (sb, 1), (qa, 0)), -1 / 2.0)
                    termB += of.ops.FermionOperator(((rb, 1), (pa, 0), (sa, 1), (qb, 0)), -1 / 2.0)
                    
                    termA -= of.utils.hermitian_conjugated(termA)
                    termB -= of.utils.hermitian_conjugated(termB)
                    
                    termA = of.transforms.normal_ordered(termA)
                    termB = of.transforms.normal_ordered(termB)
                    
                    if termA.many_body_order() > 0:
                        sq_pool.append(termA)
                    if termB.many_body_order() > 0:
                        sq_pool.append(termB)
    return sq_pool

def extract_pauli_strings_from_qubit_pool(qubit_pool, n_qubits):
    """
    Extract unique Pauli strings from a list of QubitOperators.

    Args:
        qubit_pool (list): List of QubitOperators.
        n_qubits (int): Number of qubits.

    Returns:
        tuple: (jw_pool, fermi_ops) where jw_pool is list of sparse matrices and fermi_ops is list of QubitOperators.
    """
    print(f"Extracting Pauli strings from {len(qubit_pool)} operators...")
    
    # Pool vector size depends on number of qubits
    M = int(n_qubits / 2)
    n = 2 * M # Effective number of spin-orbitals/qubits
    pool_vec = np.zeros((4 ** n,), dtype=int) 

    # Precompile regex patterns
    _X_PAT = re.compile(r"(\d{1,2}), 'X'")
    _Y_PAT = re.compile(r"(\d{1,2}), 'Y'")
    _Z_PAT = re.compile(r"(\d{1,2}), 'Z'")

    for pauli in qubit_pool:
        for line in pauli.terms:
            line = str(line)
            Bin = np.zeros((2 * n,), dtype=int)
            X_1 = _X_PAT.findall(line)
            if X_1:
                for i in X_1:
                    Bin[n + int(i)] = 1
            Y_1 = _Y_PAT.findall(line)
            if Y_1:
                for i in Y_1:
                    k = int(i)
                    Bin[n + k] = 1
                    Bin[k] = 1
            Z_1 = _Z_PAT.findall(line)
            if Z_1:
                for i in Z_1:
                    Bin[int(i)] = 1
            
            # Binary string to integer index
            index = int("".join(str(x) for x in Bin), 2)
            pool_vec[index] = 1

    nz = np.nonzero(pool_vec)[0]
    print("Qubit Pool Size:", len(nz))

    fermi_ops = []
    jw_pool = []
    m = 2 * n

    for i in nz:
        p = int(i)
        b_string = [int(j) for j in bin(p)[2:].zfill(m)]
        pauli_string = ''
        flip = []
        for k in range(n):
            if b_string[k] == 0:
                if b_string[k + n] == 1:
                    pauli_string += f'X{k} '
                    flip.append(k)
            if b_string[k] == 1:
                if b_string[k + n] == 1:
                    pauli_string += f'Y{k} '
                    flip.append(k)
        flip.sort()
        
        if len(flip) >= 2:
            z_string = list(range(flip[0] + 1, flip[1]))
            if len(flip) == 4:
                for f_idx in range(flip[2] + 1, flip[3]):
                    z_string.append(f_idx)
            
            for idx in z_string:
                b_string[idx] = (b_string[idx] + 1) % 2
            
        for k in range(n):
            if b_string[k] == 1 and b_string[k + n] == 0:
                pauli_string += f'Z{k} '
                
        A = of.ops.QubitOperator(pauli_string, 0 + 1j)
        fermi_ops.append(A)
        jw_pool.append(of.get_sparse_operator(A, n_qubits).tocsr().real.astype(np.float64))
        
    return jw_pool, fermi_ops

def _extract_pauli_strings(sq_pool, n_qubits, transform='jw'):
    """
    Convert fermionic operators to QubitOperators (Pauli strings).
    
    Args:
        sq_pool (list): List of FermionOperators.
        n_qubits (int): Number of qubits.
        transform (str): Mapping transform (default: 'jw').
        
    Returns:
        tuple: (jw_pool, fermi_ops)
    """
    print(f"Extracting Pauli strings for {transform}...")
    
    qubit_pool = []
    for op in sq_pool:
        if transform == 'jw':
            pauli = of.transforms.jordan_wigner(op)
        elif transform == 'parity':
            pauli = binary_code_transform(op, parity_code(n_qubits))
        else:
             raise ValueError(f"Transform '{transform}' is not supported.")
        qubit_pool.append(pauli)
        
    return extract_pauli_strings_from_qubit_pool(qubit_pool, n_qubits)

def generate_qubit_pool(n_qubits: int, n_electrons: int, transform: str = 'jw'):
    """
    Generate the qubit excitation pool from Spin-adapted GSD.

    Args:
        n_qubits (int): Number of qubits.
        n_electrons (int): Number of electrons.
        transform (str): Mapping transform (default: 'jw').

    Returns:
        tuple: (jw_pool, fermi_ops) where jw_pool is a list of sparse matrices and fermi_ops is a list of QubitOperators.
    """
    # Build Spin-adapted GSD pool of fermionic ops
    sq_pool = _build_fermionic_gsd_pool(n_qubits, n_electrons)
    print(f"{len(sq_pool)} operators in the UCCGSD pool.")
    
    # Extract unique Pauli strings
    return _extract_pauli_strings(sq_pool, n_qubits, transform=transform)

def _grad_component(op, state, hbra):
    """Compute 2*Re( <state| H (op |state>) ) using precomputed hbra=(H|state>)^T."""
    oket = op @ state
    return float(2 * (hbra @ oket)[0, 0].real)


class driver:
    """
    Driver class for VQITE simulation.
    """
    def __init__(self, H, ref):
        """
        Initialize the driver.
        
        Args:
            H (sparse matrix): Hamiltonian.
            ref (sparse vector): Reference state.
        """
        self.H = copy(H)
        self.ref = copy(ref)

    def energy(self, params, ansatz):
        """Compute expectation value of H."""
        state = copy(self.ref)
        for i in reversed(range(0, len(ansatz))):
            state = scipy.sparse.linalg.expm_multiply(params[i]*ansatz[i], state)
        E = (state.T@(self.H)@state).todense()[0,0]
        return np.real(E)

    def gradient(self, params, ansatz):
        grad = []
        ket = copy(self.ref)
        for i in reversed(range(0, len(ansatz))):
            ket = scipy.sparse.linalg.expm_multiply(params[i]*ansatz[i], ket)
        hbra = (self.H@ket).T
        for i in range(0, len(ansatz)):
            grad.append(2*(hbra@ansatz[i]@ket).todense()[0,0])
            targ = scipy.sparse.hstack([hbra.T, ket])
            res = scipy.sparse.linalg.expm_multiply(-params[i]*ansatz[i], targ).tocsr()
            hbra = res[:,0].T
            ket = res[:,1]
        return np.real(np.array(grad))

    def vqe(self, ansatz, params, gtol = 1e-8):
        E0 = self.energy(params, ansatz)
        x = copy(params)
        res = scipy.optimize.minimize(self.energy, np.array(x), jac = self.gradient, method = "bfgs", args = (ansatz), options = {"gtol": gtol})
        x = copy(res.x)
        return res.fun, res.x, res.jac, E0, params

    def ucc_state(self, params, ansatz):
        state = copy(self.ref)
        for i in reversed(range(0, len(ansatz))):
            state = scipy.sparse.linalg.expm_multiply(params[i]*ansatz[i], state)
        return state

    def adapt(self, pool, g_tol=1e-3, n_jobs=-1):
        energy = []
        ansatz = []
        params = np.array([])
        try:
            state = self.ucc_state(params, ansatz)
        except:
            state = copy(self.ref)
        E = (state.T @ (self.H @ state))[0, 0].real
        energy.append(E)
        print("Performing ADAPT:")
        Done = False
        while not Done:
            # Precompute ket and hbra once per iteration
            ket = state
            hbra = (self.H @ ket).T
            # Sequential gradient (simple and reliable)
            # gradient = 2 * np.array([((hbra @ (op @ ket))[0, 0]) for op in pool]).real
            gradient = Parallel(n_jobs=n_jobs)(
                delayed(_grad_component)(
                    op, ket, hbra
                )
                for op in pool
            )
            gradient = np.array(gradient)

            gnorm = np.linalg.norm(gradient)
            if gnorm < g_tol:
                print(f"g norm converged.")
                break
            max_g = np.argmax(abs(gradient))
            ansatz = [pool[max_g]] + ansatz
            params = np.append(0.0, params)
            E, params, *_ = self.vqe(ansatz, params, gtol=1e-12)
            state = self.ucc_state(params, ansatz)
            energy.append(E)
            print(f"selectec {max_g} : Optimized Energy = {E:.8f} with error = ")

        return 1

    def ML_M(self, params, ansatz):
        N = len(ansatz)
        M = np.zeros((N, N))
        # V = np.zeros((N))    
        _ket = copy(self.ref)
        vec = []
        for u in reversed(range(0, N)):
            _ket = scipy.sparse.linalg.expm_multiply(params[u]*ansatz[u], _ket)
            uket = ansatz[u]@_ket
            vec = [(_ket.T@uket).todense()[0,0]] + vec
            _uket = copy(uket)
            _vket = copy(_ket)
            for v in reversed(range(0, u+1)):
                targ = scipy.sparse.hstack([_uket, _vket])
                res = scipy.sparse.linalg.expm_multiply(params[v]*ansatz[v], targ).tocsr() 
                _uket = res[:,0]
                _vket = res[:,1]
                M[u,v] = M[v,u] = -2*(_vket.T@(ansatz[v]@_uket)).todense()[0,0]             
        for u in reversed(range(0, N)):
            for v in reversed(range(0, u+1)):
                M[u,v] += 2*vec[u]*vec[v]
                if u != v:
                    M[v,u] += 2*vec[u]*vec[v]
        return M
    
    def L2_McLachlan(self, V, M, psi):
        E = (psi.T @ (self.H @ psi))[0, 0].real
        E2 = (psi.T @ (self.H @ (self.H @ psi)))[0, 0].real
        variance = E2 - E**2
        reg = 1e-6
        M_reg = M + reg * np.eye(M.shape[0])
        invM = np.linalg.inv(M_reg)
        term = V @ (invM @ V)
        #pthvec = invM @ V
        #pthmax = np.max(np.abs(pthvec))
        L2 = (2 * variance - term)
        return L2, variance
    
    def ITE_step(self, ansatz, params, iter=0, dt = 1e-1):
        """
        Perform one Imaginary Time Evolution step.
        """
        old_E = self.energy(params, ansatz)
        # E = copy(old_E) # Unused
        V = -self.gradient(params, ansatz)
        M = self.ML_M(params, ansatz)
        reg = 1e-6
        M_reg = M + reg * np.eye(M.shape[0])
        invM = np.linalg.inv(M_reg)
        direction = invM @ V
        params += dt * direction.real
        E = self.energy(params, ansatz)    
        print(f"  [ITE Step {iter}] Energy: {old_E:.8f} -> {E:.8f}")
        return E, list(params)
    
    def compute_L2_for_A(self,i, A, ansatz, params, gradient, ML_M, L2_McLachlan):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trial_ansatz = [A] + ansatz
            trial_params = np.append(0.0, params)
            V = -gradient(trial_params, trial_ansatz)
            M = ML_M(trial_params, trial_ansatz)
            psi = copy(self.ref)
            for u in reversed(range(len(trial_ansatz))):
                psi = scipy.sparse.linalg.expm_multiply(trial_params[u] * trial_ansatz[u], psi)
            L2, variance = L2_McLachlan(V, M, psi)
            # L2, pthmax = L2_McLachlan(H, V, M, psi)
            return (i, L2, variance)

    def avqite(self, ops_pool, v_pool, dt=1e-1, grad_tol=5e-4, t=5, n_jobs=-1):
        """
        Adaptive VQITE algorithm.
        
        Args:
            ops_pool: Pool of operators.
            v_pool: Corresponding V operators.
            dt: Time step.
            grad_tol: Gradient tolerance for convergence.
            t: Total simulation time.
            n_jobs: Number of parallel jobs.
        """
        print("\n" + "="*40)
        print("Starting ADAPT-VQITE Simulation")
        print("="*40)
        
        ansatz = []
        v = []
        v_t = []
        L2_t = []
        L2_iter = []
        variance_t = []
        variance_iter = []
        params = np.array([])
        energies = [self.energy(params, ansatz)]
        t_steps = round(t / dt)
        
        L2 = np.inf
        variance = None
        
        for adapt_step in range(int(t_steps)):
            pool = copy(ops_pool)
            print(f"\n--- ADAPT Step {adapt_step + 1} / {t_steps} ---")
            
            if L2 < grad_tol:
                print(f"  Converged: L² ({L2:.1e}) < tol ({grad_tol:.1e})")
                L2_iter.append(L2)
                variance_iter.append(variance)
                adapt = False
            else:
                adapt = True
                
            while adapt:
                # ---- Parallel L2 computation over pool ----
                results = Parallel(n_jobs=n_jobs)(
                    delayed(self.compute_L2_for_A)(
                        i, A, ansatz, params, self.gradient, self.ML_M, self.L2_McLachlan
                    )
                    for i, A in enumerate(pool)
                )
                
                L2_scores = np.zeros(len(pool))
                # Store variances just in case we need them
                variances = {}
                
                for i, L2_val, variance_val in results:
                    L2_scores[i] = L2_val
                    variances[i] = variance_val

                sorted_indices = np.argsort(L2_scores)
                best_idx = sorted_indices[0]
                best_L2 = L2_scores[best_idx]
                best_variance = variances[best_idx] # Correctly retrieve variance for best op

                if best_L2 < grad_tol:
                    print(f"  Converged: L² ({best_L2:.1e}) < tol ({grad_tol:.1e})")
                    L2_iter.append(best_L2)
                    variance_iter.append(best_variance)
                    break

                # Add operator to ansatz
                chosen_op = pool[best_idx]
                ansatz = [chosen_op] + ansatz
                v = [v_pool[best_idx]] + v
                L2_iter.append(best_L2)
                variance_iter.append(best_variance)
                params = np.append(0.0, params)
                print(f"  Selected Op: {best_idx:<4} | L²: {best_L2:.6e} | Total Params: {len(params)}")

            # Optimize using VQITE or ITE_step (user-provided!)
            t_vqite_start = time()
            E, params = self.ITE_step(ansatz, params, iter=adapt_step+1, dt=dt)
            t_vqite_end = time()
            # print(f"  VQITE Step Time: {t_vqite_end - t_vqite_start:.2f}s")
            energies.append(E)

            # Compute new L2 for next step
            L2_t.append((adapt_step+1, L2_iter))
            variance_t.append((adapt_step+1, variance_iter))
            
            V = -self.gradient(params, ansatz)
            M = self.ML_M(params, ansatz)
            psi = copy(self.ref)
            for u in reversed(range(len(ansatz))):
                psi = scipy.sparse.linalg.expm_multiply(params[u] * ansatz[u], psi)
            L2, variance = self.L2_McLachlan(V, M, psi)
            v_t.append((adapt_step+1, v))
            L2_iter = []
            variance_iter = []

        return energies, params, ansatz, v_t, L2_t, variance_t

def get_t_L2(L2_t):
    """Flatten L2 trajectories for plotting/saving."""
    t_L2 = []
    Lsq = []
    for L2_at_t in L2_t:
        t_L2 += [L2_at_t[0]]*len(L2_at_t[1])
        Lsq += L2_at_t[1]
    return Lsq, t_L2

def get_t_variance(L2_t):
    """Flatten variance trajectories for plotting/saving."""
    t_L2 = []
    Lsq = []
    for L2_at_t in L2_t:
        t_L2 += [L2_at_t[0]]*len(L2_at_t[1])
        Lsq += L2_at_t[1]
    return Lsq, t_L2

def count_paulis(qubit_op):
    """Count total X, Y, Z in an OpenFermion QubitOperator."""
    x_count = y_count = z_count = 0
    for term, _ in qubit_op.terms.items():
        for _, pauli in term:
            if pauli in ('X', 'Y', 'Z'):
                x_count += 1
            elif pauli == 'Y': # Redundant in original but maintained logic
                 y_count += 1
            elif pauli == 'Z':
                 z_count += 1
    # Original logic:
    # if pauli == 'X': x+=1
    # elif pauli == 'Y': y+=1
    # elif pauli == 'Z': z+=1
    # The redundant elifs above were my misinterpretation of my own thought.
    # Restoring EXACT logic from original function:
    
    x_count = 0
    y_count = 0
    z_count = 0
    for term, coeff in qubit_op.terms.items():
        for qubit, pauli in term:
            if pauli == 'X':
                x_count += 1
            elif pauli == 'Y':
                y_count += 1
            elif pauli == 'Z':
                z_count += 1
    return x_count + y_count + z_count

def count_cn(v_gates):
    """Count CNOTs and parameters in the implementation."""
    N_cnot = [0]
    N_params = [0]
    for v in v_gates:
        _, ops = v
        count = 0
        for op in ops:
            count += count_paulis(op)
        N_cnot.append(2*(count-1))
        N_params.append(len(ops))
    return N_cnot, N_params

def save_results_csv(filename, headers, rows):
    """
    Save list of rows to a CSV file.
    """
    try:
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)
        print(f"Results saved to {filename}")
    except IOError as e:
        print(f"Error saving {filename}: {e}")

def main():
    pretty = True
    Name = "He-2"
    r = 1.5
    dt = 0.15
    total_t = 9
    
    print(f"--- Simulation: {Name} (r={r}) ---")
    
    # Define geometry
    geometry = [("He", (0., 0., n*r)) for n in range(2)]
    
    # Initialize Molecule
    mol = Molecule(geometry, basis="6-31g", freeze=0)
    #H, hf = mol.get_hamiltonian(transform='parity')
    H, hf = mol.get_hamiltonian(transform='jw')
    
    # Print Molecule Info
    mol.print_structure(pretty=pretty)
    mol.print_info(pretty=pretty)
    import sys as sys_module # Avoid conflict with 'sys' variable used below if any
    
    # Generate Pools
    # uccsd_pool, uccsd_v_pool = generate_uccsd_pool(mol.n_qubits, mol.n_electrons, transform='parity')

    # qubit_pool, qubit_v_pool = extract_pauli_strings_from_qubit_pool(uccsd_v_pool, mol.n_qubits)
    qubit_pool, qubit_v_pool = generate_qubit_pool(mol.n_qubits, mol.n_electrons, "jw")
    # Initialize Driver
    sys_driver = driver(H, hf) # Renamed 'sys' to 'sys_driver' for clarity
    
    # Run AVQITE
    qubit_E, qubit_thetas, qubit_gates, qubit_gates_t, qubit_L2_t, qubit_variance_t = sys_driver.avqite(
        qubit_pool, qubit_v_pool, dt=dt, grad_tol=5e-4, t=total_t
    )

    # Post-processing
    qubit_errors = [abs(e - mol.fci_energy) for e in qubit_E]
    t_vals = [dt * n for n in range(len(qubit_E))]
    
    qubit_L2, qubit_t_L2 = get_t_L2(qubit_L2_t)
    qubit_variance, _ = get_t_variance(qubit_variance_t)
    qubit_t_L2 = [m*dt for m in qubit_t_L2]
    
    tang_ncn, tang_nparams = count_cn(qubit_gates_t)

    print(f"\nVerification: len(var) == len(L2): {len(qubit_variance) == len(qubit_L2)}")

    # Save Results
    # CSV #1
    rows1 = zip(t_vals, qubit_E, qubit_errors, tang_ncn, tang_nparams)
    save_results_csv(f"results_{Name}.csv", ["t", "Energy", "Error", "n_cnots", "n_params"], rows1)

    # CSV #2
    rows2 = zip(qubit_t_L2, qubit_L2, qubit_variance)
    save_results_csv(f"L2_{Name}.csv", ["t_L2", "L2", "Var"], rows2)

    # To save XYZ for visualization tools (e.g., Avogadro/VMD):
    # mol.save_xyz("molecule.xyz")

if __name__ == "__main__":
    main()
