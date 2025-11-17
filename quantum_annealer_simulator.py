#!/usr/bin/env python3
"""
Quantum Annealer Simulator using QAOA (Quantum Approximate Optimization Algorithm)
Built with PennyLane for solving QUBO problems in the test generation pipeline.
"""

import pennylane as qml
from pennylane import numpy as np
import scipy.optimize
from collections import defaultdict
import time
import itertools

class QuantumAnnealerSimulator:
    """
    Quantum Annealer Simulator using QAOA for QUBO optimization.
    
    This simulator provides a quantum computing approach to solving
    combinatorial optimization problems like minimal test suite generation.
    """
    
    def __init__(self, backend='default.qubit', shots=1000, layers=2, seed=None):
        """
        Initialize the quantum annealer simulator.
        
        Args:
            backend: PennyLane device backend
            shots: Number of shots for sampling
            layers: Number of QAOA layers (depth)
            seed: Random seed for reproducibility (optional)
        """
        self.backend = backend
        self.shots = shots
        self.layers = layers
        self.seed = seed if seed is not None else 42
        self.n_qubits = 0
        self.device = None
        self.var_to_qubit = {}
        self.qubit_to_var = {}
        
    def qubo_to_pauli(self, qubo):
        """
        Convert QUBO formulation to Pauli string representation for quantum circuits.
        
        Args:
            qubo: Dictionary representing QUBO terms
            
        Returns:
            tuple: (Pauli coefficients, variable mapping)
        """
        # Extract unique variables
        variables = set()
        for (var1, var2), coeff in qubo.items():
            variables.add(var1)
            variables.add(var2)
        
        variables = sorted(list(variables))
        self.n_qubits = len(variables)
        
        # Create variable to qubit mapping
        self.var_to_qubit = {var: i for i, var in enumerate(variables)}
        self.qubit_to_var = {i: var for i, var in enumerate(variables)}
        
        # Initialize device
        self.device = qml.device(self.backend, wires=self.n_qubits, shots=self.shots)
        
        # Convert QUBO to Pauli strings
        pauli_coeffs = []
        
        for (var1, var2), coeff in qubo.items():
            if var1 == var2:
                # Diagonal term: (1 - Z_i)/2 corresponds to variable x_i
                # QUBO: coeff * x_i -> Pauli: coeff * (1 - Z_i)/2
                qubit_i = self.var_to_qubit[var1]
                
                # Identity term: coeff/2
                identity_coeff = coeff / 2
                pauli_coeffs.append((identity_coeff, "I" * self.n_qubits))
                
                # Z term: -coeff/2 * Z_i
                z_coeff = -coeff / 2
                pauli_string = ["I"] * self.n_qubits
                pauli_string[qubit_i] = "Z"
                pauli_coeffs.append((z_coeff, "".join(pauli_string)))
                
            else:
                # Off-diagonal term: x_i * x_j -> (1 - Z_i)(1 - Z_j)/4
                qubit_i = self.var_to_qubit[var1]
                qubit_j = self.var_to_qubit[var2]
                
                # Identity term: coeff/4
                identity_coeff = coeff / 4
                pauli_coeffs.append((identity_coeff, "I" * self.n_qubits))
                
                # Z_i term: -coeff/4 * Z_i
                z_i_coeff = -coeff / 4
                pauli_string = ["I"] * self.n_qubits
                pauli_string[qubit_i] = "Z"
                pauli_coeffs.append((z_i_coeff, "".join(pauli_string)))
                
                # Z_j term: -coeff/4 * Z_j
                z_j_coeff = -coeff / 4
                pauli_string = ["I"] * self.n_qubits
                pauli_string[qubit_j] = "Z"
                pauli_coeffs.append((z_j_coeff, "".join(pauli_string)))
                
                # Z_i Z_j term: coeff/4 * Z_i * Z_j
                z_ij_coeff = coeff / 4
                pauli_string = ["I"] * self.n_qubits
                pauli_string[qubit_i] = "Z"
                pauli_string[qubit_j] = "Z"
                pauli_coeffs.append((z_ij_coeff, "".join(pauli_string)))
        
        # Combine terms with same Pauli strings
        combined_coeffs = defaultdict(float)
        for coeff, pauli_str in pauli_coeffs:
            combined_coeffs[pauli_str] += coeff
        
        # Create Hamiltonian
        coeffs = []
        ops = []
        for pauli_str, coeff in combined_coeffs.items():
            if abs(coeff) > 1e-10:  # Skip negligible terms
                coeffs.append(coeff)
                # Convert string to PennyLane operators
                pauli_ops = []
                for i, pauli in enumerate(pauli_str):
                    if pauli == "I":
                        pauli_ops.append(qml.Identity(i))
                    elif pauli == "Z":
                        pauli_ops.append(qml.PauliZ(i))
                    elif pauli == "X":
                        pauli_ops.append(qml.PauliX(i))
                    elif pauli == "Y":
                        pauli_ops.append(qml.PauliY(i))
                
                if len(pauli_ops) == 1:
                    ops.append(pauli_ops[0])
                else:
                    ops.append(qml.prod(*pauli_ops))
        
        self.hamiltonian = qml.Hamiltonian(coeffs, ops)
        return coeffs, ops
    
    def qaoa_circuit(self, gamma, beta):
        """
        Construct QAOA circuit for quantum annealing.
        
        Args:
            gamma: Problem Hamiltonian parameters
            beta: Mixer Hamiltonian parameters
        """
        # Initialize in superposition
        for i in range(self.n_qubits):
            qml.Hadamard(wires=i)
        
        # Apply QAOA layers
        for layer in range(self.layers):
            # Problem Hamiltonian evolution
            qml.templates.ApproxTimeEvolution(self.hamiltonian, gamma[layer], 1)
            
            # Mixer Hamiltonian evolution (X rotations)
            for i in range(self.n_qubits):
                qml.RX(2 * beta[layer], wires=i)
    
    def qaoa_cost(self, params):
        """
        Compute the expectation value of the cost Hamiltonian.
        
        Args:
            params: QAOA parameters [gamma_1, ..., gamma_p, beta_1, ..., beta_p]
            
        Returns:
            float: Expected cost value
        """
        gamma = params[:self.layers]
        beta = params[self.layers:]
        
        @qml.qnode(self.device)
        def circuit():
            self.qaoa_circuit(gamma, beta)
            return qml.expval(self.hamiltonian)
        
        return circuit()
    
    def sample_solutions(self, params, n_samples=100):
        """
        Sample quantum solutions from the optimized QAOA circuit.
        
        Args:
            params: Optimized QAOA parameters
            n_samples: Number of solutions to sample
            
        Returns:
            list: List of sampled solutions with energies
        """
        gamma = params[:self.layers]
        beta = params[self.layers:]
        
        @qml.qnode(self.device, interface='numpy')
        def circuit():
            self.qaoa_circuit(gamma, beta)
            return qml.probs(wires=range(self.n_qubits))
        
        # Get probability distribution
        probs = circuit()
        
        solutions = []
        np.random.seed(self.seed)
        
        for _ in range(n_samples):
            # Sample from probability distribution
            outcome = np.random.choice(len(probs), p=probs)
            
            # Convert outcome to binary representation
            binary_string = format(outcome, f'0{self.n_qubits}b')
            
            # Convert to QUBO solution
            solution = {}
            energy = 0.0
            
            for i, bit_char in enumerate(binary_string):
                var = self.qubit_to_var[i]
                solution[var] = int(bit_char)
            
            # Calculate energy for this solution
            for (var1, var2), coeff in self.original_qubo.items():
                if var1 == var2:
                    energy += coeff * solution[var1]
                else:
                    energy += coeff * solution[var1] * solution[var2]
            
            solutions.append({'sample': solution, 'energy': energy})
        
        return solutions
    
    def optimize_qaoa(self, maxiter=100, method='COBYLA'):
        """
        Optimize QAOA parameters using classical optimization.
        
        Args:
            maxiter: Maximum iterations for optimization
            method: Classical optimization method
            
        Returns:
            dict: Optimization results
        """
        # Random initial parameters
        np.random.seed(self.seed)
        initial_params = np.random.uniform(0, 2*np.pi, 2*self.layers)
        
        print(f" Starting QAOA optimization with {self.layers} layers")
        print(f"  Qubits: {self.n_qubits}")
        print(f" Method: {method}")
        
        start_time = time.time()
        
        # Optimize parameters
        result = scipy.optimize.minimize(
            self.qaoa_cost,
            initial_params,
            method=method,
            options={'maxiter': maxiter}
        )
        
        optimization_time = time.time() - start_time
        
        print(f"  Optimization time: {optimization_time:.2f}s")
        print(f" Final cost: {result.fun:.6f}")
        print(f" Converged: {result.success}")
        
        return result
    
    def solve_qubo(self, qubo, maxiter=100, n_samples=100):
        """
        Main method to solve QUBO using quantum annealing simulation.
        
        Args:
            qubo: QUBO formulation dictionary
            maxiter: Maximum optimization iterations
            n_samples: Number of solutions to sample
            
        Returns:
            dict: Results containing solutions and metadata
        """
        print(f" Starting Quantum Annealer Simulation (QAOA)")
        print(f" QUBO size: {len(qubo)} terms")
        
        self.original_qubo = qubo
        
        # Convert QUBO to quantum Hamiltonian
        start_time = time.time()
        coeffs, ops = self.qubo_to_pauli(qubo)
        conversion_time = time.time() - start_time
        
        print(f"  Hamiltonian terms: {len(coeffs)}")
        print(f" Conversion time: {conversion_time:.3f}s")
        
        # Optimize QAOA parameters
        optimization_result = self.optimize_qaoa(maxiter=maxiter)
        
        # Sample solutions
        print(f" Sampling {n_samples} quantum solutions...")
        solutions = self.sample_solutions(optimization_result.x, n_samples)
        
        # Sort by energy (minimize)
        solutions.sort(key=lambda x: x['energy'])
        
        # Create sample set compatible with existing pipeline
        class QuantumSampleSet:
            def __init__(self, solutions):
                self.solutions = solutions
                self.first = solutions[0] if solutions else None
                
            def lowest(self, k=1):
                return self.solutions[:k]
        
        # Format best solution
        best_solution = solutions[0]
        best_solution['sample'] = best_solution['sample']
        best_solution['energy'] = best_solution['energy']
        
        sample_set = QuantumSampleSet(solutions)
        
        print(f" Best energy: {best_solution['energy']:.6f}")
        print(f"Energy range: [{solutions[0]['energy']:.3f}, {solutions[-1]['energy']:.3f}]")
        
        return sample_set

def create_quantum_annealer(layers=2, shots=1000, seed=None):
    """
    Factory function to create a quantum annealer simulator.
    
    Args:
        layers: Number of QAOA layers
        shots: Number of measurement shots
        seed: Random seed for reproducibility (optional)
        
    Returns:
        QuantumAnnealerSimulator: Configured quantum annealer
    """
    return QuantumAnnealerSimulator(layers=layers, shots=shots, seed=seed)

if __name__ == "__main__":
    # Test the quantum annealer with a simple QUBO
    print(" Testing Quantum Annealer Simulator")
    
    # Simple test QUBO: minimize x1 + x2 - 2*x1*x2
    test_qubo = {
        ('x1', 'x1'): 1.0,
        ('x2', 'x2'): 1.0,
        ('x1', 'x2'): -2.0
    }
    
    annealer = create_quantum_annealer(layers=2, shots=1000)
    result = annealer.solve_qubo(test_qubo, maxiter=50, n_samples=10)
    
    print(f"\nTest Results:")
    print(f"Best solution: {result.first['sample']}")
    print(f"Best energy: {result.first['energy']}")
    print(f"Expected: x1=1, x2=1, energy=-1.0") 