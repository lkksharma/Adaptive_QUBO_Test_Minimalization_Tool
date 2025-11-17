#!/usr/bin/env python3
"""
Complete Example: End-to-End Quantum Test Selection Pipeline
============================================================

This example demonstrates the complete workflow:
1. Generate synthetic coverage data
2. Run all three optimization methods
3. Compare results
4. Visualize outcomes

Run with: python examples/complete_example.py
"""

import numpy as np
import json
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from advanced_qubo_creation import create_advanced_qubo_builder
from quantum_annealer_simulator import create_quantum_annealer


def generate_realistic_coverage_data(n_tests=50, n_branches=100, seed=42):
    """
    Generate realistic synthetic coverage data with typical patterns:
    - Some branches covered by many tests (easy)
    - Some branches covered by few tests (hard)
    - Some tests covering many branches (efficient)
    - Some tests covering few branches (specific)
    """
    print("[DATA] Generating realistic coverage data...")
    np.random.seed(seed)
    
    C = np.zeros((n_tests, n_branches), dtype=int)
    
    # Pattern 1: Common branches (70% of branches covered by 30-50% of tests)
    n_common = int(0.7 * n_branches)
    for j in range(n_common):
        n_covering = np.random.randint(int(0.3 * n_tests), int(0.5 * n_tests))
        tests = np.random.choice(n_tests, n_covering, replace=False)
        C[tests, j] = 1
    
    # Pattern 2: Rare branches (20% of branches covered by 5-15% of tests)
    for j in range(n_common, int(0.9 * n_branches)):
        n_covering = np.random.randint(int(0.05 * n_tests), int(0.15 * n_tests))
        tests = np.random.choice(n_tests, n_covering, replace=False)
        C[tests, j] = 1
    
    # Pattern 3: Very rare branches (10% covered by 1-3 tests)
    for j in range(int(0.9 * n_branches), n_branches):
        n_covering = np.random.randint(1, min(4, n_tests))
        tests = np.random.choice(n_tests, n_covering, replace=False)
        C[tests, j] = 1
    
    # Ensure all tests cover at least something
    for i in range(n_tests):
        if C[i].sum() == 0:
            branches = np.random.choice(n_branches, np.random.randint(1, 10), replace=False)
            C[i, branches] = 1
    
    density = C.sum() / (n_tests * n_branches)
    print(f"   Tests: {n_tests}")
    print(f"   Branches: {n_branches}")
    print(f"   Density: {density:.1%}")
    print(f"   Avg branches per test: {C.sum(axis=1).mean():.1f}")
    print(f"   Avg tests per branch: {C.sum(axis=0).mean():.1f}")
    
    return C


def run_simulated_annealing_demo(C):
    """Run basic simulated annealing"""
    print("\n[SA] Running Simulated Annealing...")
    
    N, M = C.shape
    
    def compute_coverage(x):
        if x.sum() == 0:
            return 0.0
        return (C[x.astype(bool)].sum(axis=0) > 0).sum() / M
    
    def energy(x, alpha=0.2):
        cr = compute_coverage(x)
        return (1.0 - cr) + alpha * (x.sum() / N)
    
    # SA algorithm
    np.random.seed(42)
    x = np.zeros(N, dtype=int)
    e = energy(x)
    
    best_x, best_e = x.copy(), e
    T = 1.0
    T_end = 0.001
    steps = 5000
    cooling = (T_end / T) ** (1.0 / steps)
    
    for step in range(steps):
        # Neighbor
        y = x.copy()
        idx = np.random.randint(N)
        y[idx] = 1 - y[idx]
        
        e_new = energy(y)
        
        if e_new < e or np.random.random() < np.exp(-(e_new - e) / T):
            x, e = y, e_new
        
        if e < best_e:
            best_x, best_e = x.copy(), e
        
        T *= cooling
        
        if (step + 1) % 1000 == 0:
            print(f"   Step {step+1}/{steps}: Energy={best_e:.4f}, Coverage={compute_coverage(best_x):.1%}")
    
    result = {
        'selected_count': int(best_x.sum()),
        'coverage_ratio': compute_coverage(best_x),
        'energy': best_e,
        'selected_indices': np.where(best_x)[0].tolist()
    }
    
    print(f"[OK] SA Complete: {result['selected_count']} tests, {result['coverage_ratio']:.1%} coverage")
    return result


def run_adaptive_greedy_demo(C):
    """Run adaptive greedy algorithm"""
    print("\n[ADAPTIVE] Running Adaptive Greedy...")
    
    N, M = C.shape
    selected = np.zeros(N, dtype=bool)
    covered = np.zeros(M, dtype=bool)
    
    target = 0.8
    beta = 0.5
    
    iteration = 0
    while covered.mean() < target:
        best_gain = -1e9
        best_idx = None
        
        for i in range(N):
            if selected[i]:
                continue
            
            new_branches = np.where((C[i] > 0) & ~covered)[0]
            gain = len(new_branches) - beta
            
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        
        if best_idx is None or best_gain <= 0:
            break
        
        selected[best_idx] = True
        covered |= (C[best_idx] > 0)
        iteration += 1
        
        if iteration % 5 == 0:
            print(f"   Iteration {iteration}: {selected.sum()} tests, {covered.mean():.1%} coverage")
    
    result = {
        'selected_count': int(selected.sum()),
        'coverage_ratio': float(covered.mean()),
        'selected_indices': np.where(selected)[0].tolist()
    }
    
    print(f"[OK] Adaptive Complete: {result['selected_count']} tests, {result['coverage_ratio']:.1%} coverage")
    return result


def run_advanced_qubo_demo(C):
    """Run advanced QUBO with quantum annealing"""
    print("\n[QUANTUM]  Running Advanced QUBO (Quantum)...")
    
    N, M = C.shape
    
    # Create metadata
    test_cases = [{'id': i, 'name': f'test_{i}'} for i in range(N)]
    branches = [
        {'id': f'B{i}', 'line': i*10, 'condition': f'branch_{i}',
         'expr': f'cond_{i}', 'complexity': 'medium'}
        for i in range(M)
    ]
    
    # Initialize
    builder = create_advanced_qubo_builder(coverage_matrix=C)
    
    # Limit to top 20 tests for quantum tractability
    per_test_cov = C.sum(axis=1)
    top_idx = np.argsort(-per_test_cov)[:20]
    C_small = C[top_idx]
    test_cases_small = [test_cases[i] for i in top_idx]
    
    # Build QUBO
    qubo_result = builder.build_advanced_qubo(
        branches=branches,
        test_cases=test_cases_small,
        iteration_k=0,
        target_coverage=0.8
    )
    
    qubo = qubo_result['qubo']
    
    print(f"   QUBO size: {len(qubo)} terms")
    print(f"   Attempting quantum solution (this may take a minute)...")
    
    try:
        # Try quantum annealing
        annealer = create_quantum_annealer(layers=2, shots=1000)
        sampleset = annealer.solve_qubo(qubo, maxiter=50, n_samples=50)
        
        solution = sampleset.first['sample']
        energy = sampleset.first['energy']
        
        # Map back to original indices
        selected_small = [int(v.split('_')[1]) for v, val in solution.items() 
                         if val == 1 and v.startswith('t_')]
        selected_indices = [top_idx[i] for i in selected_small]
        
    except Exception as e:
        print(f"   [WARN]  Quantum solving failed: {e}")
        print(f"   Using greedy fallback...")
        
        # Greedy fallback
        selected = []
        covered = np.zeros(M, dtype=bool)
        while covered.mean() < 0.8:
            best = None
            best_gain = 0
            for i in range(N):
                if i in selected:
                    continue
                gain = (C[i] & ~covered).sum()
                if gain > best_gain:
                    best_gain = gain
                    best = i
            if best is None:
                break
            selected.append(best)
            covered |= (C[best] > 0)
        
        selected_indices = selected
        energy = 0.0
    
    # Calculate final coverage
    if selected_indices:
        covered = (C[selected_indices].sum(axis=0) > 0)
        coverage_ratio = covered.sum() / M
    else:
        coverage_ratio = 0.0
    
    result = {
        'selected_count': len(selected_indices),
        'coverage_ratio': coverage_ratio,
        'energy': energy,
        'selected_indices': selected_indices
    }
    
    print(f"[OK] QUBO Complete: {result['selected_count']} tests, {result['coverage_ratio']:.1%} coverage")
    return result


def compare_results(sa_result, adaptive_result, qubo_result):
    """Print comparison table"""
    print("\n" + "="*70)
    print("[DATA] RESULTS COMPARISON")
    print("="*70)
    print(f"{'Method':<30} {'Tests':<10} {'Coverage':<12} {'Energy/FV'}")
    print("-"*70)
    
    print(f"{'Simulated Annealing':<30} {sa_result['selected_count']:<10} "
          f"{sa_result['coverage_ratio']:.1%}{'':6} {sa_result['energy']:.4f}")
    
    fv_adaptive = 1.0 - adaptive_result['coverage_ratio'] + 0.5 * (
        adaptive_result['selected_count'] / 50)
    print(f"{'Adaptive Greedy':<30} {adaptive_result['selected_count']:<10} "
          f"{adaptive_result['coverage_ratio']:.1%}{'':6} {fv_adaptive:.4f}")
    
    print(f"{'Advanced QUBO (Quantum)':<30} {qubo_result['selected_count']:<10} "
          f"{qubo_result['coverage_ratio']:.1%}{'':6} {qubo_result['energy']:.4f}")
    
    print("="*70)
    
    # Find best
    best = min(
        [('SA', sa_result['selected_count']), 
         ('Adaptive', adaptive_result['selected_count']),
         ('QUBO', qubo_result['selected_count'])],
        key=lambda x: x[1]
    )
    
    print(f"\n[BEST] Most Efficient: {best[0]} with {best[1]} tests")


def save_results(sa_result, adaptive_result, qubo_result):
    """Save all results"""
    os.makedirs('results', exist_ok=True)
    
    with open('results/example_sa.json', 'w') as f:
        json.dump(sa_result, f, indent=2)
    
    with open('results/example_adaptive.json', 'w') as f:
        json.dump(adaptive_result, f, indent=2)
    
    with open('results/example_qubo.json', 'w') as f:
        json.dump(qubo_result, f, indent=2)
    
    print("\n[SAVE] Results saved to results/ directory")


def main():
    print("="*70)
    print("Quantum Test Selection Pipeline - Complete Example")
    print("="*70)
    
    # Step 1: Generate data
    C = generate_realistic_coverage_data(n_tests=50, n_branches=100)
    
    # Save coverage matrix
    os.makedirs('results', exist_ok=True)
    np.savez('results/C_example.npz', C=C)
    print("[SAVE] Coverage matrix saved to results/C_example.npz")
    
    # Step 2: Run algorithms
    sa_result = run_simulated_annealing_demo(C)
    adaptive_result = run_adaptive_greedy_demo(C)
    qubo_result = run_advanced_qubo_demo(C)
    
    # Step 3: Compare
    compare_results(sa_result, adaptive_result, qubo_result)
    
    # Step 4: Save
    save_results(sa_result, adaptive_result, qubo_result)
    
    print("\n[OK] Example complete!")
    print("\nNext steps:")
    print("  1. Check results/ directory for detailed outputs")
    print("  2. Modify parameters in this script to experiment")
    print("  3. Run on your own project: python run_full_pipeline.py")


if __name__ == '__main__':
    main()

