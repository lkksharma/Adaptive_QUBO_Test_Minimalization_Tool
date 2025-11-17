#!/usr/bin/env python3
"""
Complete Quantum Test Selection Pipeline
========================================

INPUT FORMAT REQUIREMENTS:
-------------------------
1. Coverage Matrix (required): results/C_<project>.npz
   - NumPy compressed archive containing key 'C'
   - C: Binary matrix of shape (N_tests, M_branches)
   - C[i,j] = 1 if test i covers branch j, 0 otherwise
   
   Example creation:
     import numpy as np
     C = np.random.randint(0, 2, size=(50, 100))
     np.savez('results/C_myproject.npz', C=C)

2. Branch Metadata (optional): branches.json
   - JSON array of branch dictionaries
   - Required fields: id, line, condition, expr
   - Optional fields: complexity, overlap_group, criticality
   
   Example:
     [
       {"id": "B1", "line": 10, "condition": "if x > 0", 
        "expr": "x > 0", "complexity": "low"},
       ...
     ]

3. Configuration (optional): config/baselines.json
   - JSON object with algorithm hyperparameters
   - See README.md for full specification

USAGE:
-----
  # Option 1: Specify coverage explicitly
  python run_full_pipeline.py --coverage results/C_myproject.npz
  
  # Option 2: Use environment variable
  export COVERAGE_NPZ=results/C_myproject.npz
  python run_full_pipeline.py
  
  # Option 3: Auto-detection
  python run_full_pipeline.py

  # With custom config
  python run_full_pipeline.py --config config/custom.json

OUTPUT:
------
  results/
    ├── baselines_sa_result.json
    ├── baselines_adaptive_result.json
    ├── advanced_results.json
    └── comparison.csv

PIPELINE STAGES:
---------------
  1. Load coverage matrix C (N_tests × M_branches)
  2. Run Basic Simulated Annealing baseline
  3. Run Basic Adaptive Greedy baseline
  4. Run Advanced 5-Stage QUBO with quantum annealing
  5. Generate comparison CSV with all results
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import numpy as np
from typing import Dict, List


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run complete quantum test selection pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--coverage', 
        type=str,
        help='Path to coverage matrix .npz file'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/baselines.json',
        help='Path to configuration JSON (default: config/baselines.json)'
    )
    parser.add_argument(
        '--target-coverage',
        type=float,
        default=0.8,
        help='Target branch coverage (default: 0.8)'
    )
    parser.add_argument(
        '--max-iterations',
        type=int,
        default=10,
        help='Maximum QUBO iterations (default: 10)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Output directory (default: results)'
    )
    return parser.parse_args()


def find_coverage_matrix(explicit_path: str = None) -> str:
    """Find coverage matrix, checking explicit path first, then standard locations"""
    if explicit_path and os.path.isfile(explicit_path):
        return explicit_path
    
    env_path = os.environ.get('COVERAGE_NPZ')
    if env_path and os.path.isfile(env_path):
        return env_path
    
    # Check standard locations
    standard_paths = [
        'results/C_cerberus_ast.npz',
        'results/C_tcas_synth.npz',
        'results/C_tcas_gcov.npz',
        'results/C_project.npz',
    ]
    
    for path in standard_paths:
        if os.path.isfile(path):
            return path
    
    raise FileNotFoundError(
        'No coverage matrix found. Please:\n'
        '  1. Create coverage matrix: np.savez("results/C_project.npz", C=your_matrix)\n'
        '  2. Or specify path: --coverage results/C_project.npz\n'
        '  3. Or set environment: export COVERAGE_NPZ=results/C_project.npz'
    )


def load_coverage_info(coverage_path: str) -> Dict:
    """Load and validate coverage matrix"""
    print(f'\n[DATA] Loading coverage matrix from: {coverage_path}')
    data = np.load(coverage_path)
    C = data['C']
    
    # Validate dimensions
    if C.ndim != 2:
        raise ValueError(f'Coverage matrix must be 2D, got shape {C.shape}')
    
    N, M = C.shape
    
    if N == 0 or M == 0:
        raise ValueError(f'Coverage matrix has invalid dimensions: ({N}, {M})')
    
    # Validate binary values
    if not np.all((C == 0) | (C == 1)):
        print('[WARN] Coverage matrix contains non-binary values, converting to binary')
        C = (C > 0).astype(int)
    
    # Check for empty rows/columns
    empty_tests = np.where(C.sum(axis=1) == 0)[0]
    empty_branches = np.where(C.sum(axis=0) == 0)[0]
    
    if len(empty_tests) > 0:
        print(f'[WARN] {len(empty_tests)} tests cover no branches')
    if len(empty_branches) > 0:
        print(f'[WARN] {len(empty_branches)} branches covered by no tests')
    
    density = (C.sum() / (N * M * 1.0)) if N * M > 0 else 0
    
    print(f'   Tests: {N}')
    print(f'   Branches: {M}')
    print(f'   Density: {density:.1%}')
    print(f'   Total coverage entries: {int(C.sum())}')
    
    return {
        'path': coverage_path,
        'matrix': C,
        'n_tests': N,
        'n_branches': M,
        'density': density
    }


def run_baseline(script_name: str, coverage_path: str, results_dir: str) -> Dict:
    """Run a baseline method and return its results"""
    method_name = script_name.replace('basic_', '').replace('.py', '').replace('_', ' ').title()
    print(f'\n[RUN] Running {method_name}...')
    
    env = os.environ.copy()
    env['COVERAGE_NPZ'] = coverage_path
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=True,
            text=True,
            env=env,
            check=True
        )
        print(result.stdout)
        
        # Load result
        if 'simulated' in script_name:
            result_file = os.path.join(results_dir, 'baselines_sa_result.json')
        else:
            result_file = os.path.join(results_dir, 'baselines_adaptive_result.json')
        
        with open(result_file, 'r') as f:
            return json.load(f)
            
    except subprocess.CalledProcessError as e:
        print(f'[ERROR] Error running {method_name}:')
        print(e.stderr)
        raise


def load_pipeline_config(config_path: str = 'config/baselines.json') -> Dict:
    """Load pipeline configuration with defaults"""
    defaults = {
        'quantum_batch_size': 20,
        'qaoa_maxiter': 50,
        'qaoa_samples': 50,
        'qaoa_layers': 2,
        'qaoa_shots': 1000,
        'seed': 42,
        'complexity_multipliers': {
            'isinstance': 0.5,
            'schema': 0.3,
            'error': 0.2
        }
    }
    
    if os.path.isfile(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
            defaults.update(config)
    
    return defaults


def run_advanced_qubo(coverage_path: str, target_coverage: float, 
                     max_iterations: int, results_dir: str) -> Dict:
    """Run advanced QUBO algorithm"""
    print(f'\n[QUANTUM]  Running Advanced QUBO (Quantum)...')
    print(f'   Target Coverage: {target_coverage:.0%}')
    print(f'   Max Iterations: {max_iterations}')
    
    from advanced_qubo_creation import create_advanced_qubo_builder
    from quantum_annealer_simulator import create_quantum_annealer
    
    # Load configuration
    config = load_pipeline_config()
    
    # Load coverage
    C = np.load(coverage_path)['C']
    N, M = C.shape
    
    # Get configurable parameters
    batch_size = int(config.get('quantum_batch_size', 20))
    qaoa_maxiter = int(config.get('qaoa_maxiter', 50))
    qaoa_samples = int(config.get('qaoa_samples', 50))
    qaoa_layers = int(config.get('qaoa_layers', 2))
    qaoa_shots = int(config.get('qaoa_shots', 1000))
    
    print(f'   Quantum Batch Size: {batch_size}')
    print(f'   QAOA Layers: {qaoa_layers}')
    print(f'   QAOA Max Iterations: {qaoa_maxiter}')
    
    # Warn if test suite is large
    if N > batch_size:
        print(f'[WARN] Test suite ({N} tests) exceeds batch size ({batch_size})')
        print(f'       Only top {batch_size} tests by coverage will be processed')
        print(f'       Increase "quantum_batch_size" in config for full processing')
    
    # Create test cases and branches
    test_cases = [{'id': i, 'name': f'test_{i}'} for i in range(N)]
    branches = [
        {'id': f'B{i}', 'line': i * 10, 'condition': f'branch_{i}',
         'expr': f'condition_{i}', 'complexity': 'medium'}
        for i in range(M)
    ]
    
    # Initialize builder and quantum annealer
    builder = create_advanced_qubo_builder(coverage_matrix=C, config=config)
    quantum_annealer = create_quantum_annealer(
        layers=qaoa_layers, 
        shots=qaoa_shots, 
        seed=config.get('seed', 42)
    )
    
    all_results = []
    previous_solution = None
    
    for iteration in range(max_iterations):
        print(f'\n--- Iteration {iteration + 1}/{max_iterations} ---')
        
        # Build QUBO
        qubo_result = builder.build_advanced_qubo(
            branches=branches,
            test_cases=test_cases,
            iteration_k=iteration,
            previous_solution=previous_solution,
            target_coverage=target_coverage
        )
        
        qubo = qubo_result['qubo']
        
        # Solve with quantum annealer (limit to manageable size)
        K = min(batch_size, N)  # Use configurable batch size
        per_test_cov = C.sum(axis=1)
        top_idx = np.argsort(-per_test_cov)[:K]
        
        # Warn if QUBO is very large
        if len(qubo) > 500:
            print(f'[WARN] Large QUBO detected ({len(qubo)} terms)')
            print(f'       This may take significant time to solve')
        valid_tests = {f't_{i}' for i in top_idx}

        # Filter QUBO to top-K tests
        qubo_filtered = {
            k: v for k, v in qubo.items()
            if k[0] in valid_tests and k[1] in valid_tests
        }
        
        try:
            sampleset = quantum_annealer.solve_qubo(
                qubo_filtered, 
                maxiter=qaoa_maxiter, 
                n_samples=qaoa_samples
            )
            solution = sampleset.first['sample']
            energy = sampleset.first['energy']
            
            # Extract selected tests
            selected_indices = [
                int(var.split('_')[1]) for var, val in solution.items()
                if val == 1 and var.startswith('t_')
            ]
            
            # Greedy post-fill to reach target coverage
            covered = (C[selected_indices].sum(axis=0) > 0).sum() if selected_indices else 0
            cr = float(covered) / M if M > 0 else 0.0
            
            if cr < target_coverage:
                uncovered = set(np.where((C[selected_indices].sum(axis=0) == 0))[0]) if selected_indices else set(range(M))
                remaining = [i for i in range(N) if i not in selected_indices]
                
                for test_idx in sorted(remaining, key=lambda i: -C[i].sum()):
                    if cr >= target_coverage:
                        break
                    test_covers = set(np.where(C[test_idx] > 0)[0])
                    if uncovered.intersection(test_covers):
                        selected_indices.append(test_idx)
                        covered = (C[selected_indices].sum(axis=0) > 0).sum()
                        cr = float(covered) / M if M > 0 else 0.0
            
            previous_solution = [test_cases[i] for i in selected_indices]
            
            result = {
                'iteration': iteration + 1,
                'selected_indices': selected_indices,
                'coverage_ratio': cr,
                'energy': energy,
                'selected_count': len(selected_indices),
                'method': 'Advanced_QAOA'
            }
            
            all_results.append(result)
            builder.pipeline_state['score_current'] = cr
            
            print(f'   Selected: {len(selected_indices)} tests')
            print(f'   Coverage: {cr:.1%}')
            print(f'   Energy: {energy:.3f}')
            
            if cr >= target_coverage:
                print(f'✓ Target coverage {target_coverage:.0%} achieved!')
                break
                
        except (RuntimeError, ValueError, KeyError) as e:
            print(f'[WARN]  Quantum solving failed: {e}')
            print('   Falling back to greedy selection...')
            # Simple greedy fallback
            selected_indices = []
            covered = np.zeros(M, dtype=bool)
            while covered.mean() < target_coverage and len(selected_indices) < N:
                best_gain = -1
                best_idx = None
                for i in range(N):
                    if i in selected_indices:
                        continue
                    gain = (C[i] & ~covered).sum()
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = i
                if best_idx is None:
                    break
                selected_indices.append(best_idx)
                covered |= (C[best_idx] > 0)
            
            cr = covered.mean()
            result = {
                'iteration': iteration + 1,
                'selected_indices': selected_indices,
                'coverage_ratio': float(cr),
                'energy': 0.0,
                'selected_count': len(selected_indices),
                'method': 'Greedy_Fallback'
            }
            all_results.append(result)
            break
    
    # Save results
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, 'advanced_results.json')
    with open(output_path, 'w') as f:
        json.dump({
            'iterations': all_results,
            'best_result': min(all_results, key=lambda r: r.get('energy', float('inf')))
        }, f, indent=2)
    
    print(f'\n[OK] Advanced QUBO complete → {output_path}')
    return all_results[-1] if all_results else {}


def generate_comparison_csv(sa_result: Dict, adaptive_result: Dict, 
                           advanced_result: Dict, output_dir: str):
    """Generate comparison CSV with all results"""
    print(f'\n[CSV] Generating comparison CSV...')
    
    rows = []
    
    # SA result
    rows.append([
        'Basic_Simulated_Annealing',
        f"{sa_result.get('energy', 'NA')}",
        f"{sa_result.get('coverage_ratio', 'NA')}",
        sa_result.get('selected_count', 'NA')
    ])
    
    # Adaptive result
    cr = float(adaptive_result.get('coverage_ratio', 0.0))
    sel = int(adaptive_result.get('selected_count', 0))
    N = int(adaptive_result.get('total_tests', max(sel, 1)))
    beta = float(adaptive_result.get('config', {}).get('beta_size_penalty', 0.5))
    fv = (1.0 - cr) + beta * (sel / max(N, 1))
    
    rows.append([
        'Basic_Adaptive_Greedy',
        f"{fv}",
        f"{cr}",
        sel
    ])
    
    # Advanced result
    rows.append([
        advanced_result.get('method', 'Advanced_QAOA'),
        f"{advanced_result.get('energy', 'NA')}",
        f"{advanced_result.get('coverage_ratio', 'NA')}",
        advanced_result.get('selected_count', 'NA')
    ])
    
    # Write CSV
    csv_path = os.path.join(output_dir, 'comparison.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Method', 'FV', 'Coverage', 'Tests'])
        writer.writerows(rows)
    
    print(f'[OK] Comparison saved → {csv_path}')
    print('\n[DATA] RESULTS SUMMARY:')
    print('=' * 70)
    print(f'{"Method":<30} {"FV":<15} {"Coverage":<12} {"Tests"}')
    print('-' * 70)
    for row in rows:
        print(f'{row[0]:<30} {str(row[1]):<15} {str(row[2]):<12} {row[3]}')
    print('=' * 70)


def main():
    args = parse_args()
    
    print('=' * 70)
    print('Quantum Test Selection Pipeline')
    print('=' * 70)
    
    # Find and load coverage
    coverage_path = find_coverage_matrix(args.coverage)
    coverage_info = load_coverage_info(coverage_path)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run baselines
    sa_result = run_baseline('basic_simulated_annealing.py', coverage_path, args.output_dir)
    adaptive_result = run_baseline('basic_adaptive.py', coverage_path, args.output_dir)
    
    # Run advanced QUBO
    advanced_result = run_advanced_qubo(
        coverage_path,
        args.target_coverage,
        args.max_iterations,
        args.output_dir
    )
    
    # Generate comparison
    generate_comparison_csv(sa_result, adaptive_result, advanced_result, args.output_dir)
    
    print('\n[OK] Pipeline complete!')
    print(f'   All results saved to: {args.output_dir}/')


if __name__ == '__main__':
    main()

