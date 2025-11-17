#!/usr/bin/env python3
"""
Generic Coverage Builder Template
=================================

INPUT FORMAT REQUIREMENTS FOR YOUR SOFTWARE:
-------------------------------------------
This is a TEMPLATE script. Adapt it to your specific project by implementing
the coverage collection logic for your test suite.

REQUIRED OUTPUT FORMAT:
-----------------------
Create a binary coverage matrix C with:
  - Shape: (N_tests, M_branches)
  - Type: numpy.ndarray with dtype int (0 or 1)
  - C[i, j] = 1 if test i covers branch j
  - C[i, j] = 0 otherwise

Save as:
  np.savez('results/C_<your_project>.npz', C=C)

IMPLEMENTATION OPTIONS:
----------------------

Option 1: Python with coverage.py
----------------------------------
import coverage
import pytest
import numpy as np

# 1. Initialize coverage
cov = coverage.Coverage(branch=True)
cov.start()

# 2. Run your tests
pytest.main(['-v', 'tests/'])

# 3. Stop and analyze
cov.stop()
cov.save()

# 4. Extract branch data
coverage_data = cov.get_data()
# ... process into matrix C ...

# 5. Save
np.savez('results/C_myproject.npz', C=C)


Option 2: C/C++ with gcov
--------------------------
# Compile with coverage flags
gcc -fprofile-arcs -ftest-coverage myprogram.c

# Run each test
./myprogram < test_input_1.txt
gcov myprogram.c
# Parse myprogram.c.gcov for branch coverage

# ... repeat for all tests ...
# Build matrix C from collected data
# Save with NumPy

Option 3: Java with JaCoCo
---------------------------
# Run tests with JaCoCo agent
# Parse XML reports to extract branch coverage
# Convert to NumPy matrix format

Option 4: Manual/Synthetic
---------------------------
# For demonstration or if you have pre-collected data
import numpy as np

N_tests = 50
M_branches = 100
C = np.random.randint(0, 2, size=(N_tests, M_branches))
np.savez('results/C_demo.npz', C=C)

EXAMPLE IMPLEMENTATION (Python + pytest):
-----------------------------------------
"""

import os
import sys
import subprocess
import json
import numpy as np
from typing import List, Dict, Set


def build_coverage_matrix_pytest_example(
    test_directory: str,
    source_directory: str,
    output_name: str = 'C_project'
) -> np.ndarray:
    """
    Example implementation using pytest and coverage.py
    
    Args:
        test_directory: Directory containing test files
        source_directory: Directory containing source code
        output_name: Name for output file (without extension)
        
    Returns:
        Coverage matrix C of shape (N_tests, M_branches)
    """
    
    print(f'Building coverage matrix...')
    print(f'  Tests: {test_directory}')
    print(f'  Source: {source_directory}')
    
    try:
        import coverage
        import pytest
    except ImportError:
        print('[ERROR] Required packages not installed. Run:')
        print('   pip install pytest coverage')
        sys.exit(1)
    
    # Step 1: Collect test names
    result = subprocess.run(
        [sys.executable, '-m', 'pytest', '--collect-only', '-q', test_directory],
        capture_output=True,
        text=True
    )
    
    # Parse test names (simplified - adapt to your output format)
    test_names = []
    for line in result.stdout.split('\n'):
        if '::test_' in line:
            test_names.append(line.strip())
    
    print(f'  Found {len(test_names)} tests')
    
    # Step 2: Run each test individually and collect coverage
    branch_coverage_per_test = []
    all_branches = set()
    
    for i, test_name in enumerate(test_names):
        print(f'  Running test {i+1}/{len(test_names)}...', end='\r')
        
        # Run with coverage
        cov = coverage.Coverage(branch=True, source=[source_directory])
        cov.start()
        
        # Run single test
        pytest.main(['-x', '--tb=no', '-q', test_name])
        
        cov.stop()
        
        # Extract branch coverage
        branches_covered = set()
        data = cov.get_data()
        
        for filename in data.measured_files():
            try:
                # Get branch coverage for this file
                # Note: Using private API (_analyze) as public API doesn't expose branch arcs
                # This may need updates if coverage.py changes its internal structure
                analysis = cov._analyze(filename)
                
                # Check for branch coverage support
                if hasattr(analysis, 'executed_branch_arcs'):
                    # Branch format: (start_line, end_line)
                    executed_branches = analysis.executed_branch_arcs()
                    for arc in executed_branches:
                        branch_id = f'{filename}:{arc[0]}->{arc[1]}'
                        branches_covered.add(branch_id)
                        all_branches.add(branch_id)
                elif hasattr(analysis, 'missing_branch_arcs'):
                    # Fallback: compute executed from total - missing
                    # This is for compatibility with different coverage.py versions
                    executed_branches = analysis.executed_branch_arcs()
                    for arc in executed_branches:
                        branch_id = f'{filename}:{arc[0]}->{arc[1]}'
                        branches_covered.add(branch_id)
                        all_branches.add(branch_id)
            except (AttributeError, TypeError) as e:
                print(f'[WARN] Could not extract branch coverage from {filename}: {e}')
                print('       coverage.py API may have changed')
                continue
        
        branch_coverage_per_test.append(branches_covered)
    
    print(f'  Collected coverage from {len(test_names)} tests')
    print(f'  Total unique branches: {len(all_branches)}')
    
    # Step 3: Convert to matrix format
    branch_list = sorted(all_branches)
    branch_to_idx = {b: i for i, b in enumerate(branch_list)}
    
    N = len(test_names)
    M = len(branch_list)
    
    # Validate dimensions
    if N == 0:
        raise ValueError('No tests found. Check test directory path.')
    if M == 0:
        raise ValueError('No branches found. Check if branch coverage is enabled.')
    
    C = np.zeros((N, M), dtype=int)
    
    for i, branches in enumerate(branch_coverage_per_test):
        for branch in branches:
            j = branch_to_idx[branch]
            C[i, j] = 1
    
    # Step 4: Save
    os.makedirs('results', exist_ok=True)
    output_path = f'results/{output_name}.npz'
    np.savez(output_path, C=C)
    
    print(f'[OK] Coverage matrix saved: {output_path}')
    print(f'   Shape: {C.shape}')
    print(f'   Density: {C.sum() / (N * M):.1%}')
    
    # Optional: Save branch metadata
    branch_metadata_path = f'results/{output_name}_branches.json'
    with open(branch_metadata_path, 'w') as f:
        json.dump({
            'branches': [{'id': f'B{i}', 'location': b} for i, b in enumerate(branch_list)],
            'n_tests': N,
            'n_branches': M
        }, f, indent=2)
    
    return C


def build_coverage_matrix_synthetic(
    n_tests: int = 50,
    n_branches: int = 100,
    coverage_density: float = 0.3,
    output_name: str = 'C_synthetic'
) -> np.ndarray:
    """
    Build synthetic coverage matrix for testing/demo purposes
    
    Args:
        n_tests: Number of test cases
        n_branches: Number of branches
        coverage_density: Proportion of 1s in matrix (0.0 - 1.0)
        output_name: Name for output file
        
    Returns:
        Synthetic coverage matrix
    """
    print(f'Building synthetic coverage matrix...')
    print(f'  Tests: {n_tests}')
    print(f'  Branches: {n_branches}')
    print(f'  Target density: {coverage_density:.1%}')
    
    np.random.seed(42)
    C = (np.random.random((n_tests, n_branches)) < coverage_density).astype(int)
    
    # Ensure all branches are covered by at least one test
    for j in range(n_branches):
        if C[:, j].sum() == 0:
            i = np.random.randint(0, n_tests)
            C[i, j] = 1
    
    # Ensure all tests cover at least one branch
    for i in range(n_tests):
        if C[i, :].sum() == 0:
            j = np.random.randint(0, n_branches)
            C[i, j] = 1
    
    actual_density = C.sum() / (n_tests * n_branches)
    
    # Save
    os.makedirs('results', exist_ok=True)
    output_path = f'results/{output_name}.npz'
    np.savez(output_path, C=C)
    
    print(f'[OK] Synthetic matrix saved: {output_path}')
    print(f'   Actual density: {actual_density:.1%}')
    
    return C


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Build coverage matrix for test suite')
    parser.add_argument('--mode', choices=['pytest', 'synthetic'], default='synthetic',
                       help='Coverage collection mode')
    parser.add_argument('--test-dir', default='tests/', help='Test directory (pytest mode)')
    parser.add_argument('--source-dir', default='src/', help='Source directory (pytest mode)')
    parser.add_argument('--n-tests', type=int, default=50, help='Number of tests (synthetic mode)')
    parser.add_argument('--n-branches', type=int, default=100, help='Number of branches (synthetic mode)')
    parser.add_argument('--density', type=float, default=0.3, help='Coverage density (synthetic mode)')
    parser.add_argument('--output', default='C_project', help='Output name (without extension)')
    
    args = parser.parse_args()
    
    if args.mode == 'pytest':
        build_coverage_matrix_pytest_example(
            args.test_dir,
            args.source_dir,
            args.output
        )
    else:
        build_coverage_matrix_synthetic(
            args.n_tests,
            args.n_branches,
            args.density,
            args.output
        )
    
    print('\n[OK] Coverage matrix ready!')
    print(f'   Next step: python run_full_pipeline.py --coverage results/{args.output}.npz')

