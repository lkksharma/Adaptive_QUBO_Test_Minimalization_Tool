#!/usr/bin/env python3
"""
Extract branch coverage from CLICK project
Generates coverage matrix for quantum test selection pipeline
"""

import os
import sys
import subprocess
import numpy as np
import json
from pathlib import Path

def get_test_list(project_dir, sample_size=200):
    """Collect all test cases from CLICK"""
    print("[1/4] Discovering test cases...")
    os.chdir(project_dir)
    
    result = subprocess.run(
        [sys.executable, '-m', 'pytest', '--collect-only', '-q', 'tests/'],
        capture_output=True,
        text=True
    )
    
    tests = []
    for line in result.stdout.split('\n'):
        line = line.strip()
        if '<Function ' in line or '<Method ' in line:
            # Extract test from pytest output
            continue
        elif '::test_' in line or 'test_' in line:
            # Extract test ID from nodeid format
            if '::' in line and not line.startswith('<'):
                tests.append(line)
    
    print(f"      Found {len(tests)} total test cases")
    
    # Sample tests for faster processing
    if len(tests) > sample_size:
        import random
        random.seed(42)
        tests = random.sample(tests, sample_size)
        print(f"      Sampling {sample_size} tests for analysis")
    
    return tests

def run_coverage_per_test(test_id, src_dir):
    """Run a single test and collect branch coverage"""
    try:
        # Import coverage here to use it
        import coverage
        
        # Create coverage object
        cov = coverage.Coverage(
            branch=True,
            source=[src_dir],
            omit=['*/tests/*', '*/test_*', '*/__pycache__/*']
        )
        
        cov.start()
        
        # Run the test
        result = subprocess.run(
            [sys.executable, '-m', 'pytest', '-xvs', test_id],
            capture_output=True,
            timeout=30
        )
        
        cov.stop()
        
        # Extract branch coverage
        branches_covered = set()
        data = cov.get_data()
        
        for filename in data.measured_files():
            try:
                analysis = cov._analyze(filename)
                if hasattr(analysis, 'executed_branch_arcs'):
                    arcs = analysis.executed_branch_arcs()
                    for arc in arcs:
                        branch_id = f'{filename}:{arc[0]}->{arc[1]}'
                        branches_covered.add(branch_id)
            except:
                pass
        
        return branches_covered
        
    except Exception as e:
        print(f"        [WARN] Test {test_id} failed: {e}")
        return set()

def build_coverage_matrix(project_dir):
    """Build coverage matrix for CLICK"""
    print("="*70)
    print("CLICK Coverage Extraction")
    print("="*70)
    print()
    
    # Get test list
    tests = get_test_list(project_dir)
    
    if not tests:
        print("ERROR: No tests found!")
        return None
    
    # Determine source directory
    src_dir = os.path.join(project_dir, 'src')
    if not os.path.exists(src_dir):
        src_dir = os.path.join(project_dir, 'click')
    
    print(f"\n[2/4] Installing CLICK and dependencies...")
    subprocess.run(
        [sys.executable, '-m', 'pip', 'install', '-q', '-e', '.'],
        cwd=project_dir,
        capture_output=True
    )
    print("      Installation complete")
    
    print(f"\n[3/4] Running tests and collecting coverage...")
    print(f"      Processing {len(tests)} tests...")
    
    all_branches = set()
    coverage_per_test = []
    
    for i, test in enumerate(tests):
        if (i + 1) % 10 == 0:
            print(f"      Progress: {i+1}/{len(tests)} tests")
        
        branches = run_coverage_per_test(test, src_dir)
        coverage_per_test.append(branches)
        all_branches.update(branches)
    
    print(f"      Completed {len(tests)}/{len(tests)} tests")
    print(f"      Total unique branches: {len(all_branches)}")
    
    if len(all_branches) == 0:
        print("ERROR: No branches detected!")
        return None
    
    print(f"\n[4/4] Building coverage matrix...")
    
    # Create mapping
    branch_list = sorted(all_branches)
    branch_to_idx = {b: i for i, b in enumerate(branch_list)}
    
    # Build matrix
    n_tests = len(tests)
    n_branches = len(branch_list)
    C = np.zeros((n_tests, n_branches), dtype=int)
    
    for i, branches in enumerate(coverage_per_test):
        for branch in branches:
            if branch in branch_to_idx:
                j = branch_to_idx[branch]
                C[i, j] = 1
    
    # Statistics
    density = C.sum() / (n_tests * n_branches) if n_tests * n_branches > 0 else 0
    
    print(f"\n[RESULTS]")
    print(f"   Tests: {n_tests}")
    print(f"   Branches: {n_branches}")
    print(f"   Density: {density:.1%}")
    print(f"   Total coverage entries: {int(C.sum())}")
    
    return C

if __name__ == '__main__':
    project_dir = Path(__file__).parent / 'click'
    
    C = build_coverage_matrix(str(project_dir))
    
    if C is not None:
        # Save
        output_path = Path('../results/C_click.npz')
        np.savez(output_path, C=C)
        
        print(f"\n[OK] Coverage matrix saved: {output_path}")
        print(f"     Shape: {C.shape}")
        print(f"\n[NEXT] Run pipeline:")
        print(f"     python3 run_full_pipeline.py --coverage results/C_click.npz")
    else:
        print("\n[ERROR] Failed to generate coverage matrix")
        sys.exit(1)

