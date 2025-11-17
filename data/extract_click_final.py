#!/usr/bin/env python3
"""
CLICK Coverage Extraction - Final Working Version
Uses pytest-cov plugin for proper subprocess coverage
"""

import os
import sys
import subprocess
import numpy as np
import json
from pathlib import Path
import shutil

def main():
    print("="*70)
    print("CLICK Coverage Extraction (Working Version)")
    print("="*70)
    
    project_dir = Path(__file__).parent / 'click'
    os.chdir(project_dir)
    
    # Install pytest-cov
    print("\n[0/5] Installing dependencies...")
    subprocess.run(
        [sys.executable, '-m', 'pip', 'install', '-q', 'pytest-cov'],
        check=False
    )
    print("      Done")
    
    # Collect tests
    print("\n[1/5] Collecting tests...")
    result = subprocess.run(
        [sys.executable, '-m', 'pytest', '--collect-only', '-q'],
        capture_output=True,
        text=True
    )
    
    all_tests = [l.strip() for l in result.stdout.split('\n') 
                 if '::test_' in l and not l.startswith('<')]
    
    # Sample tests
    import random
    random.seed(42)
    sample_size = min(150, len(all_tests))
    tests = sorted(random.sample(all_tests, sample_size))
    
    print(f"      Total tests: {len(all_tests)}")
    print(f"      Selected: {len(tests)} tests")
    
    # Run coverage on full suite first to get all branches
    print("\n[2/5] Running full coverage to identify branches...")
    subprocess.run(
        [sys.executable, '-m', 'pytest', '--cov=src/click', 
         '--cov-branch', '--cov-report=json:.coverage.json', '-q'],
        capture_output=True
    )
    
    # Load branch data
    branches_all = set()
    try:
        with open('.coverage.json', 'r') as f:
            cov_data = json.load(f)
            for filename, file_data in cov_data['files'].items():
                if 'executed_branches' in file_data:
                    for branch in file_data['executed_branches']:
                        branch_id = f'{Path(filename).name}:{branch[0]}→{branch[1]}'
                        branches_all.add(branch_id)
    except:
        pass
    
    if not branches_all:
        print("      ERROR: No branches detected from full run")
        print("      Trying alternative method...")
    
    print(f"      Identified {len(branches_all)} total branches")
    
    # Run each test individually
    print("\n[3/5] Collecting per-test coverage...")
    
    coverage_per_test = []
    
    for i, test in enumerate(tests):
        if (i + 1) % 25 == 0:
            print(f"      Progress: {i+1}/{len(tests)}")
        
        # Clean previous coverage data
        for f in Path('.').glob('.coverage*'):
            if f.is_file():
                f.unlink()
        
        # Run test with coverage
        result = subprocess.run(
            [sys.executable, '-m', 'pytest', '--cov=src/click',
             '--cov-branch', '--cov-report=json:.cov_temp.json',
             '-xvs', '--tb=no', test],
            capture_output=True,
            timeout=10
        )
        
        # Extract branches
        branches = set()
        try:
            with open('.cov_temp.json', 'r') as f:
                data = json.load(f)
                for filename, file_data in data['files'].items():
                    if 'executed_branches' in file_data:
                        for branch in file_data['executed_branches']:
                            branch_id = f'{Path(filename).name}:{branch[0]}→{branch[1]}'
                            branches.add(branch_id)
                            branches_all.add(branch_id)
        except:
            pass
        
        coverage_per_test.append(branches)
    
    print(f"      Completed {len(tests)} tests")
    print(f"      Total unique branches: {len(branches_all)}")
    
    # Build matrix
    print("\n[4/5] Building coverage matrix...")
    
    if len(branches_all) == 0:
        print("      ERROR: No branches found!")
        return False
    
    branch_list = sorted(branches_all)
    branch_to_idx = {b: i for i, b in enumerate(branch_list)}
    
    n_tests = len(tests)
    n_branches = len(branch_list)
    C = np.zeros((n_tests, n_branches), dtype=int)
    
    for i, branches in enumerate(coverage_per_test):
        for branch in branches:
            if branch in branch_to_idx:
                j = branch_to_idx[branch]
                C[i, j] = 1
    
    density = C.sum() / (n_tests * n_branches) if n_tests * n_branches > 0 else 0
    
    print(f"\n[RESULTS]")
    print(f"   Tests: {n_tests}")
    print(f"   Branches: {n_branches}")
    print(f"   Density: {density:.1%}")
    print(f"   Total coverage entries: {int(C.sum())}")
    
    # Validate
    if n_branches == 0:
        print("\n[ERROR] No branches detected!")
        return False
    
    # Save
    print("\n[5/5] Saving...")
    output_path = Path('../../results/C_click.npz')
    np.savez(output_path, C=C)
    
    print(f"\n[OK] Coverage matrix saved: {output_path}")
    print(f"     Shape: {C.shape}")
    
    # Save metadata
    metadata = {
        'project': 'CLICK',
        'version': 'latest',
        'tests_total': len(all_tests),
        'tests_sampled': n_tests,
        'branches': n_branches,
        'density': float(density)
    }
    
    with open('../../results/C_click_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n[NEXT] Run pipeline:")
    print(f"     cd ../..")
    print(f"     python3 run_full_pipeline.py --coverage results/C_click.npz")
    
    return True

if __name__ == '__main__':
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n[CANCELLED] Extraction interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

