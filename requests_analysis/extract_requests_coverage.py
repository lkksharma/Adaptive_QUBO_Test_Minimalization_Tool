#!/usr/bin/env python3
"""
REQUESTS Library Coverage Extraction
Generates coverage matrix for quantum test selection pipeline
"""

import os
import sys
import subprocess
import numpy as np
import json
from pathlib import Path
import random

def main():
    print("="*70)
    print("REQUESTS Library Coverage Extraction")
    print("="*70)
    
    project_dir = Path(__file__).parent / 'requests'
    
    if not project_dir.exists():
        print(f"\nERROR: {project_dir} not found!")
        print("Please run: git clone https://github.com/psf/requests.git")
        return False
    
    os.chdir(project_dir)
    
    # Install dependencies
    print("\n[1/6] Installing REQUESTS and dependencies...")
    subprocess.run(
        [sys.executable, '-m', 'pip', 'install', '-q', '-e', '.'],
        check=False
    )
    subprocess.run(
        [sys.executable, '-m', 'pip', 'install', '-q', 'pytest', 'pytest-cov', 'coverage'],
        check=False
    )
    print("      Installation complete")
    
    # Collect tests
    print("\n[2/6] Discovering test cases...")
    result = subprocess.run(
        [sys.executable, '-m', 'pytest', '--collect-only', '-q', 'tests/'],
        capture_output=True,
        text=True
    )
    
    all_tests = [l.strip() for l in result.stdout.split('\n') 
                 if '::test_' in l and not l.startswith('<')]
    
    print(f"      Found {len(all_tests)} total tests")
    
    # Sample tests for manageable processing
    random.seed(42)
    sample_size = min(200, len(all_tests))
    tests = sorted(random.sample(all_tests, sample_size))
    
    print(f"      Selected {len(tests)} tests for analysis")
    
    # Run full coverage to identify all branches
    print("\n[3/6] Running full test suite to identify branches...")
    subprocess.run(
        [sys.executable, '-m', 'pytest', 'tests/', 
         '--cov=src/requests', '--cov-branch',
         '--cov-report=json:.coverage_full.json', '-q'],
        capture_output=True,
        timeout=300
    )
    
    # Extract all branches
    branches_all = set()
    try:
        with open('.coverage_full.json', 'r') as f:
            cov_data = json.load(f)
            for filename, file_data in cov_data.get('files', {}).items():
                if 'executed_branches' in file_data:
                    for branch in file_data['executed_branches']:
                        branch_id = f'{Path(filename).name}:{branch[0]}→{branch[1]}'
                        branches_all.add(branch_id)
    except Exception as e:
        print(f"      Warning: {e}")
    
    print(f"      Identified {len(branches_all)} total branches")
    
    # Run per-test coverage
    print("\n[4/6] Collecting per-test coverage...")
    coverage_per_test = []
    
    for i, test in enumerate(tests):
        if (i + 1) % 25 == 0:
            print(f"      Progress: {i+1}/{len(tests)}")
        
        # Clean previous coverage
        for f in Path('.').glob('.coverage*'):
            if f.is_file() and f.name != '.coverage_full.json':
                try:
                    f.unlink()
                except:
                    pass
        
        # Run test with coverage
        try:
            subprocess.run(
                [sys.executable, '-m', 'pytest', 
                 '--cov=src/requests', '--cov-branch',
                 '--cov-report=json:.cov_temp.json',
                 '-xvs', '--tb=no', test],
                capture_output=True,
                timeout=15
            )
        except subprocess.TimeoutExpired:
            coverage_per_test.append(set())
            continue
        
        # Extract branches for this test
        branches = set()
        try:
            with open('.cov_temp.json', 'r') as f:
                data = json.load(f)
                for filename, file_data in data.get('files', {}).items():
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
    print("\n[5/6] Building coverage matrix...")
    
    if len(branches_all) == 0:
        print("      ERROR: No branches detected!")
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
    
    # Save
    print("\n[6/6] Saving results...")
    output_path = Path('../results/C_requests.npz')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, C=C)
    
    # Save metadata
    metadata = {
        'project': 'REQUESTS',
        'source': 'https://github.com/psf/requests',
        'description': 'HTTP library for Python',
        'stars': '52000+',
        'tests_total': len(all_tests),
        'tests_sampled': n_tests,
        'branches': n_branches,
        'density': float(density)
    }
    
    with open('../results/C_requests_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n[OK] Coverage matrix saved: {output_path}")
    print(f"     Shape: {C.shape}")
    print(f"     Metadata: {output_path.parent / 'C_requests_metadata.json'}")
    
    print(f"\n[NEXT] Run pipeline:")
    print(f"     cd ..")
    print(f"     python3 run_full_pipeline.py --coverage results/C_requests.npz --target-coverage 0.85")
    
    return True

if __name__ == '__main__':
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n[CANCELLED]")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

