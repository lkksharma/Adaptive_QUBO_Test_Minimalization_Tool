#!/usr/bin/env python3
"""
Basic Adaptive baseline (classical) without advanced QUBO reweighting.

Greedy adaptive selection driven by marginal coverage gain per test size cost.

Objective proxy: maximize CR subject to minimal set size; we implement a simple
score = (newly covered branches count) - beta * 1, picking tests iteratively.
Hyperparameters loaded from config/baselines.json:
  - target_coverage (stop when reached)
  - beta_size_penalty (encourage sparsity)

Outputs JSON to results/baselines_adaptive_result.json
"""

from __future__ import annotations

import json
import os
from typing import Dict

import numpy as np


def load_config() -> Dict:
    cfg_path = os.path.join('config', 'baselines.json')
    defaults = {
        'target_coverage': 0.8,
        'beta_size_penalty': 0.5,
        'seed': 42,
    }
    if os.path.isfile(cfg_path):
        with open(cfg_path, 'r') as f:
            data = json.load(f)
        defaults.update(data)
    return defaults


def load_coverage_matrix() -> np.ndarray:
    explicit = os.environ.get('COVERAGE_NPZ')
    cerb = os.path.join('results', 'C_cerberus_ast.npz')
    synth = os.path.join('results', 'C_tcas_synth.npz')
    gcov = os.path.join('results', 'C_tcas_gcov.npz')
    if explicit and os.path.isfile(explicit):
        C = np.load(explicit)['C']
    elif os.path.isfile(cerb):
        C = np.load(cerb)['C']
    elif os.path.isfile(synth):
        C = np.load(synth)['C']
    elif os.path.isfile(gcov):
        C = np.load(gcov)['C']
    else:
        raise FileNotFoundError('No coverage matrix found. Build cerberus via run_cerberus_coverage.py or run build_synthetic_coverage.py.')
    
    # Validate dimensions
    if C.ndim != 2:
        raise ValueError(f'Coverage matrix must be 2D, got shape {C.shape}')
    
    N, M = C.shape
    if N == 0 or M == 0:
        raise ValueError(f'Coverage matrix has invalid dimensions: ({N}, {M})')
    
    return C


def basic_adaptive(C: np.ndarray, target_coverage: float, beta_size_penalty: float, seed: int = 42) -> Dict:
    rng = np.random.default_rng(seed)
    N, M = C.shape
    selected = np.zeros(N, dtype=bool)
    covered = np.zeros(M, dtype=bool)

    # Precompute coverage sets per test for speed
    test_covers = [set(np.where(C[i] > 0)[0].tolist()) for i in range(N)]

    while covered.mean() < target_coverage:
        best_gain = -1e9
        best_idx = None
        for i in range(N):
            if selected[i]:
                continue
            new_cov = test_covers[i] - set(np.where(covered)[0].tolist())
            gain = len(new_cov) - beta_size_penalty  # cost of adding 1 test
            if gain > best_gain or (gain == best_gain and rng.random() < 0.5):
                best_gain = gain
                best_idx = i
        if best_idx is None or best_gain <= 0:
            # No positive-gain test; break to avoid selecting everything
            break
        selected[best_idx] = True
        covered[list(test_covers[best_idx])] = True

    cr = covered.mean() if M > 0 else 0.0
    sel_idx = np.where(selected)[0].tolist()
    result = {
        'method': 'Basic_Adaptive_Greedy',
        'coverage_ratio': float(cr),
        'selected_count': int(selected.sum()),
        'total_tests': int(N),
        'selected_indices': sel_idx,
        'config': {
            'target_coverage': float(target_coverage),
            'beta_size_penalty': float(beta_size_penalty),
            'seed': int(seed),
        },
    }
    return result


def main():
    cfg = load_config()
    C = load_coverage_matrix()
    res = basic_adaptive(C, float(cfg['target_coverage']), float(cfg['beta_size_penalty']), int(cfg['seed']))
    os.makedirs('results', exist_ok=True)
    out_path = os.path.join('results', 'baselines_adaptive_result.json')
    with open(out_path, 'w') as f:
        json.dump(res, f, indent=2)
    print(f"Adaptive baseline complete → {out_path}")
    print(f"CR={res['coverage_ratio']*100:.1f}%, selected={res['selected_count']}/{res['total_tests']}")


if __name__ == '__main__':
    main()

