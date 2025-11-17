#!/usr/bin/env python3
"""
Basic Simulated Annealing Baseline for Test Suite Optimization
=============================================================

INPUT FORMAT REQUIREMENTS:
-------------------------
Coverage Matrix: NumPy compressed archive (.npz)
  File: results/C_<project>.npz
  Contents: {'C': ndarray of shape (N_tests, M_branches)}
  
  Example creation:
    import numpy as np
    C = np.random.randint(0, 2, size=(50, 100))  # 50 tests, 100 branches
    np.savez('results/C_myproject.npz', C=C)

Configuration (optional): config/baselines.json
  {
    "target_coverage": 0.8,          # Target coverage ratio
    "alpha_size_penalty": 0.2,       # Size penalty coefficient
    "initial_temperature": 1.0,      # SA initial temperature
    "final_temperature": 0.001,      # SA final temperature
    "steps": 5000,                   # Number of SA iterations
    "seed": 42                       # Random seed
  }

USAGE:
-----
  # Option 1: Via environment variable
  export COVERAGE_NPZ=results/C_myproject.npz
  python basic_simulated_annealing.py
  
  # Option 2: Auto-detection (checks multiple standard locations)
  python basic_simulated_annealing.py

OUTPUT:
------
  results/baselines_sa_result.json containing:
  {
    "method": "Basic_Simulated_Annealing",
    "coverage_ratio": 0.95,
    "selected_count": 25,
    "total_tests": 100,
    "selected_indices": [0, 5, 12, ...],
    "energy": 0.024,
    "config": { ... }
  }

ALGORITHM:
---------
Minimizes: E(x) = (1 - CR(x)) + alpha * (|x| / N)
  where CR(x) is the branch coverage ratio of selected tests
  
Uses geometric cooling schedule: T_k = T_0 * r^k
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class SABaselineConfig:
    target_coverage: float = 0.8
    alpha_size_penalty: float = 0.2
    initial_temperature: float = 1.0
    final_temperature: float = 1e-3
    steps: int = 5000
    seed: int = 42


def load_config() -> SABaselineConfig:
    cfg_path = os.path.join('config', 'baselines.json')
    if os.path.isfile(cfg_path):
        with open(cfg_path, 'r') as f:
            data = json.load(f)
        return SABaselineConfig(
            target_coverage=float(data.get('target_coverage', 0.8)),
            alpha_size_penalty=float(data.get('alpha_size_penalty', 0.2)),
            initial_temperature=float(data.get('initial_temperature', 1.0)),
            final_temperature=float(data.get('final_temperature', 1e-3)),
            steps=int(data.get('steps', 5000)),
            seed=int(data.get('seed', 42)),
        )
    return SABaselineConfig()


def load_coverage_matrix() -> Tuple[np.ndarray, int, int]:
    results_dir = 'results'
    explicit = os.environ.get('COVERAGE_NPZ')
    cerb_path = os.path.join(results_dir, 'C_cerberus_ast.npz')
    synth_path = os.path.join(results_dir, 'C_tcas_synth.npz')
    gcov_path = os.path.join(results_dir, 'C_tcas_gcov.npz')
    if explicit and os.path.isfile(explicit):
        C = np.load(explicit)['C']
    elif os.path.isfile(cerb_path):
        C = np.load(cerb_path)['C']
    elif os.path.isfile(synth_path):
        C = np.load(synth_path)['C']
    elif os.path.isfile(gcov_path):
        C = np.load(gcov_path)['C']
    else:
        raise FileNotFoundError('No coverage matrix found. Build cerberus via run_cerberus_coverage.py or run build_synthetic_coverage.py.')
    
    # Validate dimensions
    if C.ndim != 2:
        raise ValueError(f'Coverage matrix must be 2D, got shape {C.shape}')
    
    num_tests, num_branches = C.shape
    
    if num_tests == 0 or num_branches == 0:
        raise ValueError(f'Coverage matrix has invalid dimensions: ({num_tests}, {num_branches})')
    
    return C, num_tests, num_branches


def compute_coverage_ratio(C: np.ndarray, x: np.ndarray) -> float:
    # x shape (N,), binary
    if x.sum() == 0:
        return 0.0
    covered = (C[x.astype(bool)].sum(axis=0) > 0).astype(np.int32)
    return float(covered.sum()) / float(C.shape[1]) if C.shape[1] > 0 else 0.0


def energy(C: np.ndarray, x: np.ndarray, alpha: float) -> float:
    N = x.shape[0]
    cr = compute_coverage_ratio(C, x)
    size_term = x.sum() / float(max(N, 1))
    return (1.0 - cr) + alpha * size_term


def neighbor(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    y = x.copy()
    idx = int(rng.integers(0, len(x)))
    y[idx] = 1 - y[idx]
    return y


def simulated_annealing(C: np.ndarray, cfg: SABaselineConfig) -> Dict:
    rng = np.random.default_rng(cfg.seed)
    N = C.shape[0]
    x = np.zeros(N, dtype=np.int8)  # start from empty set to avoid over-selection
    e = energy(C, x, cfg.alpha_size_penalty)

    T0 = cfg.initial_temperature
    T_end = cfg.final_temperature
    steps = max(1, cfg.steps)
    cooling = (T_end / T0) ** (1.0 / steps)
    T = T0

    best_x = x.copy()
    best_e = e

    for _ in range(steps):
        y = neighbor(x, rng)
        e_new = energy(C, y, cfg.alpha_size_penalty)
        if e_new < e:
            x, e = y, e_new
        else:
            # Metropolis criterion
            accept_prob = np.exp(-(e_new - e) / max(T, 1e-12))
            if rng.random() < accept_prob:
                x, e = y, e_new
        if e < best_e:
            best_x, best_e = x.copy(), e
        T *= cooling

    cr = compute_coverage_ratio(C, best_x)
    selected_indices = np.where(best_x == 1)[0].tolist()
    result = {
        'method': 'Basic_Simulated_Annealing',
        'coverage_ratio': float(cr),
        'selected_count': int(best_x.sum()),
        'total_tests': int(N),
        'selected_indices': selected_indices,
        'energy': float(best_e),
        'config': {
            'alpha_size_penalty': cfg.alpha_size_penalty,
            'initial_temperature': cfg.initial_temperature,
            'final_temperature': cfg.final_temperature,
            'steps': cfg.steps,
            'seed': cfg.seed,
        },
    }
    return result


def main():
    C, _, _ = load_coverage_matrix()
    cfg = load_config()
    result = simulated_annealing(C, cfg)
    os.makedirs('results', exist_ok=True)
    out_path = os.path.join('results', 'baselines_sa_result.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"SA baseline complete → {out_path}")
    print(f"CR={result['coverage_ratio']*100:.1f}%, selected={result['selected_count']}/{result['total_tests']}, energy={result['energy']:.4f}")


if __name__ == '__main__':
    main()
