# Quantum Test Selection Pipeline

A sophisticated test suite optimization framework using quantum-inspired algorithms and classical baselines.

## Overview

This pipeline optimizes test suites by selecting a minimal subset of tests that achieves maximum branch coverage, using:
- **Advanced QUBO (5-stage reweighting algorithm)** with quantum annealing simulation
- **Basic Simulated Annealing** baseline
- **Basic Adaptive Greedy** baseline

## Required Input Format

### 1. Coverage Matrix (`.npz` file)

**Location**: `results/C_<project_name>.npz`

**Format**: NumPy compressed archive containing:
```python
{
    'C': np.ndarray  # Shape: (N_tests, M_branches)
                     # Binary matrix where C[i,j] = 1 if test i covers branch j
}
```

**Example**:
```python
import numpy as np

# Example: 50 tests, 100 branches
N_tests = 50
M_branches = 100
C = np.random.randint(0, 2, size=(N_tests, M_branches))

# Save
np.savez('results/C_myproject.npz', C=C)
```

### 2. Branch Metadata (Optional, `branches.json`)

**Location**: Root directory or project-specific folder

**Format**: JSON array of branch objects:
```json
[
  {
    "id": "B1",
    "line": 106,
    "col_offset": 0,
    "condition": "if x > 0",
    "expr": "x > 0",
    "complexity": "low",
    "overlap_group": "group1"
  },
  ...
]
```

**Required fields**:
- `id` (string): Unique branch identifier
- `line` (int): Line number in source code
- `condition` (string): Human-readable condition
- `expr` (string): Actual expression

**Optional fields**:
- `complexity` (string): `low`, `medium`, `high`, `very_high`
- `overlap_group` (string): Logical grouping for overlap analysis
- `criticality` (float): Numerical importance weight

### 3. Configuration (Optional, `config/baselines.json`)

**Location**: `config/baselines.json`

**Format**:
```json
{
  "target_coverage": 0.8,
  "beta_size_penalty": 0.5,
  "alpha_size_penalty": 0.2,
  "initial_temperature": 1.0,
  "final_temperature": 0.001,
  "steps": 5000,
  "seed": 42
}
```

## Installation

### Requirements
```bash
pip install numpy scipy pennylane
```

### Optional (for pytest-based coverage generation)
```bash
pip install pytest coverage
```

## Usage

### Quick Start

1. **Prepare your coverage matrix**:
```bash
# Option A: Use provided builder (requires pytest)
python build_coverage_generic.py --project myproject --test-dir tests/

# Option B: Manual creation
python -c "
import numpy as np
C = your_coverage_matrix  # Shape: (tests, branches)
np.savez('results/C_myproject.npz', C=C)
"
```

2. **Run the full pipeline**:
```bash
export COVERAGE_NPZ=results/C_myproject.npz
python run_full_pipeline.py
```

3. **View results**:
```
results/
  ├── baselines_sa_result.json       # Simulated Annealing results
  ├── baselines_adaptive_result.json # Adaptive Greedy results
  ├── advanced_results.json          # Advanced QUBO results
  └── comparison.csv                 # Summary comparison
```

### Individual Methods

#### Simulated Annealing
```bash
export COVERAGE_NPZ=results/C_myproject.npz
python basic_simulated_annealing.py
```

#### Adaptive Greedy
```bash
export COVERAGE_NPZ=results/C_myproject.npz
python basic_adaptive.py
```

#### Advanced QUBO
```bash
python run_advanced_qubo.py --coverage results/C_myproject.npz
```

## Output Format

### JSON Results
Each method outputs JSON with:
```json
{
  "method": "Advanced_QAOA",
  "coverage_ratio": 0.95,
  "selected_count": 25,
  "total_tests": 100,
  "selected_indices": [0, 5, 12, ...],
  "energy": -245.67,
  "config": { ... }
}
```

### CSV Comparison
```csv
Method,FV,Coverage,Tests
Basic_Simulated_Annealing,0.024,1.0,6
Basic_Adaptive_Greedy,0.086,0.944,3
Advanced_QAOA,-3.206,1.0,6
```

## Architecture

### Pipeline Structure
```
Coverage Matrix (C)
    ↓
┌───────────────────┐
│  Basic Baselines  │
├───────────────────┤
│ - Simulated Ann.  │
│ - Adaptive Greedy │
└───────────────────┘
    ↓
┌───────────────────────┐
│  Advanced QUBO (5-stage) │
├───────────────────────┤
│ 1. Coverage Efficiency │
│ 2. Gap Escalation      │
│ 3. Test Reward Boost   │
│ 4. Overlap Refinement  │
│ 5. Global Scaling      │
└───────────────────────┘
    ↓
┌───────────────────┐
│ Quantum Annealing │
│    (QAOA/PennyLane) │
└───────────────────┘
    ↓
  Results
```

### File Descriptions

| File | Purpose |
|------|---------|
| `advanced_qubo_creation.py` | 5-stage QUBO builder |
| `basic_simulated_annealing.py` | Classical SA baseline |
| `basic_adaptive.py` | Greedy adaptive baseline |
| `quantum_annealer_simulator.py` | QAOA quantum solver |
| `run_full_pipeline.py` | End-to-end orchestration |
| `build_coverage_generic.py` | Generic coverage builder |

## Customization

### Adding Custom Coverage Generators

Create a script that produces `C` matrix:
```python
import numpy as np

def build_custom_coverage():
    # Your coverage collection logic
    C = ...  # Shape: (N_tests, M_branches)
    np.savez('results/C_custom.npz', C=C)
    return C

if __name__ == '__main__':
    build_custom_coverage()
```

### Adjusting Algorithm Parameters

Edit `config/baselines.json`:
- `target_coverage`: Stop when this coverage is reached (0.0-1.0)
- `alpha_size_penalty`/`beta_size_penalty`: Control test suite size vs coverage trade-off
- `steps`: SA iterations (more = better solutions, slower)
- `initial_temperature`/`final_temperature`: SA temperature schedule

## Citation

If you use this pipeline in research, please cite:
```
[Your paper/project citation here]
```

## License

[Specify your license]

## Support

For issues or questions, please open an issue on GitHub or contact [your contact info].

