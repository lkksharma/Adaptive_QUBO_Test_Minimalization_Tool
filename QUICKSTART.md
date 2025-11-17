# Quick Start Guide

Get up and running with the Quantum Test Selection Pipeline in 5 minutes!

## 1. Installation

```bash
# Clone or download the pipeline
cd quantum_test_selection_pipeline

# Install dependencies
pip install -r requirements.txt
```

## 2. Prepare Your Data

### Option A: Use Synthetic Data (for testing)

```bash
python build_coverage_template.py --mode synthetic --n-tests 50 --n-branches 100
```

### Option B: Use Your Real Project

Create a Python script to build your coverage matrix:

```python
import numpy as np

# Your coverage collection logic here
# Example: Load from your existing coverage tool
C = your_coverage_matrix  # Shape: (N_tests, M_branches)

# Must be binary (0 or 1)
C = (C > 0).astype(int)

# Save in required format
np.savez('results/C_myproject.npz', C=C)
```

## 3. Run the Pipeline

```bash
# Run complete pipeline
python run_full_pipeline.py --coverage results/C_myproject.npz

# Or with custom parameters
python run_full_pipeline.py \
    --coverage results/C_myproject.npz \
    --target-coverage 0.85 \
    --max-iterations 10 \
    --output-dir my_results
```

## 4. View Results

```bash
# Results are saved in the output directory
ls results/

# View CSV summary
cat results/comparison.csv

# View detailed JSON results
cat results/baselines_sa_result.json
cat results/baselines_adaptive_result.json
cat results/advanced_results.json
```

## Example Output

```csv
Method,FV,Coverage,Tests
Basic_Simulated_Annealing,0.024,1.0,6
Basic_Adaptive_Greedy,0.086,0.944,3
Advanced_QAOA,-3.206,1.0,6
```

## Next Steps

1. **Customize Configuration**: Edit `config/baselines.json` to tune algorithm parameters
2. **Integrate with CI/CD**: Add pipeline to your continuous integration workflow
3. **Analyze Results**: Use provided JSON files for deeper analysis
4. **Iterate**: Run multiple times with different parameters to find optimal settings

## Common Issues

### Issue: "No coverage matrix found"
**Solution**: Ensure your coverage matrix is saved as `results/C_*.npz` or specify path explicitly

### Issue: "Module not found: pennylane"
**Solution**: Install dependencies with `pip install -r requirements.txt`

### Issue: Quantum solver is slow
**Solution**: Reduce test count or adjust `--max-iterations` parameter

### Issue: Low coverage achieved
**Solution**: 
- Check your coverage matrix quality
- Increase `--target-coverage` threshold
- Increase `--max-iterations` for more optimization attempts

## Configuration Examples

### High Coverage, More Tests
```json
{
  "target_coverage": 0.95,
  "alpha_size_penalty": 0.1,
  "beta_size_penalty": 0.3
}
```

### Minimal Test Suite
```json
{
  "target_coverage": 0.80,
  "alpha_size_penalty": 0.5,
  "beta_size_penalty": 0.7
}
```

### Balanced Approach
```json
{
  "target_coverage": 0.85,
  "alpha_size_penalty": 0.2,
  "beta_size_penalty": 0.5
}
```

## Support

- **Documentation**: See README.md for full documentation
- **Examples**: Check `examples/` directory for more use cases
- **Issues**: Report bugs or request features on GitHub

## Performance Tips

1. **Large Test Suites**: For >100 tests, consider pre-filtering low-value tests
2. **Many Branches**: Quantum solving scales with number of variables; use batching
3. **Parallel Processing**: Run multiple instances with different seeds for robustness
4. **Caching**: Save intermediate results to avoid recomputation

## Citation

If you use this pipeline in research, please cite:
```bibtex
@software{quantum_test_selection,
  title={Quantum Test Selection Pipeline},
  author={Your Name},
  year={2024},
  url={https://github.com/yourrepo}
}
```

