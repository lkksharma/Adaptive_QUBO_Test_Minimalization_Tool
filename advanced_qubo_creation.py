#!/usr/bin/env python3
"""
Advanced Multi-Stage QUBO Creation Algorithm
============================================

INPUT FORMAT REQUIREMENTS:
-------------------------
1. Coverage Matrix: NumPy array C with shape (N_tests, M_branches)
   - C[i,j] = 1 if test i covers branch j, 0 otherwise
   - Binary integer values (0 or 1)
   - Example:
     C = np.array([[1, 0, 1],  # test 0 covers branches 0, 2
                   [0, 1, 1],  # test 1 covers branches 1, 2
                   [1, 1, 0]]) # test 2 covers branches 0, 1

2. Branches (list of dicts):
   Required fields:
   - 'id': str - unique branch identifier (e.g., "B1", "branch_42")
   - 'line': int - line number in source code
   - 'condition': str - human-readable condition description
   - 'expr': str - actual condition expression
   
   Optional fields:
   - 'complexity': str - one of ['low', 'medium', 'high', 'very_high']
   - 'overlap_group': str - logical grouping for overlap analysis
   - 'criticality': float - numerical importance weight

3. Test Cases (list of dicts):
   Required fields:
   - 'id': int or str - unique test identifier
   
   Optional fields:
   - 'name': str - test name
   - Any other test-specific metadata

USAGE EXAMPLE:
-------------
    import numpy as np
    from advanced_qubo_creation import create_advanced_qubo_builder
    
    # Step 1: Load or create coverage matrix
    C = np.load('results/C_myproject.npz')['C']  # Shape: (50, 100)
    
    # Step 2: Define branches
    branches = [
        {'id': f'B{i}', 'line': i*10, 'condition': f'cond_{i}', 
         'expr': f'x > {i}', 'complexity': 'medium'}
        for i in range(C.shape[1])  # M_branches = 100
    ]
    
    # Step 3: Define test cases
    test_cases = [
        {'id': i, 'name': f'test_{i}'}
        for i in range(C.shape[0])  # N_tests = 50
    ]
    
    # Step 4: Build QUBO
    builder = create_advanced_qubo_builder(coverage_matrix=C)
    result = builder.build_advanced_qubo(
        branches=branches,
        test_cases=test_cases,
        iteration_k=0,
        target_coverage=0.8
    )
    
    qubo = result['qubo']  # Dict with (var, var) -> coeff mappings

Implements the sophisticated 5-stage QUBO reweighting algorithm:
1. Coverage Efficiency Analysis
2. Coverage Gap Escalation with Branch Criticality  
3. Unexplored Test Reward Boosting
4. Overlap-Group Coupling Refinement
5. Pipeline Quality-Based Global Scaling

Based on the mathematical formulation:
min_{x ∈ {0,1}^n} (Σ α_i^(k) x_i + Σ β_j^(k) (1-y_j) + Σ γ_ij^(k) x_i x_j + Σ δ_ij^(k) x_i x_j + λ^(k) P(x))
"""

import numpy as np
import json
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Any
import time

class AdvancedQUBOBuilder:
    """Advanced multi-stage QUBO builder with principled reweighting.

    Notes on design choices (no arbitrary constants):
    - All weights are derived from observable structure in the coverage matrix C
      (tests × branches) or iteration feedback (achieved vs. target coverage).
    - Initial coefficients are computed using normalized counts (e.g., per-test
      coverage, per-branch degree) instead of fixed magic numbers.
    - Iterative updates are ratio-based (e.g., scale by coverage shortfall), not
      exponential factors with ad-hoc bases.
    - The QUBO contains only test variables t_i to keep qubit count bounded. The
      coverage penalty is projected into the test-variable space using a standard
      quadratic OR-approximation over C.
    """
    
    def __init__(self, coverage_matrix: np.ndarray = None, config: Dict = None):
        """
        Initialize advanced QUBO builder
        
        Args:
            coverage_matrix: Test-branch coverage matrix (tests × branches)
            config: Configuration dictionary with optional complexity_multipliers
        """
        self.coverage_matrix = coverage_matrix
        self.config = config if config is not None else {}
        
        # Load complexity multipliers from config
        default_multipliers = {
            'isinstance': 0.5,
            'schema': 0.3,
            'error': 0.2
        }
        self.complexity_multipliers = self.config.get(
            'complexity_multipliers', 
            default_multipliers
        )
        self.pipeline_state = {
            'score_current': 0.0,
            'score_prev': 0.0,
            'iteration': 0,
            'all_tests': [],
            'overlap_groups': [],
            'alpha_coeffs': {},  # α_i^(k) - adaptive sparsity coefficients
            'beta_coeffs': {},   # β_j^(k) - coverage penalty coefficients  
            'gamma_coeffs': {},  # γ_ij^(k) - coupling coefficients
            'delta_coeffs': {},  # δ_ij^(k) - overlap reward coefficients
            'lambda_coeff': 0.0  # λ^(k) - global constraint coefficient (set from shortfall)
        }
    
    def build_advanced_qubo(self, branches: List[Dict], test_cases: List[Dict], 
                           iteration_k: int, previous_solution: List[Dict] = None,
                           target_coverage: float = 0.8) -> Dict:
        """
        Build advanced QUBO using 5-stage algorithm
        
        Args:
            branches: List of branch conditions
            test_cases: List of test cases
            iteration_k: Current iteration number
            previous_solution: Previous optimization solution
            target_coverage: Target coverage ratio
            
        Returns:
            Dictionary containing QUBO matrix and metadata
        """
        print(f"Building Advanced QUBO (Iteration {iteration_k})")
        print(f"   Branches: {len(branches)}")
        print(f"   Tests: {len(test_cases)}")
        print(f"   Target Coverage: {target_coverage:.1%}")
        
        # Update pipeline state
        self.pipeline_state['iteration'] = iteration_k
        self.pipeline_state['all_tests'] = test_cases
        
        # Initialize coefficients if first iteration
        if iteration_k == 0:
            self._initialize_coefficients(branches, test_cases)
        else:
            # Apply 5-stage reweighting algorithm
            self._apply_multi_stage_reweighting(branches, test_cases, previous_solution)
        
        # Build QUBO matrix using mathematical formulation
        qubo = self._build_qubo_matrix(branches, test_cases, target_coverage)
        
        return {
            'qubo': qubo,
            'metadata': {
                'iteration': iteration_k,
                'alpha_coeffs': self.pipeline_state['alpha_coeffs'],
                'beta_coeffs': self.pipeline_state['beta_coeffs'],
                'gamma_coeffs': self.pipeline_state['gamma_coeffs'],
                'delta_coeffs': self.pipeline_state['delta_coeffs'],
                'lambda_coeff': self.pipeline_state['lambda_coeff']
            }
        }
    
    def _initialize_coefficients(self, branches: List[Dict], test_cases: List[Dict]):
        """Initialize all coefficients for first iteration"""
        print("  Initializing coefficients...")

        n_tests = len(test_cases)
        n_branches = len(branches)

        # Derive per-test coverage counts from coverage matrix if available
        if self.coverage_matrix is not None and self.coverage_matrix.shape[0] == n_tests:
            per_test_cov = self.coverage_matrix.sum(axis=1)  # shape (n_tests,)
            per_branch_deg = self.coverage_matrix.sum(axis=0)  # shape (n_branches,)
        else:
            # Fallback to uniform counts if no matrix provided
            per_test_cov = np.ones(n_tests)
            per_branch_deg = np.ones(n_branches)

        # Normalize helpers (avoid division by zero)
        eps = 1e-9
        avg_cov = max(per_test_cov.mean(), eps)
        avg_deg = max(per_branch_deg.mean(), eps)

        # Initialize α_i^(0): inverse-proportional to per-test coverage (cheaper if covers more)
        for i in range(n_tests):
            test_id = f"t_{i}"
            self.pipeline_state['alpha_coeffs'][test_id] = avg_cov / max(per_test_cov[i], eps)

        # Initialize β_j^(0): proportional to 1/degree (harder branches penalize more when few tests cover them)
        for j in range(n_branches):
            branch_id = f"b_{j}"
            self.pipeline_state['beta_coeffs'][branch_id] = avg_deg / max(per_branch_deg[j], eps)

        # Initialize γ_ij^(0): redundancy penalty = Jaccard overlap over branches
        for i in range(n_tests):
            for j in range(i + 1, n_tests):
                pair_id = f"gamma_{i}_{j}"
                self.pipeline_state['gamma_coeffs'][pair_id] = self._jaccard_overlap(i, j)

        # Initialize δ_ij^(0): useful-overlap reward = joint-only coverage ratio (negative to reward)
        for i in range(n_tests):
            for j in range(n_tests):
                pair_id = f"delta_{i}_{j}"
                self.pipeline_state['delta_coeffs'][pair_id] = -self._joint_only_ratio(i, j)

        # λ will be set adaptively per iteration based on coverage shortfall
        self.pipeline_state['lambda_coeff'] = 0.0
    
    def _apply_multi_stage_reweighting(self, branches: List[Dict], test_cases: List[Dict], 
                                      previous_solution: List[Dict]):
        """Apply 5-stage reweighting algorithm"""
        print("  Applying 5-stage reweighting algorithm...")
        
        # Stage 1: Coverage Efficiency Analysis
        self._stage1_coverage_efficiency_analysis(test_cases, previous_solution)
        
        # Stage 2: Coverage Gap Escalation with Branch Criticality
        self._stage2_coverage_gap_escalation(branches, test_cases, previous_solution)
        
        # Stage 3: Unexplored Test Reward Boosting
        self._stage3_unexplored_test_boosting(test_cases, previous_solution)
        
        # Stage 4: Overlap-Group Coupling Refinement
        self._stage4_overlap_coupling_refinement(test_cases)
        
        # Stage 5: Pipeline Quality-Based Global Scaling
        self._stage5_global_scaling()
    
    def _stage1_coverage_efficiency_analysis(self, test_cases: List[Dict], 
                                           previous_solution: List[Dict]):
        """Stage 1: Coverage Efficiency Analysis"""
        print("    Stage 1: Coverage Efficiency Analysis")
        
        # Compute data-driven efficiency: coverage per test normalized by redundancy
        n_tests = len(test_cases)
        n_branches = self.coverage_matrix.shape[1] if self.coverage_matrix is not None else 1
        eps = 1e-9
        per_test_cov = self.coverage_matrix.sum(axis=1) if self.coverage_matrix is not None else np.ones(n_tests)

        # Redundancy for test i: average Jaccard overlap with others
        redundancy = np.zeros(n_tests)
        for i in range(n_tests):
            overlaps = []
            for j in range(n_tests):
                if i == j:
                    continue
                overlaps.append(self._jaccard_overlap(i, j))
            redundancy[i] = np.mean(overlaps) if overlaps else 0.0

        efficiency = per_test_cov / (1.0 + redundancy)  # higher is better
        avg_eff = max(efficiency.mean(), eps)

        # α_i^(k) = α_i^(0) * (avg_eff / max(eff_i, eps))
        for i in range(n_tests):
            test_id = f"t_{i}"
            base_alpha = self.pipeline_state['alpha_coeffs'][test_id]  # from init step (already normalized)
            self.pipeline_state['alpha_coeffs'][test_id] = base_alpha * (avg_eff / max(efficiency[i], eps))
    
    def _stage2_coverage_gap_escalation(self, branches: List[Dict], test_cases: List[Dict], 
                                       previous_solution: List[Dict]):
        """Stage 2: Coverage Gap Escalation with Branch Criticality"""
        print("    Stage 2: Coverage Gap Escalation with Branch Criticality")
        
        # Compute coverage shortfall from previous solution
        n_branches = len(branches)
        eps = 1e-9
        if previous_solution:
            covered = self._covered_branch_indices(previous_solution)
            achieved = len(covered) / max(n_branches, 1)
        else:
            achieved = 0.0

        shortfall = max(0.0, 1.0 - achieved)  # scale in [0,1]

        # β_j^(k) = β_j^(0) * (1 + shortfall) for all j; stronger pressure when coverage is low
        for j in range(n_branches):
            branch_id = f"b_{j}"
            base_beta = self.pipeline_state['beta_coeffs'][branch_id]
            self.pipeline_state['beta_coeffs'][branch_id] = base_beta * (1.0 + shortfall)
    
    def _stage3_unexplored_test_boosting(self, test_cases: List[Dict], 
                                        previous_solution: List[Dict]):
        """Stage 3: Unexplored Test Reward Boosting"""
        print("    Stage 3: Unexplored Test Reward Boosting")
        
        # Encourage exploration by lowering α for unselected tests based on their incremental coverage
        if previous_solution:
            covered_prev = set(self._covered_branch_indices(previous_solution))
        else:
            covered_prev = set()

        for i, _ in enumerate(test_cases):
            test_id = f"t_{i}"
            inc = self._incremental_coverage(i, covered_prev)  # count of new branches test i can add
            total = int(self.coverage_matrix[i].sum()) if self.coverage_matrix is not None else 1
            diversity = inc / max(total, 1)
            base_alpha = self.pipeline_state['alpha_coeffs'][test_id]
            # Reduce α proportionally to diversity (more new coverage => cheaper)
            self.pipeline_state['alpha_coeffs'][test_id] = base_alpha / max(1.0 + diversity, 1e-9)
    
    def _stage4_overlap_coupling_refinement(self, test_cases: List[Dict]):
        """Stage 4: Overlap-Group Coupling Refinement"""
        print("    Stage 4: Overlap-Group Coupling Refinement")
        
        # Update γ and δ directly from current coverage structure (no arbitrary multipliers)
        n_tests = len(test_cases)
        for i in range(n_tests):
            for j in range(i + 1, n_tests):
                pair_id = f"gamma_{i}_{j}"
                self.pipeline_state['gamma_coeffs'][pair_id] = self._jaccard_overlap(i, j)
        for i in range(n_tests):
            for j in range(n_tests):
                pair_id = f"delta_{i}_{j}"
                self.pipeline_state['delta_coeffs'][pair_id] = -self._joint_only_ratio(i, j)
    
    def _stage5_global_scaling(self):
        """Stage 5: Pipeline Quality-Based Global Scaling"""
        print("    Stage 5: Pipeline Quality-Based Global Scaling")
        
        # Set λ based on coverage shortfall between achieved and target (data-driven)
        # This is consumed in _build_qubo_matrix by scaling β.
        score_current = self.pipeline_state['score_current']  # reuse as achieved coverage in [0,1]
        target = getattr(self, '_target_coverage', 0.8)
        shortfall = max(0.0, target - score_current)
        self.pipeline_state['lambda_coeff'] = shortfall
    
    def _build_qubo_matrix(self, branches: List[Dict], test_cases: List[Dict], 
                          target_coverage: float) -> Dict:
        """Build QUBO matrix using mathematical formulation"""
        print("  Building QUBO matrix...")
        
        qubo = {}

        n_tests = len(test_cases)
        n_branches = len(branches)
        eps = 1e-9

        # Keep target in state for Stage 5
        self._target_coverage = target_coverage

        # Term 1: Adaptive Sparsity Term - Σ α_i^(k) x_i
        for i in range(n_tests):
            test_id = f"t_{i}"
            alpha_coeff = float(self.pipeline_state['alpha_coeffs'].get(test_id, 1.0))
            qubo[(test_id, test_id)] = qubo.get((test_id, test_id), 0.0) + alpha_coeff

        # Term 2: Coverage Penalty projection onto test variables
        # For each branch j, distribute β_j over tests that cover it, and add pairwise
        # positive terms so that selecting multiple tests for the same branch gives
        # diminishing returns (approximate y_j = OR_i (c_ij x_i)).
        if self.coverage_matrix is not None and self.coverage_matrix.shape[0] == n_tests:
            for j in range(n_branches):
                # Effective coverage penalty scaled by lambda (coverage shortfall)
                beta_j_base = float(self.pipeline_state['beta_coeffs'].get(f"b_{j}", 1.0))
                beta_j = beta_j_base * (1.0 + float(self.pipeline_state.get('lambda_coeff', 0.0)))
                col = self.coverage_matrix[:, j]
                idx = np.where(col > 0.5)[0]
                d_j = max(len(idx), 1)
                # Linear credits (negative): encourage selecting at least one covering test
                for i in idx:
                    test_i_id = f"t_{i}"
                    qubo[(test_i_id, test_i_id)] = qubo.get((test_i_id, test_i_id), 0.0) - (beta_j / d_j)
                # Pairwise penalties (positive): avoid selecting many tests for the same branch
                for a in range(len(idx)):
                    for b in range(a + 1, len(idx)):
                        ti = f"t_{idx[a]}"
                        tj = f"t_{idx[b]}"
                        qubo[(ti, tj)] = qubo.get((ti, tj), 0.0) + (beta_j / (d_j * d_j))

        # Term 3: Coupling Terms - Σ γ_ij^(k) x_i x_j (redundancy)
        for i in range(n_tests):
            for j in range(i + 1, n_tests):
                test_i_id = f"t_{i}"
                test_j_id = f"t_{j}"
                pair_id = f"gamma_{i}_{j}"
                gamma_coeff = float(self.pipeline_state['gamma_coeffs'].get(pair_id, 0.0))
                if abs(gamma_coeff) > 0:
                    qubo[(test_i_id, test_j_id)] = qubo.get((test_i_id, test_j_id), 0.0) + gamma_coeff

        # Term 4: Overlap Rewards - Σ δ_ij^(k) x_i x_j (useful joint coverage)
        for i in range(n_tests):
            for j in range(n_tests):
                test_i_id = f"t_{i}"
                test_j_id = f"t_{j}"
                pair_id = f"delta_{i}_{j}"
                delta_coeff = float(self.pipeline_state['delta_coeffs'].get(pair_id, 0.0))
                if abs(delta_coeff) > 0:
                    qubo[(test_i_id, test_j_id)] = qubo.get((test_i_id, test_j_id), 0.0) + delta_coeff

        # Term 5: Global Constraints - implemented by scaling β via λ in stages (already applied)

        print(f"  QUBO matrix built with {len(qubo)} terms")
        return qubo
    
    # Helper methods for calculations
    def _calculate_branch_complexity(self, branch: Dict) -> float:
        """Optional hook if numeric 'criticality' is provided in branch. Defaults to 1.0."""
        if 'criticality' in branch and isinstance(branch['criticality'], (int, float)):
            return float(branch['criticality'])
        return 1.0
    
    def _calculate_complexity_multiplier(self, branch: Dict) -> float:
        """Calculate complexity multiplier for branch criticality using configurable weights"""
        condition = branch.get('condition', '')
        base_multiplier = 1.0
        
        # Increase multiplier for complex conditions using configurable multipliers
        for keyword, multiplier in self.complexity_multipliers.items():
            if keyword in condition:
                base_multiplier += multiplier
        
        return base_multiplier
    
    def _jaccard_overlap(self, i: int, j: int) -> float:
        """Jaccard overlap between tests i and j based on coverage matrix."""
        if self.coverage_matrix is None:
            return 0.0
        a = self.coverage_matrix[i] > 0.5
        b = self.coverage_matrix[j] > 0.5
        inter = np.logical_and(a, b).sum()
        union = np.logical_or(a, b).sum()
        return float(inter / union) if union > 0 else 0.0

    def _joint_only_ratio(self, i: int, j: int) -> float:
        """Proportion of branches that require both i and j (covered by union but not by either alone).

        Here approximated as branches where both cover (intersection) divided by total branches,
        minus average of per-test coverages to reduce rewarding trivial overlaps.
        """
        if self.coverage_matrix is None:
            return 0.0
        a = self.coverage_matrix[i] > 0.5
        b = self.coverage_matrix[j] > 0.5
        inter = np.logical_and(a, b).sum()
        total = self.coverage_matrix.shape[1]
        base = 0.5 * ((a.sum() / max(total, 1)) + (b.sum() / max(total, 1)))
        ratio = (inter / max(total, 1)) - base
        return float(max(ratio, 0.0))

    def _covered_branch_indices(self, selection: List[Dict]) -> List[int]:
        """Get indices of branches covered by the given selection using the coverage matrix."""
        if self.coverage_matrix is None:
            return []
        indices = []
        rows = []
        for t in selection:
            if 'id' in t:
                rows.append(int(t['id']))
        if not rows:
            return []
        covered = (self.coverage_matrix[rows].sum(axis=0) > 0.5).nonzero()[0]
        return list(map(int, covered))

    def _incremental_coverage(self, test_idx: int, covered_prev: Set[int]) -> int:
        if self.coverage_matrix is None:
            return 0
        cols = np.where(self.coverage_matrix[test_idx] > 0.5)[0]
        return int(len(set(cols) - set(covered_prev)))
    
    def _get_covered_branches(self, previous_solution: List[Dict], branches: List[Dict]) -> List[Dict]:
        """Get branches covered by previous solution"""
        if not previous_solution:
            return []
        
        # Simplified - assume all branches are covered if solution exists
        return branches[:len(previous_solution)]
    
    def _get_unexplored_tests(self, previous_solution: List[Dict], all_tests: List[Dict]) -> List[Dict]:
        """Get unexplored tests"""
        if not previous_solution:
            return all_tests
        
        selected_ids = {t.get('id') for t in previous_solution if 'id' in t}
        return [t for t in all_tests if t.get('id') not in selected_ids]
    
    def _calculate_unique_branches(self, test: Dict) -> int:
        """Calculate number of unique branches covered by test"""
        # Simplified calculation
        return 3  # Placeholder
    
    def _calculate_total_coverage(self, test: Dict) -> int:
        """Calculate total coverage of test"""
        # Simplified calculation
        return 5  # Placeholder
    
    def _create_overlap_groups(self, test_cases: List[Dict]) -> List[List[Tuple[int, int]]]:
        """Create overlap groups for test pairs"""
        groups = []
        for i in range(len(test_cases)):
            for j in range(i + 1, len(test_cases)):
                # Simplified grouping - could be enhanced with actual overlap analysis
                groups.append([(i, j)])
        return groups
    
    def _calculate_overlap_ratio(self, test1: Dict, test2: Dict) -> float:
        """Calculate overlap ratio between two tests"""
        # Simplified calculation
        return 0.3  # Placeholder
    
    def _calculate_global_penalty(self, test_cases: List[Dict], target_coverage: float) -> float:
        """Calculate global penalty function"""
        # Penalty for not achieving target coverage
        return 0.1 * (1.0 - target_coverage)

def create_advanced_qubo_builder(coverage_matrix: np.ndarray = None, config: Dict = None):
    """Factory function to create advanced QUBO builder"""
    return AdvancedQUBOBuilder(coverage_matrix, config)

