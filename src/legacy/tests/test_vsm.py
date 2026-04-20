"""
Legacy unit tests for VSM encoding and DIMACS parsing.

These tests target the original single-channel VSM convention:
  vsm[i, j] in {-1, 0, +1}
where sign encodes literal polarity inside each clause.
"""

import pytest
import numpy as np
import torch
from typing import List, Tuple, Optional


class MockVSMEncoder:
    def __init__(self, max_clauses: int = 100, max_vars: int = 50):
        self.max_clauses = max_clauses
        self.max_vars = max_vars

    def cnf_to_vsm(self, clauses: List[List[int]], num_vars: int) -> np.ndarray:
        vsm = np.zeros((len(clauses), num_vars), dtype=np.float32)
        for i, clause in enumerate(clauses):
            for lit in clause:
                j = abs(lit) - 1
                if 0 <= j < num_vars:
                    vsm[i, j] = 1.0 if lit > 0 else -1.0
        return vsm

    def parse_dimacs(self, content: str) -> Tuple[List[List[int]], int, int]:
        lines = content.strip().splitlines()
        num_vars = 0
        declared_num_clauses = 0
        clauses: List[List[int]] = []
        cur: List[int] = []

        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith("c"):
                continue
            if line.startswith("p"):
                parts = line.split()
                if len(parts) >= 4 and parts[1] == "cnf":
                    num_vars = int(parts[2])
                    declared_num_clauses = int(parts[3])
                continue

            for tok in line.split():
                try:
                    lit = int(tok)
                except ValueError:
                    continue
                if lit == 0:
                    if cur:
                        clauses.append(cur)
                        cur = []
                else:
                    cur.append(lit)

        if cur:
            clauses.append(cur)

        if num_vars == 0:
            all_vars = [abs(l) for c in clauses for l in c]
            num_vars = max(all_vars) if all_vars else 0

        return clauses, num_vars, declared_num_clauses

    def pad_vsm(
        self,
        vsm: np.ndarray,
        pad_clauses: Optional[int] = None,
        pad_vars: Optional[int] = None,
    ) -> torch.Tensor:
        if pad_clauses is None:
            pad_clauses = self.max_clauses
        if pad_vars is None:
            pad_vars = self.max_vars

        num_clauses, num_vars = vsm.shape
        padded = np.zeros((pad_clauses, pad_vars), dtype=np.float32)
        c = min(num_clauses, pad_clauses)
        v = min(num_vars, pad_vars)
        padded[:c, :v] = vsm[:c, :v]
        return torch.from_numpy(padded)


@pytest.fixture
def encoder():
    return MockVSMEncoder(max_clauses=100, max_vars=50)


class TestVSMBasicConversion:
    def test_simple_clause(self, encoder):
        clauses = [[1, 2, 3]]
        vsm = encoder.cnf_to_vsm(clauses, num_vars=3)
        assert vsm.shape == (1, 3)
        assert np.array_equal(vsm[0], [1.0, 1.0, 1.0])

    def test_negative_literals(self, encoder):
        clauses = [[-1, 2, -3]]
        vsm = encoder.cnf_to_vsm(clauses, num_vars=3)
        assert vsm.shape == (1, 3)
        assert np.array_equal(vsm[0], [-1.0, 1.0, -1.0])

    def test_multiple_clauses(self, encoder):
        clauses = [[1, -2], [-1, 3], [2, -3]]
        vsm = encoder.cnf_to_vsm(clauses, num_vars=3)
        assert vsm.shape == (3, 3)
        assert np.array_equal(vsm[0], [1.0, -1.0, 0.0])
        assert np.array_equal(vsm[1], [-1.0, 0.0, 1.0])
        assert np.array_equal(vsm[2], [0.0, 1.0, -1.0])

    def test_unit_clauses(self, encoder):
        clauses = [[1], [-2], [3]]
        vsm = encoder.cnf_to_vsm(clauses, num_vars=3)
        assert vsm.shape == (3, 3)
        assert np.array_equal(vsm[0], [1.0, 0.0, 0.0])
        assert np.array_equal(vsm[1], [0.0, -1.0, 0.0])
        assert np.array_equal(vsm[2], [0.0, 0.0, 1.0])

    def test_empty_clause_handling(self, encoder):
        clauses = [[1, 2], [], [3, -4]]
        vsm = encoder.cnf_to_vsm(clauses, num_vars=4)
        assert vsm.shape == (3, 4)
        assert np.all(vsm[1] == 0)


class TestVSMProperties:
    def test_vsm_values_range(self, encoder):
        clauses = [[1, -2, 3], [-1, 2, -3], [4, -5]]
        vsm = encoder.cnf_to_vsm(clauses, num_vars=5)
        unique = np.unique(vsm)
        assert all(v in (-1.0, 0.0, 1.0) for v in unique.tolist())

    def test_vsm_sparsity(self, encoder):
        np.random.seed(42)
        num_vars = 50
        num_clauses = 200
        clauses = []
        for _ in range(num_clauses):
            vars_in_clause = np.random.choice(
                np.arange(1, num_vars + 1), size=3, replace=False
            )
            signs = np.random.choice([-1, 1], size=3)
            clauses.append([int(v * s) for v, s in zip(vars_in_clause, signs)])
        vsm = encoder.cnf_to_vsm(clauses, num_vars=num_vars)
        expected_density = 3 / num_vars
        actual_density = np.count_nonzero(vsm) / vsm.size
        assert abs(actual_density - expected_density) < 0.05

    def test_clause_non_empty(self, encoder):
        clauses = [[1, 2], [-3, 4, 5], [6]]
        vsm = encoder.cnf_to_vsm(clauses, num_vars=6)
        for i, clause in enumerate(clauses):
            if clause:
                assert np.any(vsm[i] != 0)


class TestDIMACSParsing:
    def test_simple_dimacs(self, encoder):
        dimacs = """
        c This is a comment
        p cnf 3 2
        1 -2 0
        2 3 0
        """
        clauses, num_vars, num_clauses = encoder.parse_dimacs(dimacs)
        assert num_vars == 3
        assert num_clauses == 2
        assert clauses == [[1, -2], [2, 3]]

    def test_dimacs_with_long_clauses(self, encoder):
        dimacs = """
        p cnf 5 1
        1 2 3 4 5 0
        """
        clauses, num_vars, _ = encoder.parse_dimacs(dimacs)
        assert num_vars == 5
        assert clauses[0] == [1, 2, 3, 4, 5]

    def test_dimacs_negative_only(self, encoder):
        dimacs = """
        p cnf 3 1
        -1 -2 -3 0
        """
        clauses, num_vars, _ = encoder.parse_dimacs(dimacs)
        assert num_vars == 3
        assert clauses[0] == [-1, -2, -3]


class TestTensorConversion:
    def test_padding(self, encoder):
        clauses = [[1, -2], [3]]
        vsm = encoder.cnf_to_vsm(clauses, num_vars=3)
        t = encoder.pad_vsm(vsm, pad_clauses=5, pad_vars=7)
        assert t.shape == (5, 7)
        assert t[:2, :3].abs().sum().item() > 0
        assert torch.all(t[2:, :] == 0)
        assert torch.all(t[:, 3:] == 0)

