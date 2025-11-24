"""Basis functions for building controllers."""
import torch
from typing import List, Tuple
from itertools import combinations_with_replacement


def polynomial_basis(state: torch.Tensor, order: int = 2, include_constant: bool = True) -> torch.Tensor:
    """Generate polynomial basis functions up to specified order.

    For state [x1, x2], order=2 generates:
        [1, x1, x2, x1^2, x2^2, x1*x2]

    Args:
        state: State vector of shape (n,)
        order: Maximum polynomial order
        include_constant: Whether to include constant term (1)

    Returns:
        Basis functions as 1D tensor
    """
    n_vars = len(state)
    basis_terms = []

    # Constant term
    if include_constant:
        basis_terms.append(torch.ones(1, dtype=state.dtype, device=state.device))

    # Generate all monomials up to specified order
    for deg in range(1, order + 1):
        # For each degree, get all combinations with replacement
        # e.g., for 2 vars, degree 2: [(0,0), (0,1), (1,1)] -> [x1^2, x1*x2, x2^2]
        for indices in combinations_with_replacement(range(n_vars), deg):
            # Compute monomial by multiplying the selected variables
            term = torch.ones(1, dtype=state.dtype, device=state.device)
            for idx in indices:
                term = term * state[idx]
            basis_terms.append(term)

    return torch.cat(basis_terms)


def get_basis_size(n_vars: int, order: int = 2, include_constant: bool = True) -> int:
    """Calculate the number of basis functions.

    Formula: For n variables and order k, number of monomials is C(n+k, k)
    Total = sum from degree 0 to order

    Args:
        n_vars: Number of state variables
        order: Maximum polynomial order
        include_constant: Whether constant term is included

    Returns:
        Number of basis functions
    """
    from math import comb

    total = 0
    if include_constant:
        total += 1

    for deg in range(1, order + 1):
        total += comb(n_vars + deg - 1, deg)

    return total


def get_basis_names(var_names: List[str], order: int = 2, include_constant: bool = True) -> List[str]:
    """Get human-readable names for basis functions.

    Args:
        var_names: Names of variables (e.g., ['A', 'B'])
        order: Maximum polynomial order
        include_constant: Whether constant term is included

    Returns:
        List of basis function names (e.g., ['1', 'A', 'B', 'A^2', 'B^2', 'A*B'])
    """
    n_vars = len(var_names)
    names = []

    # Constant term
    if include_constant:
        names.append('1')

    # Generate monomial names
    for deg in range(1, order + 1):
        for indices in combinations_with_replacement(range(n_vars), deg):
            # Count occurrences of each variable
            var_counts = [0] * n_vars
            for idx in indices:
                var_counts[idx] += 1

            # Build name string
            term_parts = []
            for var_idx, count in enumerate(var_counts):
                if count == 0:
                    continue
                elif count == 1:
                    term_parts.append(var_names[var_idx])
                else:
                    term_parts.append(f"{var_names[var_idx]}^{count}")

            names.append('*'.join(term_parts))

    return names
