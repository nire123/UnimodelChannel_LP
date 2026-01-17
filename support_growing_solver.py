"""
Support-Growing Binomial Channel Minimax Solver
================================================

Implements the support-growing algorithm from the spec:
1. Start with initial support S_0 (size 2M+1, includes {0, 0.5, 1})
2. Solve LP on current S → get (ε, z, Q)
3. Oracle: compute g(x; z) on full grid, find local minima
4. Add violators (g < ε - delta_add) to S
5. Check for strict improvement (ε decreases by delta_prune)
6. If improved: prune to support(Q) ∪ newly_added
7. Repeat until grid-optimal

Uses HiGHS dual simplex for sparse solutions (exactly M points).
"""

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from scipy.special import gammaln
from typing import Tuple, Dict, List
from dataclasses import dataclass


@dataclass
class AlgorithmParams:
    """Parameters for the support-growing algorithm"""
    tol_gap: float = 1e-6        # Oracle gap for stopping
    delta_add: float = 1e-8       # Margin for adding points
    delta_prune: float = 1e-6     # Strict improvement threshold
    support_tol: float = 1e-8     # Threshold for Q(x) > 0
    max_iterations: int = 100     # Maximum iterations
    symmetric : bool = False
    verbose: bool = True


def binomial_channel(K: int, N: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate binomial channel W[y, i] = Pr(Bin(N, x_i) = y)
    
    Args:
        K: number of input points
        N: number of trials (output size is N+1)
    
    Returns:
        x_grid: (K,) input values in [0, 1]
        W: (N+1, K) channel matrix, W[y, i] = W(y | x_i)
    """
    x_grid = np.linspace(0, 1, K)
    y_vals = np.arange(N + 1)
    W = np.zeros((N + 1, K))
    
    for i, x in enumerate(x_grid):
        if x == 0:
            W[0, i] = 1.0
        elif x == 1:
            W[N, i] = 1.0
        else:
            log_binom = gammaln(N + 1) - gammaln(y_vals + 1) - gammaln(N - y_vals + 1)
            log_prob = log_binom + y_vals * np.log(x) + (N - y_vals) * np.log(1 - x)
            W[:, i] = np.exp(log_prob)
    
    return x_grid, W


def solve_minimax_lp(W: np.ndarray, w: float, S: np.ndarray, symmetric : bool = False, dual_simple : bool = False) -> Dict:
    N_plus_1, K = W.shape
    S = np.asarray(S, dtype=int)
    n_support = len(S)

    W_S = W[:, S].T                      # (n_support, N_plus_1)

    z = cp.Variable(N_plus_1)
    u = cp.Variable((n_support, N_plus_1), nonneg=True)
    t = cp.Variable()

    constraints = [
        z >= 0,
        z <= 1,          # optional; keep if you like
        u <= W_S,
        u <= cp.reshape(z, (1, N_plus_1)),   # broadcast z across rows
    ]

    # Add symmetry constraints if requested
    if symmetric:
        left_indices = np.arange(N_plus_1 // 2)
        right_indices = N - left_indices
        constraints.append(z[left_indices] == z[right_indices])
            
    expr = cp.sum(u, axis=1) - w * cp.sum(z)  # (n_support,)
    t_constr = (t <= expr)                    # vector inequality (ONE constraint object)
    constraints.append(t_constr)

    prob = cp.Problem(cp.Maximize(t), constraints)
    prob.solve(
        solver=cp.SCIPY,
        scipy_options={'method': 'highs-ds'} if dual_simple else {'method': 'highs-ipm'},
        verbose=False,
        # canon_backend="CPP",   # try this if your build supports it (sometimes faster)
    )

    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"LP solver failed: {prob.status}")

    eps = float(t.value)
    z_opt = np.asarray(z.value).reshape(-1)

    # Duals of t <= expr give Q on S (up to normalization)
    Q = np.asarray(t_constr.dual_value).reshape(-1)
    Q = np.maximum(Q, 0)
    sQ = Q.sum()
    if sQ > 1e-12:
        Q /= sQ

    return {'eps': eps, 'z': z_opt, 'Q': Q, 'S': S, 'status': prob.status}


def compute_g(W: np.ndarray, z: np.ndarray, w: float) -> np.ndarray:
    """
    Compute g(x_i; z) = sum_y min{W[y,i], z[y]} - w * sum_y z[y]
    
    Args:
        W: (N+1, K) channel matrix
        z: (N+1,) threshold vector
        w: parameter (1/M)
    
    Returns:
        g_vec: (K,) score function values
    """
    clipped = np.minimum(W, z[:, np.newaxis])
    r = clipped.sum(axis=0)
    g_vec = r - w * z.sum()
    return g_vec


def find_local_minima(g_vec: np.ndarray, boundary_points: List[int] = None) -> np.ndarray:
    """
    Find local minima indices in g_vec.
    
    A point i is a local minimum if:
    - g[i] <= g[i-1] and g[i] <= g[i+1] (interior points)
    - Always include boundary_points (e.g., 0 and K-1)
    
    Args:
        g_vec: (K,) score function values
        boundary_points: indices to always include (e.g., [0, K-1])
    
    Returns:
        local_min_indices: array of local minimum indices
    """
    K = len(g_vec)
    local_minima = []
    
    # Add boundary points (always keep them)
    if boundary_points is not None:
        local_minima.extend(boundary_points)
    
    # Interior points
    for i in range(1, K - 1):
        if g_vec[i] <= g_vec[i-1] and g_vec[i] <= g_vec[i+1]:
            local_minima.append(i)
    
    return np.array(sorted(set(local_minima)))

def symmetrize_indices(cands: np.ndarray, K: int) -> np.ndarray:
    cands = np.asarray(cands, dtype=int)
    mir = (K - 1) - cands
    return np.unique(np.concatenate([cands, mir]))


def validate_solution(result: Dict, M: int, tol: float = 1e-5) -> bool:
    """
    Validate that solution matches theoretical structure:
    1. |support(Q)| = M
    2. Q is uniform on support
    
    Args:
        result: LP solution
        M: expected support size
        tol: numerical tolerance
    
    Returns:
        valid: True if all checks pass
    """
    Q = result['Q']
    support_mask = Q > tol
    support_size = support_mask.sum()
    
    valid = True
    
    # Check support size
    if support_size != M:
        print(f"  ✗ Support size: expected {M}, got {support_size}")
        valid = False
    
    # Check uniformity
    Q_support = Q[support_mask]
    expected_prob = 1.0 / M
    max_deviation = np.abs(Q_support - expected_prob).max()
    if max_deviation > tol:
        print(f"  ✗ Uniformity: max deviation {max_deviation:.2e} > {tol:.2e}")
        valid = False
    
    return valid


def solve_minimax_lp_fixed_z_full(W: np.ndarray, w: float, z_fixed: np.ndarray,
                              tol: float = 1e-10) -> Dict:
    """
    Solve the minimax LP with fixed z, optimizing over Q on full support.
    
    Given z, solve:
        eps = max_{Q} sum_x Q(x) * [sum_y min(W(y|x), z_y) - w * sum_y z_y]
        
    subject to:
        sum_x Q(x) = 1
        Q(x) >= 0  for all x in [0, K-1]
        Q_X(W(y|X) > z_y) <= w <= Q_X(W(y|X) >= z_y)  for all y
    
    Does NOT use a restricted support S - optimizes over all K inputs.
    """
    N_plus_1, K = W.shape
    z_fixed = np.asarray(z_fixed).reshape(-1)
    
    # W is already (N_plus_1, K), transpose for convenience
    W_T = W.T                            # (K, N_plus_1)
    
    # Compute u(x,y) = min(W(y|x), z_y) for all x in [0,K-1], y in [N+1]
    z_broadcast = np.reshape(z_fixed, (1, N_plus_1))  # (1, N_plus_1)
    u = np.minimum(W_T, z_broadcast)                   # (K, N_plus_1)
    
    # Compute expr[x] = sum_y u(x,y) - w * sum_y z_y for each x
    expr = np.sum(u, axis=1) - w * np.sum(z_fixed)     # (K,)
    
    # Variables
    Q = cp.Variable(K, nonneg=True)
    
    constraints = [
        cp.sum(Q) == 1,
    ]
    
    # Add the constraints for each y:
    # Q_X(W(y|X) > z_y) <= w <= Q_X(W(y|X) >= z_y)
    for y in range(N_plus_1):
        z_y = z_fixed[y]
        W_y = W_T[:, y]  # W(y|x) for all x in [0, K-1]
        
        # Classify each x based on W(y|x) vs z_y with tolerance
        mask_gt = W_y > z_y + tol
        mask_ge = W_y >= z_y - tol
        
        # Q_X(W(y|X) > z_y) <= w
        if np.any(mask_gt):
            constraints.append(cp.sum(Q[mask_gt]) <= w)
        
        # Q_X(W(y|X) >= z_y) >= w
        if np.any(mask_ge):
            constraints.append(cp.sum(Q[mask_ge]) >= w)
    
    # Objective
    objective = cp.Maximize(Q @ expr)
    
    prob = cp.Problem(objective, constraints)
    prob.solve(
        solver=cp.SCIPY,
        scipy_options={'method': 'highs-ds'},
        verbose=False,
    )
    
    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"LP solver failed: {prob.status}")
    
    eps = float(prob.value)
    Q_opt = np.asarray(Q.value).reshape(-1)
    
    # Extract the support (non-zero entries)
    S = np.where(Q_opt > 1e-10)[0]
    
    return {
        'eps': eps,
        'z': z_fixed,
        'Q': Q_opt,
        'S': S,
        'status': prob.status
    }

def solve_minimax_lp_fixed_z(W: np.ndarray, w: float, S: np.ndarray, 
                              z_fixed: np.ndarray,
                              tol: float = 1e-10) -> Dict:
    """
    Solve the minimax LP with fixed z, optimizing only over Q.
    
    Given z, solve:
        eps = max_{Q} sum_x Q(x) * [sum_y min(W(y|x), z_y) - w * sum_y z_y]
        
    subject to:
        sum_x Q(x) = 1
        Q(x) >= 0
        Q_X(W(y|X) > z_y) <= w <= Q_X(W(y|X) >= z_y)  for all y
    
    Uses tolerance to handle W(y|x) ≈ z_y:
    - If |W(y|x) - z_y| < tol, treat as W(y|x) = z_y
    - Strict inequality: W(y|x) > z_y + tol
    - Non-strict inequality: W(y|x) >= z_y - tol
    """
    N_plus_1, K = W.shape
    S = np.asarray(S, dtype=int)
    n_support = len(S)
    z_fixed = np.asarray(z_fixed).reshape(-1)
    
    W_S = W[:, S].T                      # (n_support, N_plus_1)
    
    # Compute u(x,y) = min(W(y|x), z_y) for all x in S, y in [N+1]
    z_broadcast = np.reshape(z_fixed, (1, N_plus_1))  # (1, N_plus_1)
    u = np.minimum(W_S, z_broadcast)                   # (n_support, N_plus_1)
    
    # Compute expr[x] = sum_y u(x,y) - w * sum_y z_y for each x in S
    expr = np.sum(u, axis=1) - w * np.sum(z_fixed)     # (n_support,)
    
    # Variables
    Q = cp.Variable(n_support, nonneg=True)
    
    
    
    
    
    constraints = [
        cp.sum(Q) == 1,
    ]


    
    if True:
        left_indices = np.arange(n_support // 2)
        right_indices = (n_support - 1) - left_indices
        constraints.append(Q[left_indices] == Q[right_indices])
        
        constraints.append(Q[-1] == w)
        constraints.append(Q[0] == w)

    
    # Add the constraints for each y:
    # Q_X(W(y|X) > z_y) <= w <= Q_X(W(y|X) >= z_y)
    for y in range(N_plus_1):
        z_y = z_fixed[y]
        W_y = W_S[:, y]  # W(y|x) for all x in S
        
        # Classify each x based on W(y|x) vs z_y with tolerance
        # W(y|x) > z_y (strictly greater, accounting for tolerance)
        mask_gt = W_y > z_y + tol
        
        # W(y|x) >= z_y (greater or equal, accounting for tolerance)
        # This includes W(y|x) ≈ z_y within tolerance
        mask_ge = W_y >= z_y - tol
        
        # Q_X(W(y|X) > z_y) <= w
        if np.any(mask_gt):
            constraints.append(cp.sum(Q[mask_gt]) <= w)
        
        # Q_X(W(y|X) >= z_y) >= w
        if np.any(mask_ge):
            constraints.append(cp.sum(Q[mask_ge]) >= w)
    
    # Objective
    objective = cp.Maximize(Q @ expr)
    
    prob = cp.Problem(objective, constraints)
    prob.solve(
        solver=cp.SCIPY,
        scipy_options={'method': 'highs-ds'},
        verbose=False,
    )
    
    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"LP solver failed: {prob.status}")
    
    eps = float(prob.value)
    Q_opt = np.asarray(Q.value).reshape(-1)
    
    return {
        'eps': eps,
        'z': z_fixed,
        'Q': Q_opt,
        'S': S,
        'status': prob.status
    }


def solve_minimax_lp_fixed_z_old(W: np.ndarray, w: float, S: np.ndarray, 
                              z_fixed: np.ndarray) -> Dict:
    """
    Solve the minimax LP with fixed z, optimizing only over Q.
    
    Given z, solve:
        eps = max_{Q} sum_x Q(x) * [sum_y min(W(y|x), z_y) - w * sum_y z_y]
        
    subject to:
        sum_x Q(x) = 1
        Q(x) >= 0
        Q_X(W(y|X) > z_y) <= w <= Q_X(W(y|X) >= z_y)  for all y
    
    where Q_X(W(y|X) > z_y) = sum_{x: W(y|x) > z_y} Q(x)
    """
    N_plus_1, K = W.shape
    S = np.asarray(S, dtype=int)
    n_support = len(S)
    z_fixed = np.asarray(z_fixed).reshape(-1)
    
    W_S = W[:, S].T                      # (n_support, N_plus_1)
    
    # Compute u(x,y) = min(W(y|x), z_y) for all x in S, y in [N+1]
    z_broadcast = np.reshape(z_fixed, (1, N_plus_1))  # (1, N_plus_1)
    u = np.minimum(W_S, z_broadcast)                   # (n_support, N_plus_1)
    
    # Compute expr[x] = sum_y u(x,y) - w * sum_y z_y for each x in S
    expr = np.sum(u, axis=1) - w * np.sum(z_fixed)     # (n_support,)
    
    # Variables
    Q = cp.Variable(n_support, nonneg=True)
    
    constraints = [
        cp.sum(Q) == 1,
    ]
    
    # Add the constraints for each y:
    # Q_X(W(y|X) > z_y) <= w <= Q_X(W(y|X) >= z_y)
    for y in range(N_plus_1):
        # Find indices where W(y|x) > z_y
        mask_gt = W_S[:, y] > z_fixed[y]   # (n_support,) boolean
        # Find indices where W(y|x) >= z_y
        mask_ge = W_S[:, y] >= z_fixed[y]  # (n_support,) boolean
        
        # Q_X(W(y|X) > z_y) = sum_{x: W(y|x) > z_y} Q(x)
        if np.any(mask_gt):
            constraints.append(cp.sum(Q[mask_gt]) <= w)
        
        # Q_X(W(y|X) >= z_y) >= w
        if np.any(mask_ge):
            constraints.append(cp.sum(Q[mask_ge]) >= w)
    
    # Objective
    objective = cp.Maximize(Q @ expr)
    
    prob = cp.Problem(objective, constraints)
    prob.solve(
        solver=cp.SCIPY,
        scipy_options={'method': 'highs-ds'},
        verbose=False,
    )
    
    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"LP solver failed: {prob.status}")
    
    eps = float(prob.value)
    Q_opt = np.asarray(Q.value).reshape(-1)
    
    return {
        'eps': eps,
        'z': z_fixed,
        'Q': Q_opt,
        'S': S,
        'status': prob.status
    }




def support_growing_solver(W: np.ndarray, x_grid: np.ndarray, M: int, 
                           params: AlgorithmParams = None) -> Dict:
    """
    Main support-growing algorithm.
    
    Args:
        W: (N+1, K) channel matrix
        x_grid: (K,) input grid points
        M: codebook size
        params: algorithm parameters
    
    Returns:
        dict with:
            - eps: final minimax value
            - z: final threshold vector
            - Q: final prior
            - S: final support indices
            - history: list of (iteration, eps, |S|) tuples
            - converged: whether algorithm converged
    """
    if params is None:
        params = AlgorithmParams()
    
    N_plus_1, K = W.shape
    w = 1.0 / M
    
    # Initialize support S_0: 2M+1 points including {0, 0.5, 1}
    # We want indices for 0, 0.5, 1, and M-1 points on each side
    idx_0 = 0
    idx_half = K // 2
    idx_1 = K - 1
    
    # Create M+1 uniformly spaced points
    S_0_indices = np.linspace(0, K-1, M+1, dtype=int)
    # Ensure we have 0, 0.5, 1
    S_0 = np.unique(np.concatenate([[idx_0, idx_half, idx_1], S_0_indices]))
    
    if params.verbose:
        print("="*70)
        print("SUPPORT-GROWING ALGORITHM")
        print("="*70)
        print(f"Parameters:")
        print(f"  N={N_plus_1-1}, K={K}, M={M}, w=1/M={w:.6f}")
        print(f"  tol_gap={params.tol_gap:.2e}, delta_add={params.delta_add:.2e}")
        print(f"  delta_prune={params.delta_prune:.2e}, support_tol={params.support_tol:.2e}")
        print(f"\nInitial support S_0: size={len(S_0)}")
        print(f"  Indices: {S_0}")
        print(f"  x-values: {x_grid[S_0]}")
        print("="*70)
    
    S = S_0.copy()
    history = []
    iteration = 0
    converged = False
    eps_old = np.inf
    newly_added = set()
    
    # Boundary points to always keep
    boundary_points = [0, K-1]








    eps_best = np.inf

    # optional: use a mask for fast membership/updates
    S_mask = np.zeros(K, dtype=bool)
    S_mask[S] = True


    
    while iteration < params.max_iterations:
        
        if params.verbose:
            print(f"\n{'='*70}")
            print(f"Iteration {iteration}: |S| = {len(S)}")
            print(f"{'='*70}")
        

        # Solve LP on current support
        
        S = symmetrize_indices(S, K)

        result = solve_minimax_lp(W, w, S, symmetric=params.symmetric)
                
        eps = result["eps"]
        z   = result["z"]
        Q   = result["Q"]
        
        if params.verbose:
            print(f"LP solved: ε = {eps:.8f}, status = {result['status']}")
    
        # oracle
        g_vec = compute_g(W, z, w)
        g_min = g_vec.min()
        gap = eps - g_min
        
        if False:
            if 1-g_min/eps < 0.01:
                if params.verbose:
                    print(f"\n{'='*70}")
                    print("CONVERGED: Grid-optimal solution found!")
                    print(f"{'='*70}")
                converged = True
                break
        
        
        if gap <= params.tol_gap:
            if params.verbose:
                print(f"\n{'='*70}")
                print("CONVERGED: Grid-optimal solution found!")
                print(f"{'='*70}")
            converged = True
            break

        if params.verbose:
            print(f"Oracle: min(g) = {g_min:.8f}, gap = {gap:.2e}")

        # Record history
        history.append((iteration, eps, len(S), gap))
    
        # candidates (your local-min approach is fine)
        local_min = find_local_minima(g_vec, boundary_points=[0, K-1])
        local_max = find_local_minima(-g_vec, boundary_points=[0, K-1])
        saddle_point = np.unique(np.concatenate([local_min, local_max]))
        
        support_indices = S[result["Q"] > params.support_tol]

        new_support = np.unique(np.concatenate([saddle_point, support_indices]))
        
        candidates = local_min[g_vec[local_min] <= eps - params.delta_add]

    
        # remove ones already in S
        candidates = candidates[~S_mask[candidates]]
        if candidates.size == 0:
            candidates = np.array([int(np.argmin(g_vec))])
            if params.verbose:
                print(f"No local minima below threshold, adding global min at i={candidates[0]}")
    
        # support(Q) indices within current S
        

        if params.verbose:
            print(f"Adding {len(candidates)} candidates to S")
            print(f"  Indices: {candidates}")
            print(f"  g-values: {g_vec[candidates]}")
    
        # update rule
        if eps <= eps_best - params.delta_prune:
            eps_best = eps
            # prune (but KEEP candidates to prevent thrash)
            S_mask[:] = False
            S_mask[support_indices] = True
            S_mask[candidates] = True
            S_mask[0] = True
            S_mask[K-1] = True
        else:
            # stagnation: just grow
            S_mask[candidates] = True
    
        S = np.flatnonzero(S_mask)
        # S = new_support
    
        iteration += 1
    
    # Final solution
    result_final = solve_minimax_lp(W, w, S, symmetric=False, dual_simple=True)
    z = result_final['z']
    eps = result_final['eps']
    g_vec_final = compute_g(W, z, w)

    if False:
        S_new = np.where(np.isclose(g_vec_final, eps))[0]
        
        assert (S_new[::-1]+S_new == K-1).all()
        t = solve_minimax_lp_fixed_z(W, w, S_new, result_final['z'])
        
        assert np.isclose(t['eps'], eps)
        
        
        # t = solve_minimax_lp_fixed_z(W, w, result_final['S'], result_final['z'])
        
        
        # t1 = solve_minimax_lp_fixed_z_full(W, w, result_final['z'], tol=1e-10)

    
    
    suppurt_ix = result_final['S'][result['Q'] > 1/M/2]
    Q_support =  suppurt_ix/(K-1)
    W_support = W[:, suppurt_ix]
    
    
    


    
    
    
    g_vec_final = compute_g(W, result_final['z'], w)
    gap_final = result_final['eps'] - g_vec_final.min()


    W = W_support

    max_vals = W.max(axis=1, keepdims=True)
    
    # Find all positions that achieve the maximum (boolean array)
    is_max = np.isclose(W, max_vals)
    
    num_of_max = is_max.sum(axis=1)
    ((W*is_max)/num_of_max[:,None]).sum(axis=0)
    
    error_as_a_function_of_x = 1-((W*is_max)/num_of_max[:,None]).sum(axis=0)
    
    if params.verbose:
        print(f"\n{'='*70}")
        print("FINAL RESULTS")
        print(f"{'='*70}")
        print(f"Iterations: {iteration}")
        print(f"Converged: {converged}")
        print(f"Final ε: {result_final['eps']:.8f}")
        print(f"Final gap: {gap_final:.2e}")
        print(f"Final |S|: {len(S)}")
        print(f"\nFinal support indices: {S}")
        print(f"Final support x-values: {x_grid[S]}")
        print(f"\nValidating final solution:")
        valid = validate_solution(result_final, M, tol=1e-5)
        if valid:
            print(f"  ✓ Structure validated: {M} uniform points")
        print(f"{'='*70}")
    
    return {
        'eps': result_final['eps'],
        'z': result_final['z'],
        'Q': result_final['Q'],
        'S': S,
        'x_support': x_grid[S],
        'history': history,
        'converged': converged,
        'g_vec': g_vec_final,
        'Q_support' : Q_support,
        'W_support' : W_support,
        'error_as_a_function_of_x' : error_as_a_function_of_x
    }


def plot_convergence(history: List[Tuple], M: int):
    """Plot convergence history."""
    iterations = [h[0] for h in history]
    eps_vals = [h[1] for h in history]
    S_sizes = [h[2] for h in history]
    gaps = [h[3] for h in history]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # ε vs iteration
    axes[0].plot(iterations, eps_vals, 'b-o', linewidth=2, markersize=6)
    axes[0].set_xlabel('Iteration', fontsize=11)
    axes[0].set_ylabel('ε (minimax value)', fontsize=11)
    axes[0].set_title('Convergence of ε', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    # |S| vs iteration
    axes[1].plot(iterations, S_sizes, 'g-s', linewidth=2, markersize=6)
    axes[1].axhline(y=M, color='r', linestyle='--', linewidth=1.5, label=f'Target M={M}')
    axes[1].set_xlabel('Iteration', fontsize=11)
    axes[1].set_ylabel('|S| (support size)', fontsize=11)
    axes[1].set_title('Support Size Evolution', fontsize=12)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    
    # Gap vs iteration (log scale)
    axes[2].semilogy(iterations, gaps, 'r-^', linewidth=2, markersize=6)
    axes[2].set_xlabel('Iteration', fontsize=11)
    axes[2].set_ylabel('Gap (ε - min g)', fontsize=11)
    axes[2].set_title('Oracle Gap (log scale)', fontsize=12)
    axes[2].grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    return fig


def plot_final_solution(x_grid: np.ndarray, g_vec: np.ndarray, result: Dict, M: int):
    """Plot final g(x; z) with support points."""
    Q = result['Q']
    S = result['S']
    eps = result['eps']
    
    support_mask = Q > 1e-5
    support_indices = S[support_mask]
    support_x = x_grid[support_indices]
    support_g = g_vec[support_indices]
    
    plt.figure(figsize=(12, 6))
    plt.plot(x_grid, g_vec, 'b-', linewidth=2, label='g(x; z)')
    plt.scatter(support_x, support_g, c='red', s=120, zorder=5, 
                label=f'Support (M={len(support_x)})')
    plt.axhline(y=eps, color='green', linestyle='--', linewidth=1.5, 
                label=f'ε* = {eps:.6f}')
    plt.xlabel('x (input)', fontsize=12)
    plt.ylabel('g(x; z)', fontsize=12)
    plt.title(f'Final Solution: Score Function g(x; z)', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt.gcf()


def main(N,M,K,symmetric, verbose):
    """Run support-growing experiment."""
    # Parameters
    
        
    if verbose:
        # Generate channel
        print("Generating binomial channel...")

    x_grid, W = binomial_channel(K, N)
    
    if verbose:
        print(f"Channel: Bin({N}), Grid: {K} points")
    
    # Set algorithm parameters
    params = AlgorithmParams(
        tol_gap=1e-6,
        delta_add=1e-8,
        delta_prune=1e-6,
        support_tol=1e-8,
        max_iterations=50,
        symmetric = symmetric,
        verbose=verbose
    )
    
    # Run support-growing algorithm
    result = support_growing_solver(W, x_grid, M, params)
    
    # Plot convergence
    if verbose:
        print("\nGenerating convergence plots...")
        fig_conv = plot_convergence(result['history'], M)
        fig_conv.savefig('support_growing_convergence.png', 
                         dpi=150, bbox_inches='tight')
    
        # Plot final solution
        fig_final = plot_final_solution(x_grid, result['g_vec'], result, M)
        fig_final.savefig('support_growing_final.png', 
                          dpi=150, bbox_inches='tight')
    
        print(f"\nPlots saved to /mnt/user-data/outputs/")
    
    return result, x_grid, W



if __name__ == "__main__":


    all_res = dict()
    for M in range(3,12):
        print(M)
        #M = 25
        N = M**2
        K = 2001
        verbose = True
        symmetric = False
    
    
    
        result, x_grid, W = main(N,M,K,symmetric, verbose)
        all_res[M] = result
        
        
import matplotlib.pyplot as plt
import numpy as np

# Your data
data = {m:(x['Q_support'], x['error_as_a_function_of_x']) for m,x in all_res.items() if m > 2}

# Create the plot
fig = plt.figure(figsize=(12, 6))  # Made wider to accommodate legend
for m, (codewords, errors) in data.items():
    plt.plot(codewords, errors, 'o-', label=f'M = {m}', markersize=8, linewidth=2)

plt.xlabel('Codeword (x)', fontsize=12)
plt.ylabel('Error Probability P(error|x)', fontsize=12)
plt.title('Error Probability as a Function of Transmitted Codeword', fontsize=14)

# Legend outside on the right
plt.legend(fontsize=11, bbox_to_anchor=(1.05, 1), loc='upper left')

plt.grid(True, alpha=0.3)
plt.xlim(-0.05, 1.05)
plt.ylim(bottom=-0.005)
plt.tight_layout()

fig.savefig('error_prob.png', dpi=150, bbox_inches='tight')
plt.show()        
        
