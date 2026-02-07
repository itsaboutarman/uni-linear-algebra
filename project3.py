import numpy as np
import time

# --- 1. Problem Parameters Setup ---
# Set parameters based on the problem description
n = 100                 # Matrix dimension (100x100)
r = 0.25                # Parameter 'r'
tol = 0.5e-14           # Stopping criterion (tolerance) from the prompt
max_iter = 20000        # Maximum iterations to prevent infinite loops

# --- 2. Build Matrix A ---
# A is a tridiagonal matrix
A = np.zeros((n, n))
diag_val = 1 + 2 * r        # Main diagonal entries
off_diag_val = -r           # Off-diagonal entries

# Fill the matrix A
for i in range(n):
    A[i, i] = diag_val
    if i > 0:
        A[i, i-1] = off_diag_val
    if i < n - 1:
        A[i, i+1] = off_diag_val

# --- 3. Build vector b and initial guess X0 ---
b = np.ones(n)              # Vector b (all ones)
X0 = np.zeros(n)            # Initial guess (zero vector)

print(f"Problem setup complete: n={n}, r={r}, tol={tol}\n")

# (a) Jacobi Iteration Algorithm


def solve_jacobi(A, b, X0, tol, max_iter):
    """ (a) Implementation of the Jacobi method """
    n = len(b)
    X_old = np.copy(X0)
    X_new = np.copy(X0)

    for k in range(max_iter):
        for i in range(n):
            # Calculate the sigma sum using values from the *previous* iteration
            sigma = 0
            for j in range(n):
                if i != j:
                    sigma += A[i, j] * X_old[j]

            X_new[i] = (b[i] - sigma) / A[i, i]

        # Check stopping criterion using L-infinity norm
        if np.linalg.norm(X_new - X_old, np.inf) < tol:
            return X_new, k + 1  # k+1 because iterations are 0-indexed

        # Prepare for the next iteration
        X_old = np.copy(X_new)

    return X_new, max_iter  # Return if max_iter is reached

# (b) Gauss-Seidel Iteration Algorithm


def solve_gauss_seidel(A, b, X0, tol, max_iter):
    """ (b) Implementation of the Gauss-Seidel method """
    n = len(b)
    X = np.copy(X0)

    for k in range(max_iter):
        X_old_for_tol = np.copy(X)  # Keep a copy for the tolerance check

        for i in range(n):
            sigma_L = 0  # Lower-triangular part (using new X values)
            for j in range(i):
                sigma_L += A[i, j] * X[j]  # Uses new X[j]

            sigma_U = 0  # Upper-triangular part (using old X values)
            for j in range(i + 1, n):
                # Uses old X_old_for_tol[j]
                sigma_U += A[i, j] * X_old_for_tol[j]

            X[i] = (b[i] - sigma_L - sigma_U) / A[i, i]

        # Check stopping criterion
        if np.linalg.norm(X - X_old_for_tol, np.inf) < tol:
            return X, k + 1

    return X, max_iter

# (c) SOR (Successive Over-Relaxation) Algorithm


def solve_sor(A, b, X0, omega, tol, max_iter):
    """ (c) Implementation of the SOR method """
    n = len(b)
    X = np.copy(X0)

    for k in range(max_iter):
        X_old_for_tol = np.copy(X)

        for i in range(n):
            sigma_L = 0
            for j in range(i):
                sigma_L += A[i, j] * X[j]  # Uses new X[j]

            sigma_U = 0
            for j in range(i + 1, n):
                # Uses old X_old_for_tol[j]
                sigma_U += A[i, j] * X_old_for_tol[j]

            # Calculate the Gauss-Seidel term first
            gs_term = (b[i] - sigma_L - sigma_U) / A[i, i]

            # Apply the SOR formula (weighted average)
            X[i] = (1 - omega) * X_old_for_tol[i] + omega * gs_term

        # Check stopping criterion
        if np.linalg.norm(X - X_old_for_tol, np.inf) < tol:
            return X, k + 1

    return X, max_iter

# Main execution block to run all parts and print results


# --- Run Part (a) ---
start_time = time.time()
sol_j, iter_j = solve_jacobi(A, b, X0, tol, max_iter)
time_j = time.time() - start_time
print(f"--- (a) Jacobi Method ---")
print(f"Iterations required: {iter_j}")
print(f"Execution time: {time_j:.4f} seconds\n")


# --- Run Part (b) ---
start_time = time.time()
sol_gs, iter_gs = solve_gauss_seidel(A, b, X0, tol, max_iter)
time_gs = time.time() - start_time
print(f"--- (b) Gauss-Seidel Method ---")
print(f"Iterations required: {iter_gs}")
print(f"Execution time: {time_gs:.4f} seconds\n")


# --- Run Part (c) ---
omegas_to_test = [1.1, 1.2, 1.3, 1.5, 1.9]  # From the problem statement
print(f"--- (c) SOR Method ---")
results_sor = {}

for w in omegas_to_test:
    start_time = time.time()
    sol_sor, iter_sor = solve_sor(A, b, X0, w, tol, max_iter)
    time_sor = time.time() - start_time
    results_sor[w] = iter_sor
    print(
        f"  For ω = {w}:\tIterations = {iter_sor}\t(Time: {time_sor:.4f} sec)")

# Find the best omega from the tested list
best_w = min(results_sor, key=results_sor.get)
print(
    f"\n'More optimal' value from the list: ω = {best_w} (with {results_sor[best_w]} iterations).")


# --- Final Comparison ---
print("\n--- Final Results Comparison ---")
print(f"Jacobi Iterations: \t\t{iter_j}")
print(f"Gauss-Seidel Iterations: \t{iter_gs}")
print(f"Best SOR Iterations (at ω={best_w}): \t{results_sor[best_w]}")
