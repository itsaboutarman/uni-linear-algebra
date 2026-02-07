import numpy as np


def solve_football_pagerank():
    print("="*40)
    print("Football Team Ranking using PageRank")
    print("="*40)

    # Number of teams
    n = 6

    S = np.zeros((n, n))

    S[0, :] = [0, 17, 25, 25, 10, 30]
    S[1, :] = [38, 0, 24, 48, 21, 29]
    S[2, :] = [20, 31, 0, 14, 24, 17]
    S[3, :] = [36, 3, 25, 0, 24, 45]
    S[4, :] = [24, 30, 13, 14, 0, 0]
    S[5, :] = [28, 24, 20, 10, 23, 0]

    print("Score Matrix S:")
    print(S)
    print("-" * 40)

    R = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                # S[i, j] is goals i scored against j
                # S[j, i] is goals j scored against i
                numerator = S[i, j] + 1
                denominator = S[i, j] + S[j, i] + 2
                R[i, j] = numerator / denominator
            else:
                R[i, j] = 0

    x = np.ones(n)

    # Parameters
    tolerance = 1e-6
    max_iter = 1000
    lambda_val = 0

    for k in range(max_iter):
        # Matrix-vector multiplication
        x_new = R @ x

        # Calculate eigenvalue (using infinity norm or 2-norm)
        # Here we use 2-norm for normalization as described in standard algorithms
        lambda_new = np.linalg.norm(x_new, 2)

        # Normalize the vector
        x_new = x_new / lambda_new

        # Check convergence
        if abs(lambda_new - lambda_val) <= tolerance:
            print(f"Converged after {k+1} iterations.")
            x = x_new
            lambda_val = lambda_new
            break

        x = x_new
        lambda_val = lambda_new

    print("-" * 40)
    print(f"Dominant Eigenvalue (λ): {lambda_val:.6f}")
    print("Eigenvector (v):", np.round(x, 4))
    print("-" * 40)

    # Ranking
    # We sort indices based on the score in descending order
    # argsort gives ascending, so we use [::-1] to reverse
    ranking_indices = np.argsort(x)[::-1]

    print("FINAL RANKING:")
    for rank, idx in enumerate(ranking_indices):
        print(f"Rank {rank+1}: Team {idx+1} (Score: {x[idx]:.4f})")


# Run the function
solve_football_pagerank()
