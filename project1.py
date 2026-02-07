import numpy as np

def classical_gram_schmidt(V):
    """
    Classical Gram-Schmidt Algorithm (CGS)
    
    Input:
        V: Matrix whose columns are the input vectors
    
    Output:
        Q: Orthonormal matrix whose columns are orthonormal vectors
    """
    n, m = V.shape
    Q = np.zeros((n, m))
    
    for k in range(m):
        # Copy the k-th column
        q_k = V[:, k].copy()
        
        # Subtract all components parallel to previous vectors
        for i in range(k):
            r_ik = np.dot(V[:, k], Q[:, i])
            q_k = q_k - r_ik * Q[:, i]
        
        # Normalize
        norm_qk = np.linalg.norm(q_k)
        if norm_qk > 1e-10:  # Avoid division by zero
            Q[:, k] = q_k / norm_qk
        else:
            Q[:, k] = q_k
    
    return Q


def modified_gram_schmidt(V):
    """
    Modified Gram-Schmidt Algorithm (MGS)
    
    Input:
        V: Matrix whose columns are the input vectors
    
    Output:
        Q: Orthonormal matrix whose columns are orthonormal vectors
    """
    n, m = V.shape
    Q = V.copy().astype(float)
    
    for k in range(m):
        # Normalize the k-th column
        norm_qk = np.linalg.norm(Q[:, k])
        if norm_qk > 1e-10:
            Q[:, k] = Q[:, k] / norm_qk
        
        # Update subsequent columns
        for j in range(k + 1, m):
            r_kj = np.dot(Q[:, k], Q[:, j])
            Q[:, j] = Q[:, j] - r_kj * Q[:, k]
    
    return Q


def compute_orthogonality_error(Q):
    """
    Compute orthogonality error using ||Q^T Q - I||_2
    
    Input:
        Q: Orthonormal matrix
    
    Output:
        Orthogonality error (2-norm)
    """
    I = np.eye(Q.shape[1])
    error = np.linalg.norm(Q.T @ Q - I, ord=2)
    return error


# Define input vectors
v1 = np.array([1, 1/2, 1/3, 1/4, 1/5])
v2 = np.array([1/2, 1/3, 1/4, 1/5, 1/6])
v3 = np.array([1/3, 1/4, 1/5, 1/6, 1/7])

# Construct matrix V
V = np.column_stack([v1, v2, v3])

print("=" * 70)
print("Comparison of Classical and Modified Gram-Schmidt Algorithms")
print("=" * 70)

# Run Classical Gram-Schmidt
Q_classical = classical_gram_schmidt(V)
error_classical = compute_orthogonality_error(Q_classical)

print("\n--- Classical Gram-Schmidt (CGS) ---")
print("\nResulting Q matrix:")
print(Q_classical)
print(f"\nOrthogonality error: ||Q^T Q - I||_2 = {error_classical:.6e}")

# Run Modified Gram-Schmidt
Q_modified = modified_gram_schmidt(V)
error_modified = compute_orthogonality_error(Q_modified)

print("\n" + "=" * 70)
print("--- Modified Gram-Schmidt (MGS) ---")
print("\nResulting Q matrix:")
print(Q_modified)
print(f"\nOrthogonality error: ||Q^T Q - I||_2 = {error_modified:.6e}")

# Comparison
print("\n" + "=" * 70)
print("--- Comparison of Results ---")
print(f"CGS Error: {error_classical:.6e}")
print(f"MGS Error: {error_modified:.6e}")
print(f"Improvement ratio: {error_classical / error_modified:.2f}x")

# Compute Q^T Q for both methods
print("\n--- Detailed Orthogonality Check ---")
print("\nCGS - Q^T Q:")
print(Q_classical.T @ Q_classical)
print("\nMGS - Q^T Q:")
print(Q_modified.T @ Q_modified)

print("\n" + "=" * 70)
print("Conclusion: MGS algorithm has better numerical stability than CGS")
print("=" * 70)