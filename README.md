# Numerical Linear Algebra (NLA) Projects
### Amirkabir University of Technology (Tehran Polytechnic)

This repository contains a collection of practical projects implemented for the **Numerical Linear Algebra** course under the supervision of **Dr. Mehdi Dehghan**. The projects focus on implementing fundamental linear algebra algorithms, ensuring numerical stability, and exploring real-world applications such as image processing and ranking systems.

---

## Projects Overview

### 1. Gram-Schmidt Orthogonalization
A comparative study between **Classical Gram-Schmidt (CGS)** and **Modified Gram-Schmidt (MGS)** algorithms.
* **Technical Focus:** Analyzing numerical stability and loss of orthogonality using the Frobenius norm of $||Q^T Q - I||_2$.
* **Key Finding:** MGS significantly reduces re-orthogonalization errors in floating-point arithmetic compared to CGS.

### 2. Tridiagonal System Solvers
Efficient implementation of solvers for large-scale tridiagonal systems $Ax = b$.
* **Algorithms:** **Cholesky Decomposition** and the **Thomas Algorithm** (TDMA).
* **Technical Focus:** Performance benchmarking against `numpy.linalg.solve` for sparse matrices with $N=1000$.

### 3. Iterative Methods for Linear Systems
Implementation of stationary iterative solvers for high-dimensional systems ($100 \times 100$).
* **Methods:** **Jacobi**, **Gauss-Seidel**, and **Successive Over-Relaxation (SOR)**.
* **Technical Focus:** Analyzing convergence rates and identifying the optimal relaxation parameter ($\omega$) for the SOR method.

### 4. PageRank Algorithm & Power Method
Ranking a set of entities (e.g., football teams) based on a transition matrix derived from scoring results.
* **Technical Focus:** Utilizing the **Power Method** to compute the dominant eigenvector (Perron-Frobenius vector) of a stochastic-like matrix.

### 5. Constrained Least Squares Steganography
A digital steganography approach where a hidden message is embedded into image pixels.
* **Technical Focus:** Solving a **Constrained Least Squares** problem to minimize image distortion ($||z||_2$) while ensuring 100% message recovery accuracy.

### 6. Handwritten Digit Classification via SVD
Classification of the **USPS Handwritten Digits** dataset using the concept of "Singular Images."
* **Technical Focus:** Extracting orthogonal basis vectors for each digit's subspace using **Singular Value Decomposition (SVD)** and classifying test data based on the minimum residual norm.

