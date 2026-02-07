# Numerical Linear Algebra (NLA) Course Projects
### Amirkabir University of Technology (Tehran Polytechnic)

This repository contains a collection of practical projects implemented for the Numerical Linear Algebra course under the supervision of Dr. Mehdi Dehghan. The projects focus on implementing fundamental linear algebra algorithms, ensuring numerical stability, and exploring real-world applications. Each project is organized into its own directory and includes the source code and implementation details.

## Projects Overview

### [Project 1: Gram-Schmidt Orthogonalization](./project1.py)
A comparative study between Classical Gram-Schmidt (CGS) and Modified Gram-Schmidt (MGS) algorithms. It focuses on analyzing numerical stability and loss of orthogonality using the Frobenius norm.

### [Project 2: Tridiagonal System Solvers](./project2.py)
Implementation of efficient solvers for large-scale tridiagonal systems. This project includes the Cholesky Decomposition and the Thomas Algorithm (TDMA), with performance benchmarking against standard libraries.

### [Project 3: Iterative Methods for Linear Systems](./project3.py)
Implementation of stationary iterative solvers for high-dimensional systems. It covers Jacobi, Gauss-Seidel, and Successive Over-Relaxation (SOR) methods, including an analysis of convergence rates.

### [Project 4: PageRank Algorithm and Power Method](./project4.py)
Implementation of the Power Method to compute the dominant eigenvector of a transition matrix, applied to ranking systems (e.g., football team rankings).

### [Project 5: Constrained Least Squares Steganography](./project5/main.py)
A digital steganography approach where hidden messages are embedded into image pixels. It utilizes Constrained Least Squares to minimize image distortion while ensuring message recovery.

### [Project 6: Handwritten Digit Classification via SVD](./project6/main.py)
Classification of the USPS Handwritten Digits dataset using Singular Value Decomposition (SVD). The project extracts orthogonal basis vectors for each digit class and classifies test data based on the minimum residual norm.
