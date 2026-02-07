import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_svmlight_file


class SVDClassifier:
    def __init__(self, n_components=10):
        """
        Initializes the classifier.
        n_components: Number of singular vectors (k) to use for classification.
        """
        self.n_components = n_components
        self.u_matrices = {}  # To store U matrix for each digit class
        self.classes = []

    def fit(self, X, y):
        """
        Step 1 & 2: Data Preparation and SVD Calculation.
        Groups data by digit and computes SVD for each group.
        """
        self.classes = np.unique(y)
        print(f"Training SVD models for classes: {self.classes}...")

        for digit in self.classes:
            # Filter data for the current digit
            digit_data = X[y == digit]

            # Transpose to shape (features, samples) for SVD: (256, N)
            digit_data_T = digit_data.T

            # Compute SVD: A = U * S * Vt
            # We only need U. full_matrices=False makes it faster.
            U, S, Vt = np.linalg.svd(digit_data_T, full_matrices=False)

            # Store the U matrix
            self.u_matrices[digit] = U

    def get_residual(self, vector, U_k):
        """
        Calculates the residual norm: || (I - Uk * Uk^T) * z ||
        This is equivalent to || z - projection ||
        """
        # Projection of vector onto the subspace spanned by U_k
        weights = np.dot(U_k.T, vector)
        projection = np.dot(U_k, weights)

        residual = np.linalg.norm(vector - projection)
        return residual

    def predict(self, X_test, k=None):
        """
        Step 4: Classification.
        Assigns each test image to the class with the minimum residual.
        """
        if k is None:
            k = self.n_components

        predictions = []

        for i in range(len(X_test)):
            test_vector = X_test[i]  # Shape (256,)
            residuals = []

            for digit in self.classes:
                U = self.u_matrices[digit]
                # Select the first k basis vectors
                U_k = U[:, :k]

                res = self.get_residual(test_vector, U_k)
                residuals.append(res)

            # Find the index (class) with minimum residual
            predicted_class = self.classes[np.argmin(residuals)]
            predictions.append(predicted_class)

        return np.array(predictions)

    def visualize_singular_images(self):
        """
        Step 3: Display Singular Images.
        Plots the first 3 singular vectors (u1, u2, u3) for each class.
        """
        print("Visualizing first 3 singular images for each class...")
        n_classes = len(self.classes)
        # Handle case where few classes exist to avoid plot error
        if n_classes == 0:
            print("No classes found to visualize.")
            return

        fig, axes = plt.subplots(n_classes, 3, figsize=(6, 2 * n_classes))

        # Ensure axes is 2D array even if n_classes is 1
        if n_classes == 1:
            axes = np.array([axes])

        for idx, digit in enumerate(self.classes):
            U = self.u_matrices[digit]
            for mode in range(3):
                # Reshape the column vector back to 16x16 image
                img_vector = U[:, mode]
                img_matrix = img_vector.reshape(16, 16)

                ax = axes[idx, mode]
                ax.imshow(img_matrix, cmap='gray')
                ax.axis('off')
                if mode == 0:
                    ax.set_title(f"Digit {digit} - Mode 1")
                else:
                    ax.set_title(f"Mode {mode+1}")

        plt.tight_layout()
        plt.show()

    def plot_residual_spaghetti(self, X_test, y_test, target_digits, k=10):
        """
        Step 5 (Part 2): Spaghetti plot of residuals for specific digits.
        """
        print(f"Plotting residual analysis for digits: {target_digits}...")

        for target in target_digits:
            if target not in self.classes:
                continue

            indices = np.where(y_test == target)[0]
            samples = X_test[indices]

            if len(samples) == 0:
                continue

            plt.figure(figsize=(10, 6))

            for sample in samples:
                sample_residuals = []
                for digit_basis in self.classes:
                    U = self.u_matrices[digit_basis]
                    U_k = U[:, :k]
                    res = self.get_residual(sample, U_k)
                    sample_residuals.append(res)

                plt.plot(self.classes.astype(int), sample_residuals,
                         color='black', alpha=0.3, linewidth=0.5)

            plt.title(f"Residuals for Test Images of Digit {target} (k={k})")
            plt.xlabel("Basis Class")
            plt.ylabel("Residual Norm")
            plt.xticks(self.classes.astype(int))
            plt.grid(True, alpha=0.3)
            plt.show()


def load_usps_file_svmlight(filename):
    """
    Helper function to load USPS data from a LIBSVM/SVMLight format file.
    This handles files formatted like: 'label 1:val 2:val ...'
    """
    print(f"Loading data from {filename} (LIBSVM format)...")
    try:
        # load_svmlight_file returns a sparse matrix
        # n_features=256 ensures we get strictly 256 columns even if some are empty
        data_sparse, labels = load_svmlight_file(filename, n_features=256)

        # Convert sparse matrix to dense numpy array
        X = data_sparse.toarray()
        y = labels.astype(int)

        # Fix labels if necessary (some datasets use 1-10 instead of 0-9)
        # Typically 10 maps to 0 in USPS if present
        if 10 in y and 0 not in y:
            print("Adjusting labels: mapped 10 to 0.")
            y[y == 10] = 0

        return X, y
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None, None


def main():
    # --- Step 1: Data Preparation ---
    # Using the SVMLight loader now
    X_train, y_train = load_usps_file_svmlight("usps")
    if X_train is None:
        return

    X_test, y_test = load_usps_file_svmlight("usps.t")
    if X_test is None:
        return

    print(f"Training Data Shape: {X_train.shape}")
    print(f"Testing Data Shape: {X_test.shape}")

    # --- Step 2: Fit Models ---
    classifier = SVDClassifier()
    classifier.fit(X_train, y_train)

    # --- Step 3: Visualization ---
    classifier.visualize_singular_images()

    # --- Step 5 (Part 1): Evaluate Performance for different k ---
    k_values = [1, 5, 10, 15, 20]
    print("\n--- Evaluation Results ---")
    for k in k_values:
        y_pred = classifier.predict(X_test, k=k)
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy with k={k}: {acc:.4f}")

    # --- Step 5 (Part 2): Residual Plots for 2, 7, 8 ---
    classifier.plot_residual_spaghetti(
        X_test, y_test, target_digits=[2, 7, 8], k=10)


if __name__ == "__main__":
    main()
