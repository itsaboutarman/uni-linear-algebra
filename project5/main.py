import numpy as np
from PIL import Image
import os

# Specify the exact name of your image file here
IMAGE_NAME = 'input_image.jpg'

# Alpha values for testing (you can modify these)
ALPHAS = [0.1, 20.0, 100.0, 5000.0]


def load_image_auto_path(filename):
    """
    Finds the script's directory and loads the image from the same folder.
    """
    # 1. Find the current directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 2. Construct the full path to the image
    file_path = os.path.join(script_dir, filename)

    print(f"--> Searching for image at: {file_path}")

    # Check if the file exists to prevent confusing errors
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Error: The file '{filename}' was not found in '{script_dir}'.")

    # 3. Load and convert to grayscale
    img = Image.open(file_path).convert('L')
    img_arr = np.array(img) / 255.0  # Normalize to [0, 1]

    return img_arr.shape, img_arr.flatten(), script_dir


def save_image(vector, shape, output_path):
    """
    Converts the vector back to an image and saves it.
    """
    # Reshape vector to matrix and scale back to [0, 255]
    img_arr = (vector.reshape(shape) * 255).astype(np.uint8)
    img = Image.fromarray(img_arr)
    img.save(output_path)


def main():
    try:
        # 1. Load image
        img_shape, x, save_dir = load_image_auto_path(IMAGE_NAME)
        n = len(x)
        k = 128  # Message length (number of bits)

        print(f"Image loaded successfully. Size: {img_shape}, Pixels: {n}")

        # 2. Generate random message and decoding matrix
        # Message consists of +1 and -1
        s = np.sign(np.random.randn(k))
        # Matrix D with dimensions k by n
        D = np.random.randn(k, n)

        # Compute D * D^T once (since it's constant and doesn't change in the loop)
        # This improves performance
        DDt = D @ D.T

        print("-" * 40)

        # 3. Loop to test different Alpha values
        for alpha in ALPHAS:
            # Mathematical formula from Part (a):
            # z = D.T * inv(DDt) * (alpha*s - Dx)

            # Calculate the Right Hand Side (RHS) of the equation
            rhs = (alpha * s) - (D @ x)

            # Solve linear system instead of explicit inverse (more stable and accurate)
            # DDt * w = rhs  => w is solved
            w = np.linalg.solve(DDt, rhs)

            # Compute final z
            z = D.T @ w

            # Construct the new image
            x_new = x + z

            # Clip values to ensure pixels remain between 0 and 1
            # This is important because displays cannot show values < 0 or > 1
            x_new_clipped = np.clip(x_new, 0, 1)

            # 4. Decode the message
            y = D @ x_new_clipped
            s_hat = np.sign(y)

            # Calculate accuracy
            accuracy = np.mean(s_hat == s) * 100
            print(f"Alpha: {alpha:6.1f} | Accuracy: {accuracy:6.2f}%")

            # 5. Save output image for visual inspection
            output_filename = f"output_alpha_{int(alpha)}.png"
            output_full_path = os.path.join(save_dir, output_filename)
            save_image(x_new_clipped, img_shape, output_full_path)

        print("-" * 40)
        print("Done! Check the folder for output images.")

    except FileNotFoundError as e:
        print("\n!!! FILE ERROR !!!")
        print(e)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
