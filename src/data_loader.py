from pathlib import Path
import numpy as np
from PIL import Image


BASE_DIR = Path(__file__).resolve().parent.parent

TRAINING_DIR = BASE_DIR / "data" / "att_faces" / "Training"
TESTING_DIR = BASE_DIR / "data" / "att_faces" / "Testing"


def load_images(folder):
    """
    Load all .pgm face images from a folder.

    Returns:
        X:
            Face matrix.
            Shape = (number_of_pixels, number_of_images)

        labels:
            Subject label for each image.
    """

    image_files = sorted(folder.rglob("*.pgm"))

    images = []
    labels = []

    for image_path in image_files:

        with Image.open(image_path) as img:
            img = img.convert("L")
            image_array = np.array(img, dtype=np.float64)

        # Convert 92 x 112 image into a 1D vector
        vector = image_array.flatten()

        images.append(vector)

        # Parent folder = subject
        label = image_path.parent.name
        labels.append(label)

    X = np.column_stack(images)

    return X, labels


def compute_mean_face(X):
    """
    Calculate the mean face.

    X shape:
        (pixels, images)

    Returns:
        mean_face shape:
        (pixels,)
    """

    mean_face = np.mean(X, axis=1)

    return mean_face


def center_data(X, mean_face):
    """
    Subtract the mean face from every face.
    """

    centered = X - mean_face[:, np.newaxis]

    return centered


if __name__ == "__main__":

    # -----------------------------
    # Load dataset
    # -----------------------------

    X_train, y_train = load_images(TRAINING_DIR)

    X_test, y_test = load_images(TESTING_DIR)

    print("=" * 50)
    print("FACE MATRIX")
    print("=" * 50)

    print(f"Training matrix shape : {X_train.shape}")
    print(f"Testing matrix shape  : {X_test.shape}")

    # -----------------------------
    # Mean face
    # -----------------------------

    mean_face = compute_mean_face(X_train)

    print("\n" + "=" * 50)
    print("MEAN FACE")
    print("=" * 50)

    print(f"Mean face shape       : {mean_face.shape}")
    print(f"Mean pixel value      : {mean_face.mean():.2f}")
    print(f"Minimum mean value    : {mean_face.min():.2f}")
    print(f"Maximum mean value    : {mean_face.max():.2f}")

    # -----------------------------
    # Mean centering
    # -----------------------------

    P = center_data(X_train, mean_face)

    print("\n" + "=" * 50)
    print("MEAN CENTERING")
    print("=" * 50)

    print(f"Centered matrix shape : {P.shape}")

    # Check that the mean of each pixel row
    # is approximately zero after centering.
    row_means = np.mean(P, axis=1)

    print(f"Maximum absolute row mean after centering: "
          f"{np.max(np.abs(row_means)):.10f}")