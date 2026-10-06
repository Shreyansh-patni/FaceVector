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
        X: Face matrix
           Shape = (number_of_pixels, number_of_images)

        labels: Subject labels for each image
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

        # Parent folder is the subject, e.g. s1, s2, ...
        label = image_path.parent.name
        labels.append(label)

    # Convert list of vectors into matrix
    X = np.column_stack(images)

    return X, labels


if __name__ == "__main__":

    X_train, y_train = load_images(TRAINING_DIR)

    X_test, y_test = load_images(TESTING_DIR)

    print("=" * 50)
    print("FACE MATRIX")
    print("=" * 50)

    print(f"Training matrix shape : {X_train.shape}")
    print(f"Testing matrix shape  : {X_test.shape}")

    print(f"\nTraining labels       : {len(y_train)}")
    print(f"Testing labels        : {len(y_test)}")

    print(f"\nPixels per image      : {X_train.shape[0]}")
    print(f"Training images       : {X_train.shape[1]}")
    print(f"Testing images        : {X_test.shape[1]}")

    print("\nFirst training label  :", y_train[0])
    print("First testing label   :", y_test[0])