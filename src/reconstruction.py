import numpy as np
import matplotlib.pyplot as plt

from data_loader import (
    load_images,
    compute_mean_face,
    center_data,
    TRAINING_DIR,
    TESTING_DIR
)

from pca_engine import compute_pca
from projection import project_faces


NUM_COMPONENTS = 50


def reconstruct_faces(
    X,
    mean_face,
    eigenfaces,
    num_components=50
):
    """
    Reconstruct faces from their PCA representation.

    Formula:
        X_hat = mean_face + V_k W
    """

    V = eigenfaces[:, :num_components]

    W = project_faces(
        X,
        mean_face,
        eigenfaces,
        num_components
    )

    reconstructed = mean_face[:, np.newaxis] + V @ W

    return reconstructed


if __name__ == "__main__":

    # ---------------------------------------
    # Load dataset
    # ---------------------------------------

    X_train, y_train = load_images(TRAINING_DIR)
    X_test, y_test = load_images(TESTING_DIR)

    # ---------------------------------------
    # Mean face
    # ---------------------------------------

    mean_face = compute_mean_face(X_train)

    # ---------------------------------------
    # Center training data
    # ---------------------------------------

    P = center_data(X_train, mean_face)

    # ---------------------------------------
    # PCA
    # ---------------------------------------

    eigenfaces, eigenvalues = compute_pca(P)

    # ---------------------------------------
    # Reconstruct test faces
    # ---------------------------------------

    reconstructed = reconstruct_faces(
        X_test,
        mean_face,
        eigenfaces,
        NUM_COMPONENTS
    )

    print("=" * 50)
    print("FACE RECONSTRUCTION")
    print("=" * 50)

    print(f"Original test matrix      : {X_test.shape}")
    print(f"Reconstructed test matrix : {reconstructed.shape}")
    print(f"Components used           : {NUM_COMPONENTS}")

    # ---------------------------------------
    # Visualize 3 test faces
    # ---------------------------------------

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(8, 12)
    )

    for row in range(3):

        original = X_test[:, row].reshape(112, 92)

        reconstructed_face = reconstructed[:, row].reshape(
            112,
            92
        )

        # Original
        axes[row, 0].imshow(
            original,
            cmap="gray"
        )

        axes[row, 0].set_title(
            f"Original - {y_test[row]}"
        )

        axes[row, 0].axis("off")

        # Reconstructed
        axes[row, 1].imshow(
            reconstructed_face,
            cmap="gray"
        )

        axes[row, 1].set_title(
            f"Reconstructed - {y_test[row]}"
        )

        axes[row, 1].axis("off")

    plt.suptitle(
        f"PCA Face Reconstruction ({NUM_COMPONENTS} Components)",
        fontsize=15
    )

    plt.tight_layout()

    output_path = "outputs/face_reconstruction.png"

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"\nReconstruction saved to: {output_path}"
    )