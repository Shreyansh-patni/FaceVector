import numpy as np

from data_loader import (
    load_images,
    compute_mean_face,
    center_data,
    TRAINING_DIR,
    TESTING_DIR
)

from pca_engine import compute_pca


# Number of principal components
NUM_COMPONENTS = 50


def project_faces(X, mean_face, eigenfaces, num_components=50):
    """
    Project faces into the PCA/eigenface subspace.

    Formula:
        W = V_k^T (X - mean_face)

    Returns:
        W = PCA coefficients
    """

    # Select top k eigenfaces
    V = eigenfaces[:, :num_components]

    # Mean-center the faces
    centered = X - mean_face[:, np.newaxis]

    # Project into PCA subspace
    W = V.T @ centered

    return W


if __name__ == "__main__":

    # ---------------------------------------
    # Load training and testing data
    # ---------------------------------------

    X_train, y_train = load_images(TRAINING_DIR)
    X_test, y_test = load_images(TESTING_DIR)

    # ---------------------------------------
    # Calculate mean face
    # ---------------------------------------

    mean_face = compute_mean_face(X_train)

    # ---------------------------------------
    # Center training data
    # ---------------------------------------

    P = center_data(X_train, mean_face)

    # ---------------------------------------
    # Calculate PCA
    # ---------------------------------------

    eigenvalues, eigenfaces = compute_pca(P)

    # ---------------------------------------
    # Project training faces
    # ---------------------------------------

    W_train = project_faces(
        X_train,
        mean_face,
        eigenfaces,
        NUM_COMPONENTS
    )

    # ---------------------------------------
    # Project testing faces
    # ---------------------------------------

    W_test = project_faces(
        X_test,
        mean_face,
        eigenfaces,
        NUM_COMPONENTS
    )

    # ---------------------------------------
    # Display results
    # ---------------------------------------

    print("\n" + "=" * 50)
    print("PCA PROJECTION")
    print("=" * 50)

    print(f"Original face dimensions : {X_train.shape[0]}")

    print(f"Number of components     : {NUM_COMPONENTS}")

    print(f"\nTraining projection shape: {W_train.shape}")

    print(f"Testing projection shape : {W_test.shape}")

    print(
        f"\nCompression: "
        f"{X_train.shape[0]} → {NUM_COMPONENTS} dimensions"
    )

    print("\nFirst training face signature:")
    print(W_train[:, 0])