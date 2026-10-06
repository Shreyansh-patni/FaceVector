import numpy as np

from data_loader import load_images, compute_mean_face, center_data
from data_loader import TRAINING_DIR


def compute_pca(P):
    """
    Compute eigenvalues and eigenvectors using P^T P.

    P shape:
        (pixels, training_images)

    Returns:
        eigenvalues
        eigenvectors
        eigenfaces
    """

    # ---------------------------------------
    # Step 1: Compute the smaller covariance
    # ---------------------------------------

    C = P.T @ P

    print("=" * 50)
    print("PCA / EIGENFACE CALCULATION")
    print("=" * 50)

    print(f"P shape       : {P.shape}")
    print(f"P^T P shape   : {C.shape}")

    # ---------------------------------------
    # Step 2: Eigenvalue/eigenvector analysis
    # ---------------------------------------

    eigenvalues, eigenvectors = np.linalg.eigh(C)

    # ---------------------------------------
    # Step 3: Sort eigenvalues from largest
    # to smallest
    # ---------------------------------------

    indices = np.argsort(eigenvalues)[::-1]

    eigenvalues = eigenvalues[indices]
    eigenvectors = eigenvectors[:, indices]

    # ---------------------------------------
    # Step 4: Convert eigenvectors of P^T P
    # into eigenvectors of P P^T
    # ---------------------------------------

    eigenfaces = P @ eigenvectors

    # ---------------------------------------
    # Step 5: Normalize eigenfaces
    # ---------------------------------------

    norms = np.linalg.norm(eigenfaces, axis=0)

    valid = norms > 1e-10

    eigenfaces = eigenfaces[:, valid]
    eigenvalues = eigenvalues[valid]

    eigenfaces = eigenfaces / np.linalg.norm(
        eigenfaces,
        axis=0,
        keepdims=True
    )

    return eigenvalues, eigenfaces


if __name__ == "__main__":

    # Load training faces
    X_train, y_train = load_images(TRAINING_DIR)

    # Compute mean face
    mean_face = compute_mean_face(X_train)

    # Center the data
    P = center_data(X_train, mean_face)

    # PCA
    eigenvalues, eigenfaces = compute_pca(P)

    print("\nEigenvalues:")
    print(eigenvalues[:10])

    print("\nEigenfaces matrix shape:")
    print(eigenfaces.shape)

    print("\nLargest eigenvalue:")
    print(eigenvalues[0])

    print("\nNumber of eigenfaces:")
    print(eigenfaces.shape[1])