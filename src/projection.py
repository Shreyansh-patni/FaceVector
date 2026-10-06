import numpy as np


def project_faces(
    X,
    mean_face,
    eigenfaces,
    num_components
):
    """
    Project faces into the PCA/eigenface subspace.

    Formula:

        W = V_k^T (X - mean_face)

    Parameters
    ----------
    X : numpy.ndarray
        Face matrix.

    mean_face : numpy.ndarray
        Mean face vector.

    eigenfaces : numpy.ndarray
        Eigenface matrix.

    num_components : int
        Number of principal components.

    Returns
    -------
    W : numpy.ndarray
        PCA coefficient matrix.
    """

    # Select top k eigenfaces
    V = eigenfaces[:, :num_components]

    # Mean-center faces
    centered = X - mean_face[:, np.newaxis]

    # Projection
    W = V.T @ centered

    return W