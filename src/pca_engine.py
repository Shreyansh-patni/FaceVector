import numpy as np


def compute_pca(P):
    """
    Compute PCA using the smaller matrix P.T @ P.

    P shape:
        (number_of_pixels, number_of_training_images)

    Example:
        (10304, 360)

    The eigenvectors of P.T @ P are converted into
    eigenfaces using:

        V = P @ U

    Result:
        eigenfaces shape = (10304, 359)
    """

    # --------------------------------------------------------
    # Step 1: Compute the smaller covariance-related matrix
    # --------------------------------------------------------

    C = P.T @ P

    # --------------------------------------------------------
    # Step 2: Eigenvalue decomposition
    # --------------------------------------------------------

    eigenvalues, eigenvectors = np.linalg.eigh(C)

    # --------------------------------------------------------
    # Step 3: Sort eigenvalues from largest to smallest
    # --------------------------------------------------------

    indices = np.argsort(
        eigenvalues
    )[::-1]

    eigenvalues = eigenvalues[
        indices
    ]

    eigenvectors = eigenvectors[
        :,
        indices
    ]

    # --------------------------------------------------------
    # Step 4: Remove zero / near-zero eigenvalues
    # --------------------------------------------------------

    threshold = 1e-10

    valid = eigenvalues > threshold

    eigenvalues = eigenvalues[
        valid
    ]

    eigenvectors = eigenvectors[
        :,
        valid
    ]

    # --------------------------------------------------------
    # Step 5: Convert eigenvectors into eigenfaces
    #
    # U = eigenvectors of P.T @ P
    #
    # V = P @ U
    # --------------------------------------------------------

    eigenfaces = P @ eigenvectors

    # --------------------------------------------------------
    # Step 6: Normalize eigenfaces
    # --------------------------------------------------------

    norms = np.linalg.norm(
        eigenfaces,
        axis=0
    )

    valid_norms = norms > 1e-12

    eigenfaces = eigenfaces[
        :,
        valid_norms
    ]

    eigenvalues = eigenvalues[
        valid_norms
    ]

    eigenfaces = (
        eigenfaces /
        np.linalg.norm(
            eigenfaces,
            axis=0,
            keepdims=True
        )
    )

    # --------------------------------------------------------
    # Return
    # --------------------------------------------------------

    return eigenfaces, eigenvalues