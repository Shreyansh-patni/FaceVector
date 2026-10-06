import numpy as np


def euclidean_distance(a, b):
    """
    Calculate Euclidean distance between two vectors.
    """

    return np.linalg.norm(a - b)


def recognize_face(
    test_signature,
    training_signatures,
    training_labels
):
    """
    Find the training face with the smallest
    Euclidean distance.

    Parameters
    ----------
    test_signature : numpy.ndarray
        PCA representation of the test face.

    training_signatures : numpy.ndarray
        PCA representations of training faces.

    training_labels : list
        Training subject labels.

    Returns
    -------
    predicted_label : str
    minimum_distance : float
    distances : numpy.ndarray
    """

    distances = np.linalg.norm(
        training_signatures
        - test_signature[:, np.newaxis],
        axis=0
    )

    best_index = np.argmin(distances)

    predicted_label = training_labels[best_index]

    minimum_distance = distances[best_index]

    return (
        predicted_label,
        minimum_distance,
        distances
    )