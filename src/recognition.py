import numpy as np

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


def euclidean_distance(a, b):
    """Calculate Euclidean distance between two vectors."""
    return np.linalg.norm(a - b)


def recognize_face(
    test_signature,
    training_signatures,
    training_labels
):
    """
    Find the training face with the smallest
    Euclidean distance from the test face.
    """

    distances = np.linalg.norm(
        training_signatures - test_signature[:, np.newaxis],
        axis=0
    )

    best_index = np.argmin(distances)

    predicted_label = training_labels[best_index]
    minimum_distance = distances[best_index]

    return predicted_label, minimum_distance, distances


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

    eigenvalues, eigenfaces = compute_pca(P)

    # ---------------------------------------
    # Project training and testing faces
    # ---------------------------------------

    W_train = project_faces(
        X_train,
        mean_face,
        eigenfaces,
        NUM_COMPONENTS
    )

    W_test = project_faces(
        X_test,
        mean_face,
        eigenfaces,
        NUM_COMPONENTS
    )

    # ---------------------------------------
    # Recognize testing faces
    # ---------------------------------------

    correct = 0

    print("\n" + "=" * 60)
    print("FACE RECOGNITION RESULTS")
    print("=" * 60)

    for i in range(len(y_test)):

        test_signature = W_test[:, i]

        predicted_label, distance, distances = recognize_face(
            test_signature,
            W_train,
            y_train
        )

        actual_label = y_test[i]

        if predicted_label == actual_label:
            correct += 1

        print(
            f"Test {i + 1:02d}: "
            f"Actual = {actual_label:>3} | "
            f"Predicted = {predicted_label:>3} | "
            f"Distance = {distance:.2f}"
        )

    # ---------------------------------------
    # Accuracy
    # ---------------------------------------

    accuracy = (correct / len(y_test)) * 100

    print("\n" + "=" * 60)
    print("FINAL RESULT")
    print("=" * 60)

    print(f"Correct predictions : {correct}/{len(y_test)}")
    print(f"Accuracy            : {accuracy:.2f}%")