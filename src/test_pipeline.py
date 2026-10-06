from data_loader import (
    load_images,
    compute_mean_face,
    center_data,
    TRAINING_DIR,
    TESTING_DIR
)

from pca_engine import compute_pca

from projection import project_faces

from recognition import recognize_face


NUM_COMPONENTS = 50


# ---------------------------------------
# Load dataset
# ---------------------------------------

X_train, y_train = load_images(TRAINING_DIR)
X_test, y_test = load_images(TESTING_DIR)

print("=" * 50)
print("FACEVECTOR PIPELINE TEST")
print("=" * 50)

print(f"Training data : {X_train.shape}")
print(f"Testing data  : {X_test.shape}")


# ---------------------------------------
# Mean face
# ---------------------------------------

mean_face = compute_mean_face(X_train)

print(f"Mean face     : {mean_face.shape}")


# ---------------------------------------
# Center data
# ---------------------------------------

P = center_data(
    X_train,
    mean_face
)

print(f"Centered data : {P.shape}")


# ---------------------------------------
# PCA
# ---------------------------------------

eigenvalues, eigenfaces = compute_pca(P)

print(f"Eigenfaces    : {eigenfaces.shape}")


# ---------------------------------------
# Projection
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

print(f"Training PCA  : {W_train.shape}")
print(f"Testing PCA   : {W_test.shape}")


# ---------------------------------------
# Recognition
# ---------------------------------------

correct = 0

for i in range(len(y_test)):

    predicted, distance, _ = recognize_face(
        W_test[:, i],
        W_train,
        y_train
    )

    if predicted == y_test[i]:
        correct += 1


accuracy = correct / len(y_test) * 100


# ---------------------------------------
# Final result
# ---------------------------------------

print("\n" + "=" * 50)
print("PIPELINE RESULT")
print("=" * 50)

print(f"Components : {NUM_COMPONENTS}")
print(f"Correct    : {correct}/{len(y_test)}")
print(f"Accuracy   : {accuracy:.2f}%")