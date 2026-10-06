import numpy as np

from data_loader import (
    load_images,
    compute_mean_face,
    center_data,
    TRAINING_DIR
)

from pca_engine import compute_pca


# Load training data
X_train, labels = load_images(TRAINING_DIR)

# Mean face
mean_face = compute_mean_face(X_train)

# Center data
P = center_data(X_train, mean_face)

# Compute eigenfaces
eigenfaces, eigenvalues = compute_pca(P)


# ---------------------------------------
# Orthogonality check
# ---------------------------------------

orthogonality_matrix = eigenfaces.T @ eigenfaces

identity_matrix = np.eye(eigenfaces.shape[1])

error = np.max(
    np.abs(orthogonality_matrix - identity_matrix)
)


print("=" * 50)
print("EIGENFACE ORTHOGONALITY CHECK")
print("=" * 50)

print(f"Eigenfaces shape : {eigenfaces.shape}")

print(
    f"\nMaximum deviation from identity matrix: "
    f"{error:.10f}"
)

print("\nTop-left 5 x 5 section of V^T V:")
print(
    np.round(
        orthogonality_matrix[:5, :5],
        4
    )
)

if error < 1e-5:
    print("\nRESULT: Eigenfaces are approximately orthonormal.")
else:
    print("\nRESULT: Check requires further investigation.")