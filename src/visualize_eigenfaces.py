import matplotlib.pyplot as plt
import numpy as np

from data_loader import (
    load_images,
    compute_mean_face,
    center_data,
    TRAINING_DIR
)

from pca_engine import compute_pca


# ---------------------------------------
# Load training data
# ---------------------------------------

X_train, labels = load_images(TRAINING_DIR)

# ---------------------------------------
# Mean face
# ---------------------------------------

mean_face = compute_mean_face(X_train)

# ---------------------------------------
# Mean-center
# ---------------------------------------

P = center_data(X_train, mean_face)

# ---------------------------------------
# PCA
# ---------------------------------------

eigenfaces, eigenvalues = compute_pca(P)


# ---------------------------------------
# Visualize first 9 eigenfaces
# ---------------------------------------

fig, axes = plt.subplots(3, 3, figsize=(10, 12))

for i, ax in enumerate(axes.flat):

    face = eigenfaces[:, i]

    # Convert vector back to image
    face_image = face.reshape(112, 92)

    ax.imshow(face_image, cmap="gray")
    ax.set_title(f"Eigenface {i + 1}")
    ax.axis("off")


plt.suptitle(
    "Top 9 Eigenfaces",
    fontsize=16
)

plt.tight_layout()

# Save figure
output_path = "outputs/eigenfaces.png"

plt.savefig(
    output_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print(f"\nEigenfaces visualization saved to: {output_path}")