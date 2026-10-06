import numpy as np
import matplotlib.pyplot as plt

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
# Explained variance
# ---------------------------------------

total_variance = np.sum(eigenvalues)

explained_variance_ratio = (
    eigenvalues / total_variance
)

cumulative_variance = np.cumsum(
    explained_variance_ratio
)


print("=" * 50)
print("EXPLAINED VARIANCE ANALYSIS")
print("=" * 50)

print(f"Total principal components : {len(eigenvalues)}")

for k in [10, 25, 50, 100, 150, 200, 250, 300]:
    if k <= len(cumulative_variance):

        percentage = cumulative_variance[k - 1] * 100

        print(
            f"Top {k:3d} components "
            f"→ {percentage:.2f}% variance"
        )


# ---------------------------------------
# Find components for common thresholds
# ---------------------------------------

for threshold in [0.80, 0.90, 0.95, 0.99]:

    components_needed = (
        np.argmax(
            cumulative_variance >= threshold
        ) + 1
    )

    print(
        f"{threshold * 100:.0f}% variance "
        f"→ {components_needed} components"
    )


# ---------------------------------------
# Plot cumulative explained variance
# ---------------------------------------

plt.figure(figsize=(10, 6))

plt.plot(
    range(1, len(cumulative_variance) + 1),
    cumulative_variance * 100
)

plt.axvline(
    50,
    linestyle="--",
    label="50 Components"
)

plt.axhline(
    95,
    linestyle="--",
    label="95% Variance"
)

plt.xlabel("Number of Principal Components")

plt.ylabel("Cumulative Explained Variance (%)")

plt.title(
    "PCA Cumulative Explained Variance"
)

plt.grid(True)

plt.legend()

plt.tight_layout()

output_path = "outputs/explained_variance.png"

plt.savefig(
    output_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print(
    f"\nPlot saved to: {output_path}"
)