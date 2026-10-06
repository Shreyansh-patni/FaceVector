import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from src.data_loader import (
    load_images,
    compute_mean_face,
    center_data,
    TRAINING_DIR,
    TESTING_DIR,
)

from src.pca_engine import compute_pca
from src.projection import project_faces
from src.recognition import recognize_face


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="FaceVector | PCA Face Recognition",
    page_icon="🧠",
    layout="wide",
)


# ============================================================
# CONSTANTS
# ============================================================

NUM_COMPONENTS = 50

IMAGE_HEIGHT = 112
IMAGE_WIDTH = 92


# ============================================================
# IMAGE DISPLAY FUNCTIONS
# ============================================================

def prepare_display_image(face_vector):
    """
    Convert a face vector into a valid 8-bit grayscale image.

    Used only for visualization.
    Does not affect PCA calculations.
    """

    image = np.asarray(face_vector).reshape(
        IMAGE_HEIGHT,
        IMAGE_WIDTH
    )

    image = np.clip(
        image,
        0,
        255
    )

    return image.astype(np.uint8)


def prepare_eigenface_image(eigenface):
    """
    Normalize an eigenface for visualization.

    Eigenfaces contain positive and negative values.
    Min-max normalization makes their structure visible.

    This affects visualization only.
    """

    image = np.asarray(eigenface).reshape(
        IMAGE_HEIGHT,
        IMAGE_WIDTH
    )

    minimum = image.min()
    maximum = image.max()

    if maximum - minimum < 1e-12:
        return np.zeros_like(
            image,
            dtype=np.uint8
        )

    normalized = (
        (image - minimum)
        /
        (maximum - minimum)
        * 255
    )

    return normalized.astype(
        np.uint8
    )


# ============================================================
# RECOGNITION COMPATIBILITY
# ============================================================

def run_recognition(
    test_projection,
    training_projections,
    training_labels
):
    """
    Supports recognize_face() returning either:

        predicted_label, distance

    or:

        predicted_label, distance, index
    """

    result = recognize_face(
        test_projection,
        training_projections,
        training_labels
    )

    if len(result) == 2:

        predicted_label, distance = result

    elif len(result) == 3:

        predicted_label, distance, _ = result

    else:

        raise ValueError(
            "recognize_face() returned an unexpected "
            f"number of values: {len(result)}"
        )

    return predicted_label, distance


# ============================================================
# PREPARE MODEL
# ============================================================

@st.cache_data
def prepare_model():

    # --------------------------------------------------------
    # LOAD DATASET
    # --------------------------------------------------------

    X_train, y_train = load_images(
        TRAINING_DIR
    )

    X_test, y_test = load_images(
        TESTING_DIR
    )


    # --------------------------------------------------------
    # MEAN FACE
    # --------------------------------------------------------

    mean_face = compute_mean_face(
        X_train
    )


    # --------------------------------------------------------
    # CENTER TRAINING DATA
    # --------------------------------------------------------

    P = center_data(
        X_train,
        mean_face
    )


    # --------------------------------------------------------
    # GRAM MATRIX
    # --------------------------------------------------------

    # P has shape:
    # 10304 × 360
    #
    # P.T @ P therefore has shape:
    # 360 × 360

    gram_matrix = P.T @ P


    # --------------------------------------------------------
    # PCA
    # --------------------------------------------------------

    eigenfaces, eigenvalues = compute_pca(
        P
    )

    eigenfaces = np.asarray(
        eigenfaces
    )

    eigenvalues = np.asarray(
        eigenvalues
    ).flatten()


    # Safety conversion
    if eigenfaces.ndim == 1:

        eigenfaces = eigenfaces.reshape(
            -1,
            1
        )


    # --------------------------------------------------------
    # SELECT COMPONENTS
    # --------------------------------------------------------

    k = min(
        NUM_COMPONENTS,
        eigenfaces.shape[1]
    )


    # --------------------------------------------------------
    # SELECT PRINCIPAL COMPONENTS
    # --------------------------------------------------------

    eigenfaces_k = eigenfaces[
        :,
        :k
    ]


    # --------------------------------------------------------
    # ORTHOGONALITY CHECK
    # --------------------------------------------------------

    orthogonality_matrix = (
        eigenfaces_k.T
        @
        eigenfaces_k
    )

    identity_matrix = np.eye(k)

    orthogonality_error = np.max(
        np.abs(
            orthogonality_matrix
            -
            identity_matrix
        )
    )

    is_orthogonal = np.allclose(
        orthogonality_matrix,
        identity_matrix,
        atol=1e-6
    )


    # --------------------------------------------------------
    # PROJECT TRAINING DATA
    # --------------------------------------------------------

    W_train = project_faces(
        X_train,
        mean_face,
        eigenfaces,
        k
    )


    # --------------------------------------------------------
    # PROJECT TESTING DATA
    # --------------------------------------------------------

    W_test = project_faces(
        X_test,
        mean_face,
        eigenfaces,
        k
    )


    # --------------------------------------------------------
    # FACE RECOGNITION
    # --------------------------------------------------------

    predictions = []
    distances = []

    for i in range(
        W_test.shape[1]
    ):

        predicted_label, distance = run_recognition(
            W_test[:, i],
            W_train,
            y_train
        )

        predictions.append(
            predicted_label
        )

        distances.append(
            distance
        )


    predictions = np.asarray(
        predictions
    )

    distances = np.asarray(
        distances
    )


    # --------------------------------------------------------
    # ACCURACY
    # --------------------------------------------------------

    correct = np.sum(
        predictions == y_test
    )

    accuracy = (
        correct /
        len(y_test)
    )


    # --------------------------------------------------------
    # EXPLAINED VARIANCE
    # --------------------------------------------------------

    positive_eigenvalues = eigenvalues[
        eigenvalues > 1e-12
    ]


    if len(
        positive_eigenvalues
    ) > 0:

        total_variance = np.sum(
            positive_eigenvalues
        )

        explained_variance = (
            positive_eigenvalues[:k]
            /
            total_variance
        )

        cumulative_variance = np.cumsum(
            explained_variance
        )

    else:

        explained_variance = np.zeros(
            k
        )

        cumulative_variance = np.zeros(
            k
        )


    # --------------------------------------------------------
    # RETURN MODEL
    # --------------------------------------------------------

    return {

        "X_train": X_train,
        "y_train": y_train,

        "X_test": X_test,
        "y_test": y_test,

        "mean_face": mean_face,

        "P": P,
        "gram_matrix": gram_matrix,

        "eigenfaces": eigenfaces,
        "eigenvalues": eigenvalues,

        "eigenfaces_k": eigenfaces_k,

        "orthogonality_matrix":
            orthogonality_matrix,

        "orthogonality_error":
            orthogonality_error,

        "is_orthogonal":
            is_orthogonal,

        "W_train": W_train,
        "W_test": W_test,

        "predictions": predictions,
        "distances": distances,

        "correct": correct,
        "accuracy": accuracy,

        "k": k,

        "explained_variance":
            explained_variance,

        "cumulative_variance":
            cumulative_variance,
    }


# ============================================================
# LOAD MODEL
# ============================================================

with st.spinner(
    "Loading FaceVector model..."
):

    model = prepare_model()


# ============================================================
# EXTRACT MODEL DATA
# ============================================================

X_train = model["X_train"]
y_train = model["y_train"]

X_test = model["X_test"]
y_test = model["y_test"]

mean_face = model["mean_face"]

P = model["P"]
gram_matrix = model["gram_matrix"]

eigenfaces = model["eigenfaces"]
eigenvalues = model["eigenvalues"]

eigenfaces_k = model["eigenfaces_k"]

orthogonality_matrix = (
    model["orthogonality_matrix"]
)

orthogonality_error = (
    model["orthogonality_error"]
)

is_orthogonal = (
    model["is_orthogonal"]
)

W_train = model["W_train"]
W_test = model["W_test"]

predictions = model["predictions"]
distances = model["distances"]

correct = model["correct"]
accuracy = model["accuracy"]

k = model["k"]

cumulative_variance = (
    model["cumulative_variance"]
)


# ============================================================
# HEADER
# ============================================================

st.title(
    "🧠 FaceVector"
)

st.subheader(
    "Eigenvector & PCA-Based Face Recognition System"
)

st.write(
    "A Linear Algebra based face recognition system "
    "using Principal Component Analysis (PCA), "
    "eigenvalues, eigenvectors and projections."
)


# ============================================================
# PROJECT OVERVIEW
# ============================================================

st.markdown("---")

st.header(
    "📊 Project Overview"
)

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Training Images",
        X_train.shape[1]
    )


with col2:

    st.metric(
        "Testing Images",
        X_test.shape[1]
    )


with col3:

    st.metric(
        "Image Size",
        f"{IMAGE_WIDTH} × {IMAGE_HEIGHT}"
    )


with col4:

    st.metric(
        "Recognition Accuracy",
        f"{accuracy * 100:.2f}%"
    )


# ============================================================
# MATRIX REPRESENTATION
# ============================================================

st.markdown("---")

st.header(
    "📐 Matrix Representation"
)

st.write(
    "Each grayscale face image is converted into a "
    "column vector and the vectors are combined "
    "to form the face data matrix."
)


col1, col2 = st.columns(2)


with col1:

    st.subheader(
        "Training Matrix"
    )

    st.code(
        f"X_train = {X_train.shape}"
    )

    st.write(
        f"Pixels per image: "
        f"{X_train.shape[0]:,}"
    )

    st.write(
        f"Training images: "
        f"{X_train.shape[1]}"
    )


with col2:

    st.subheader(
        "Testing Matrix"
    )

    st.code(
        f"X_test = {X_test.shape}"
    )

    st.write(
        f"Testing images: "
        f"{X_test.shape[1]}"
    )


# ============================================================
# MEAN FACE
# ============================================================

st.markdown("---")

st.header(
    "🖼️ Mean Face"
)

st.write(
    "The mean face is calculated by averaging "
    "all training face vectors."
)


col1, col2, col3 = st.columns(3)


with col1:

    st.image(
        prepare_display_image(
            mean_face
        ),
        caption="Mean Face",
        width=250
    )


with col2:

    st.code(
        "μ = (1/N) Σ Xᵢ"
    )

    st.write(
        "The mean face provides the reference "
        "for centering the training data."
    )


with col3:

    st.metric(
        "Mean Face Dimensions",
        f"{mean_face.shape[0]:,}"
    )

    st.write(
        "Each pixel represents the average "
        "intensity at that position."
    )


# ============================================================
# MATHEMATICAL PIPELINE
# ============================================================

st.markdown("---")

st.header(
    "🧮 Mathematical Pipeline"
)

st.markdown(
    """
**Image → Vector → Mean Face → Mean Centering → PᵀP
→ Eigenvalues & Eigenvectors → Eigenfaces → Projection
→ Euclidean Distance → Face Recognition → Reconstruction**
"""
)

st.info(
    "The smaller PᵀP matrix is used because it is "
    "much smaller than PPᵀ for this dataset."
)


# ============================================================
# CENTERING AND GRAM MATRIX
# ============================================================

st.markdown("---")

st.header(
    "🔢 Mean Centering & PᵀP"
)

col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Centered Matrix P",
        f"{P.shape[0]:,} × {P.shape[1]}"
    )


with col2:

    st.metric(
        "PᵀP Matrix",
        f"{gram_matrix.shape[0]} × "
        f"{gram_matrix.shape[1]}"
    )


with col3:

    st.metric(
        "PPᵀ Would Be",
        f"{P.shape[0]:,} × "
        f"{P.shape[0]:,}"
    )


st.code(
    "P = X - μ\n"
    "G = PᵀP"
)


st.write(
    f"Our centered matrix contains "
    f"{P.shape[0]:,} pixel dimensions and "
    f"{P.shape[1]} training images."
)

st.write(
    f"Instead of computing a "
    f"{P.shape[0]:,} × {P.shape[0]:,} "
    f"matrix, we solve the eigenproblem on "
    f"the smaller "
    f"{gram_matrix.shape[0]} × "
    f"{gram_matrix.shape[1]} matrix."
)


# ============================================================
# PCA ANALYSIS
# ============================================================

st.markdown("---")

st.header(
    "🔬 PCA Analysis"
)

col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Original Dimensions",
        f"{X_train.shape[0]:,}"
    )


with col2:

    st.metric(
        "PCA Components",
        k
    )


with col3:

    st.metric(
        "Dimensionality Reduction",
        f"{X_train.shape[0]:,} → {k}"
    )


if len(
    cumulative_variance
) > 0:

    variance_retained = (
        cumulative_variance[-1]
        * 100
    )

    st.info(
        f"The first **{k} components** retain "
        f"approximately **{variance_retained:.2f}%** "
        f"of the variance."
    )


# ============================================================
# EIGENVALUE ANALYSIS
# ============================================================

st.markdown("---")

st.header(
    "📊 Eigenvalue Analysis"
)

st.write(
    "Eigenvalues indicate the amount of variation "
    "captured by the corresponding principal directions."
)


num_eigenvalues = min(
    10,
    len(eigenvalues)
)


eigenvalue_data = []

for i in range(
    num_eigenvalues
):

    eigenvalue_data.append(
        {
            "Component":
                i + 1,

            "Eigenvalue":
                f"{eigenvalues[i]:.4f}"
        }
    )


st.dataframe(
    eigenvalue_data,
    use_container_width=True,
    hide_index=True
)


# Eigenvalue spectrum

positive_values = eigenvalues[
    eigenvalues > 1e-12
]


if len(
    positive_values
) > 0:

    components_all = np.arange(
        1,
        len(positive_values) + 1
    )


    fig_eigen, ax_eigen = plt.subplots(
        figsize=(10, 5)
    )


    ax_eigen.plot(
        components_all,
        positive_values
    )


    ax_eigen.set_xlabel(
        "Principal Component"
    )


    ax_eigen.set_ylabel(
        "Eigenvalue"
    )


    ax_eigen.set_title(
        "Eigenvalue Spectrum"
    )


    ax_eigen.grid(
        True,
        alpha=0.3
    )


    st.pyplot(
        fig_eigen,
        clear_figure=True
    )


# ============================================================
# ORTHOGONALITY
# ============================================================

st.markdown("---")

st.header(
    "📐 Eigenface Orthogonality"
)

st.write(
    "Because the selected eigenfaces are normalized, "
    "their dot products should form an identity matrix."
)


col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Eigenfaces Checked",
        k
    )


with col2:

    st.metric(
        "Maximum Deviation",
        f"{orthogonality_error:.10f}"
    )


with col3:

    if is_orthogonal:

        st.success(
            "✓ Approximately Orthonormal"
        )

    else:

        st.error(
            "✗ Orthogonality Check Failed"
        )


st.code(
    "VᵀV ≈ I"
)

st.write(
    "The maximum deviation measures how far "
    "the computed VᵀV matrix is from the identity matrix."
)


# ============================================================
# FACE RECOGNITION
# ============================================================

st.markdown("---")

st.header(
    "👤 Face Recognition"
)

selected_index = st.selectbox(
    "Select a test face",
    range(
        len(y_test)
    ),
    format_func=lambda x:
        f"Test Image {x + 1} | "
        f"Actual: {y_test[x]}"
)


test_vector = X_test[
    :,
    selected_index
]

actual_label = y_test[
    selected_index
]

predicted_label = predictions[
    selected_index
]

distance = distances[
    selected_index
]


# ============================================================
# RECOGNITION RESULT
# ============================================================

col1, col2, col3 = st.columns(3)


with col1:

    st.subheader(
        "Original Test Face"
    )

    st.image(
        prepare_display_image(
            test_vector
        ),
        caption=f"Actual: {actual_label}",
        width=250
    )


with col2:

    st.subheader(
        "Recognition Result"
    )

    if (
        predicted_label
        ==
        actual_label
    ):

        st.success(
            "✅ Correct Recognition"
        )

    else:

        st.error(
            "❌ Incorrect Recognition"
        )


    st.metric(
        "Predicted Subject",
        predicted_label
    )

    st.metric(
        "Actual Subject",
        actual_label
    )

    st.metric(
        "Euclidean Distance",
        f"{distance:.4f}"
    )


with col3:

    st.subheader(
        "PCA Representation"
    )

    st.write(
        "Original dimensions"
    )

    st.code(
        f"{X_test.shape[0]:,}"
    )

    st.write(
        "Reduced dimensions"
    )

    st.code(
        f"{k}"
    )

    st.write(
        "Query projection shape"
    )

    st.code(
        f"({k},)"
    )

    st.write(
        "Training PCA shape"
    )

    st.code(
        f"{W_train.shape}"
    )


# ============================================================
# FACE RECONSTRUCTION
# ============================================================

st.markdown("---")

st.header(
    "🔄 Face Reconstruction"
)

st.write(
    "The selected test face is reconstructed "
    "from its PCA representation."
)


W_selected = W_test[
    :,
    selected_index
]


reconstructed = (
    mean_face
    +
    eigenfaces_k @ W_selected
)


col1, col2, col3 = st.columns(3)


with col1:

    st.image(
        prepare_display_image(
            test_vector
        ),
        caption="Original Face",
        width=250
    )


with col2:

    st.image(
        prepare_display_image(
            reconstructed
        ),
        caption=f"Reconstructed ({k} components)",
        width=250
    )


with col3:

    reconstruction_error = np.linalg.norm(
        test_vector -
        reconstructed
    )

    st.metric(
        "Reconstruction Error",
        f"{reconstruction_error:.2f}"
    )

    st.write(
        "The reconstruction is an approximation "
        "because only the selected principal "
        "components are used."
    )


# ============================================================
# EIGENFACES
# ============================================================

st.markdown("---")

st.header(
    "🧩 Eigenfaces"
)

st.write(
    "Eigenfaces are the principal components "
    "learned from the training face dataset."
)

st.info(
    "Eigenfaces are normalized only for visualization. "
    "The original PCA values remain unchanged."
)


num_to_show = min(
    10,
    eigenfaces.shape[1]
)


fig, axes = plt.subplots(
    2,
    5,
    figsize=(12, 5)
)

axes = axes.flatten()


for i in range(10):

    axes[i].axis(
        "off"
    )

    if i < num_to_show:

        eigenface_image = (
            prepare_eigenface_image(
                eigenfaces[:, i]
            )
        )

        axes[i].imshow(
            eigenface_image,
            cmap="gray"
        )

        axes[i].set_title(
            f"Eigenface {i + 1}"
        )


plt.tight_layout()

st.pyplot(
    fig,
    clear_figure=True
)


# ============================================================
# EXPLAINED VARIANCE
# ============================================================

st.markdown("---")

st.header(
    "📈 Explained Variance"
)

st.write(
    "Cumulative explained variance shows how much "
    "of the dataset's variation is captured by "
    "the principal components."
)


if len(
    cumulative_variance
) > 0:

    components = np.arange(
        1,
        len(cumulative_variance) + 1
    )


    fig_variance, ax = plt.subplots(
        figsize=(10, 5)
    )


    ax.plot(
        components,
        cumulative_variance * 100
    )


    ax.axhline(
        80,
        linestyle="--",
        label="80%"
    )


    ax.axhline(
        90,
        linestyle="--",
        label="90%"
    )


    ax.axhline(
        95,
        linestyle="--",
        label="95%"
    )


    ax.set_xlabel(
        "Number of Principal Components"
    )


    ax.set_ylabel(
        "Cumulative Explained Variance (%)"
    )


    ax.set_title(
        "PCA Explained Variance"
    )


    ax.grid(
        True,
        alpha=0.3
    )


    ax.legend()


    st.pyplot(
        fig_variance,
        clear_figure=True
    )


# ============================================================
# RECOGNITION PERFORMANCE
# ============================================================

st.markdown("---")

st.header(
    "🎯 Recognition Performance"
)

col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Correct Predictions",
        f"{correct}/{len(y_test)}"
    )


with col2:

    st.metric(
        "Incorrect Predictions",
        f"{len(y_test) - correct}/{len(y_test)}"
    )


with col3:

    st.metric(
        "Accuracy",
        f"{accuracy * 100:.2f}%"
    )


# ============================================================
# PREDICTION TABLE
# ============================================================

st.subheader(
    "Test Set Predictions"
)

prediction_data = []


for i in range(
    len(y_test)
):

    result = (
        "Correct"
        if y_test[i]
        ==
        predictions[i]
        else
        "Incorrect"
    )


    prediction_data.append(
        {
            "Test Image":
                i + 1,

            "Actual":
                y_test[i],

            "Predicted":
                predictions[i],

            "Distance":
                f"{distances[i]:.4f}",

            "Result":
                result,
        }
    )


st.dataframe(
    prediction_data,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# FINAL PROJECT SUMMARY
# ============================================================

st.markdown("---")

st.header(
    "📌 Final Project Summary"
)

summary_col1, summary_col2 = st.columns(2)


with summary_col1:

    st.markdown(
        f"""
**Dataset**

- Training images: **{X_train.shape[1]}**
- Testing images: **{X_test.shape[1]}**
- Image size: **{IMAGE_WIDTH} × {IMAGE_HEIGHT}**
- Original dimension: **{X_train.shape[0]:,}**
- Training matrix: **{X_train.shape}**
- Testing matrix: **{X_test.shape}**
"""
    )


with summary_col2:

    variance_final = (
        cumulative_variance[-1] * 100
        if len(cumulative_variance) > 0
        else 0
    )

    st.markdown(
        f"""
**PCA & Recognition**

- PCA components: **{k}**
- Reduced dimension: **{k}**
- Variance retained: **{variance_final:.2f}%**
- Recognition accuracy: **{accuracy * 100:.2f}%**
- Correct predictions: **{correct}/{len(y_test)}**
- Orthogonality: **{"Verified" if is_orthogonal else "Not verified"}**
"""
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "FaceVector | MFAD Mini-Project | "
    "PCA + Eigenvectors + Projections + Face Recognition"
)