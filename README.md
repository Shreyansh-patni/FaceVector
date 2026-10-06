# FaceVector

### Eigenvector & PCA-Based Face Recognition System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Linear%20Algebra-blue?logo=numpy)](https://numpy.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Interactive%20UI-red?logo=streamlit)](https://streamlit.io/)
[![Course](https://img.shields.io/badge/MFAD-UE25MA242A-green.svg)]()
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**FaceVector** is an academic linear algebra and data science mini-project that implements a facial recognition and reconstruction pipeline using **Principal Component Analysis (PCA)**, **spectral matrix decomposition**, **eigenfaces**, and **subspace projections**.

---

## 📌 Overview

Facial recognition in high-dimensional image space ($10,304$ pixels per image) is computationally expensive and sensitive to noise. **FaceVector** solves this problem by projecting $10,304$-dimensional facial images into a compact $50$-dimensional **PCA subspace**. 

By exploiting the **Gram matrix trick** ($P^T P$ matrix optimization), the system reduces a $10,304 \times 10,304$ covariance eigenvalue problem to a $360 \times 360$ matrix decomposition, achieving **95.00% classification accuracy** and retaining **~82.03% cumulative variance**.

---

## 🎯 Problem Statement

This project directly addresses the core **UE25MA242A (MFAD)** problem statement:

> *"Projections, eigenvectors, Principal Component Analysis, and face recognition algorithms."*

---

## 🚀 Key Results

| Metric / Specification | Value / Benchmark Result |
|---|---|
| **Dataset** | AT&T / ORL Face Dataset |
| **Total Subjects** | 40 subjects |
| **Training Images** | 360 images (9 per subject) |
| **Testing Images** | 40 images (1 per subject) |
| **Image Dimensions** | $92 \times 112$ pixels (8-bit grayscale) |
| **Original Feature Dimensions** | $10,304$ pixels |
| **PCA Components Selected ($k$)** | **50** |
| **Reduced Dimensionality** | **50** features per image |
| **Explained Variance Retained** | **~82.03%** |
| **Correct Predictions** | **38 / 40** |
| **Recognition Accuracy** | **95.00%** |
| **Incorrect Predictions** | **2 / 40** |
| **Distance Metric** | Minimum Euclidean Distance in PCA Space |

> [!NOTE]  
> **Accuracy vs. Explained Variance**:  
> - **95.00%** is the **classification accuracy** (38/40 test faces correctly identified).  
> - **~82.03%** is the **explained variance retained** by the top 50 principal components.

---

## 🖥️ Demo / Interactive Application

The interactive **Streamlit** dashboard provides real-time visualization of data matrices, mean face calculation, eigenface decomposition, subspace projection, recognition, and image reconstruction.

## 🚀 Live Demo

**Try FaceVector live:**  
👉 [https://facevector.streamlit.app/](https://facevector.streamlit.app/)

## 🖥️ Preview

<p align="center">
  <img src="Screenshot.png" alt="FaceVector App Preview" width="100%">
</p>

## Preview

![FaceVector App Preview](Screenshot.png)
---

## 🧮 Mathematical Pipeline

$$\text{Image } (92 \times 112) \longrightarrow \text{Vector } (\mathbb{R}^{10304}) \longrightarrow \text{Matrix } X (\mathbb{R}^{10304 \times 360}) \longrightarrow \text{Mean Face } \mathbf{\mu}$$

$$\downarrow$$

$$\text{Reconstruction } \mathbf{\hat{x}} = \mathbf{\mu} + V \mathbf{w} \longleftarrow \text{1-NN Distance Matching} \longleftarrow \text{Projection } \mathbf{w} = V^T (\mathbf{x} - \mathbf{\mu}) \longleftarrow \text{Eigenfaces } V = P U$$

### 1. Matrix Representation
Each $92 \times 112$ grayscale image is flattened into a column vector $\mathbf{x}_i \in \mathbb{R}^{10304}$.
- **Training Matrix $X_{\text{train}}$**: $\mathbb{R}^{10304 \times 360}$
- **Testing Matrix $X_{\text{test}}$**: $\mathbb{R}^{10304 \times 40}$

### 2. Mean Face & Mean Centering
The mean face vector $\mathbf{\mu} \in \mathbb{R}^{10304}$ is the average across all 360 training face vectors:
$$\mathbf{\mu} = \frac{1}{N} \sum_{i=1}^{360} \mathbf{x}_i$$

Subtracting $\mathbf{\mu}$ yields the mean-centered data matrix $P \in \mathbb{R}^{10304 \times 360}$:
$$P = X_{\text{train}} - \mathbf{\mu} \mathbf{1}^T$$

### 3. Why $P^T P$ Instead of $P P^T$?
The full spatial covariance matrix $P P^T$ has dimensions $10,304 \times 10,304$ ($\approx 106 \text{ million entries}$). Finding its eigenvectors directly is computationally expensive.

Instead, we compute the smaller **Gram matrix** $G = P^T P \in \mathbb{R}^{360 \times 360}$.

If $\mathbf{u}_i$ is an eigenvector of $P^T P$ with eigenvalue $\lambda_i$:
$$P^T P \mathbf{u}_i = \lambda_i \mathbf{u}_i \implies P P^T (P \mathbf{u}_i) = \lambda_i (P \mathbf{u}_i)$$

Thus, $\mathbf{v}_i = P \mathbf{u}_i$ is an eigenvector of $P P^T$ with the exact same eigenvalue $\lambda_i$.

### 4. Eigenfaces Orthonormalization
Eigenfaces are normalized to form an orthonormal basis $V \in \mathbb{R}^{10304 \times 50}$:
$$\mathbf{v}_i = \frac{P \mathbf{u}_i}{\|P \mathbf{u}_i\|_2} \implies V^T V \approx I_{50}$$

### 5. PCA Projection & Subspace Mapping
Each centered face vector is projected onto the top 50 eigenfaces:
$$\mathbf{w} = V^T (\mathbf{x} - \mathbf{\mu}) \in \mathbb{R}^{50}$$

### 6. Recognition via Minimum Euclidean Distance
For a test projection $\mathbf{w}_{\text{test}} \in \mathbb{R}^{50}$, identity is assigned by finding the closest training signature:
$$j^* = \arg\min_{j \in \{1, \dots, 360\}} \|\mathbf{w}_{\text{test}} - \mathbf{w}_{\text{train}, j}\|_2$$

### 7. Image Reconstruction
A face image is reconstructed from its 50-dimensional PCA projection using:
$$\mathbf{\hat{x}} = \mathbf{\mu} + V \mathbf{w}$$

---

## 🖼️ Visual Gallery

### Project Overview & Metrics
![Project Overview](docs/screenshots/01-project-overview.png)

### Matrix Representation
![Matrix Representation](docs/screenshots/02-matrix-representation.png)

### Mean Face Computation
![Mean Face](docs/screenshots/03-mean-face.png)

### Mean Centering & $P^T P$ Gram Matrix
![PCA Analysis](docs/screenshots/04-pca-analysis.png)

### Eigenface Orthonormality Check ($V^T V \approx I$)
![Eigenface Orthogonality](docs/screenshots/06-orthogonality.png)

### Face Recognition Interface
![Face Recognition](docs/screenshots/07-face-recognition.png)

### Face Reconstruction (50 Components)
![Face Reconstruction](docs/screenshots/08-face-reconstruction.png)

### Top Learned Eigenfaces
![Eigenfaces](docs/screenshots/09-eigenfaces.png)

### Cumulative Explained Variance Plot
![Explained Variance](docs/screenshots/10-explained-variance.png)

### Final Performance Summary
![Final Results](docs/screenshots/11-final-results.png)

---

## 💻 Installation & Setup

### Prerequisites
- Python 3.10+
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/Shreyansh-patni/FaceVector.git
cd FaceVector
```

### 2. Create and Activate Virtual Environment
**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\activate
```

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application
```bash
streamlit run app.py
```

### 5. Run Technical Pipeline & Verification Tests
To run standalone pipeline verification scripts:
```bash
python src/test_pipeline.py
python src/check_orthogonality.py
python src/variance_analysis.py
python src/reconstruction.py
```

---

## 📂 Project Structure

```text
FaceVector/
├── app.py                      # Interactive Streamlit Web Application
├── README.md                   # Repository Documentation
├── requirements.txt            # Project Runtime Dependencies
├── .gitignore                  # Git Ignore Rules
├── LICENSE                     # MIT Open-Source License
│
├── src/                        # Core Python Engine
│   ├── __init__.py
│   ├── data_loader.py          # Image loading & vectorization
│   ├── pca_engine.py           # PᵀP eigenvalue & eigenface computation
│   ├── projection.py           # Feature projection into PCA subspace
│   ├── recognition.py          # Euclidean distance matching
│   ├── dataset_check.py        # Dataset structure verification
│   ├── check_orthogonality.py  # Orthonormality test (VᵀV ≈ I)
│   ├── variance_analysis.py    # Cumulative explained variance analysis
│   ├── visualize_eigenfaces.py # Eigenface visualizer
│   ├── reconstruction.py       # Face image reconstruction
│   └── test_pipeline.py        # Comprehensive verification test
│
├── outputs/                    # Generated Visualizations & Figures
│   ├── eigenfaces.png          # Top eigenfaces plot
│   ├── face_reconstruction.png # Face reconstruction comparison
│   ├── explained_variance.png  # Cumulative variance curve
│   └── eigenvalue_spectrum.png # Eigenvalue magnitude spectrum
│
├── docs/                       # Academic Documentation & Reports
│   ├── MFAD_Report.md          # Comprehensive Academic Project Report
│   ├── Methodology.md          # In-depth Mathematical Formulation
│   ├── Viva_Questions.md       # 35+ Viva Questions and Detailed Answers
│   └── screenshots/            # UI Application Screenshots
│       ├── 01-project-overview.png
│       ├── 02-matrix-representation.png
│       ├── 03-mean-face.png
│       ├── 04-pca-analysis.png
│       ├── 05-eigenvalue-analysis.png
│       ├── 06-orthogonality.png
│       ├── 07-face-recognition.png
│       ├── 08-face-reconstruction.png
│       ├── 09-eigenfaces.png
│       ├── 10-explained-variance.png
│       └── 11-final-results.png
│
└── data/                       # Dataset Documentation & Files
    └── README.md               # Dataset details & setup guide
```

---

## 🛠️ Built With

- **Python 3.10+**: Core programming language.
- **NumPy**: Linear algebra, matrix operations, matrix multiplication, and eigenvalue solver (`np.linalg.eigh`).
- **Pillow (PIL)**: Image processing and grayscale vector conversion.
- **Matplotlib**: Eigenface rendering, variance plotting, and reconstruction figures.
- **Streamlit**: Web front-end for interactive demonstration.

---

## ⚠️ Limitations

- **Illumination Sensitivity**: PCA assumes linear variance; extreme shadows or directional lighting degrade performance.
- **Pose & Alignment Requirements**: Images must be centered and aligned.
- **Nearest-Neighbor Classifier**: Simple 1-NN Euclidean distance is less robust on very large datasets compared to non-linear classifiers.
- **Linear Subspace Assumption**: Real facial manifolds are non-linear; modern deep learning (e.g., FaceNet, CNNs) captures non-linear features more effectively.

---

## 🔮 Future Scope

- **Facial Alignment Preprocessing**: Integrating automatic eye/nose alignment.
- **Illumination Normalization**: Histogram equalization prior to mean centering.
- **Alternative Distance Metrics**: Testing Cosine distance and Mahalanobis distance in PCA space.
- **Comparison with Modern Models**: Benchmarking PCA Eigenfaces against Fisherfaces (LDA) and Deep Learning (CNNs).
- **Real-Time Webcam Support**: Live facial identification stream using OpenCV.

---

## 🎓 Academic Context

- **Course**: UE25MA242A — Mathematical Foundation for AI & Data Science (MFAD)
- **Project Title**: FaceVector: Eigenvector & PCA-Based Face Recognition System

---

## 📜 License

This project is open-source and released under the [MIT License](LICENSE).
