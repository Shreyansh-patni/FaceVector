# FaceVector: Technical Methodology & Mathematical Formulation

## 1. Matrix Representation of Face Images

Each face image in the AT&T dataset has dimensions $H \times W = 112 \times 92$ pixels in 8-bit grayscale ($[0, 255]$).

The image is flattened into a 1D column vector $\mathbf{x}_i \in \mathbb{R}^d$, where:
$$d = 112 \times 92 = 10,304 \text{ pixels}$$

For $N = 360$ training images, the data matrix $X_{\text{train}} \in \mathbb{R}^{d \times N}$ is constructed by column-stacking:
$$X_{\text{train}} = \begin{bmatrix} \mathbf{x}_1 & \mathbf{x}_2 & \dots & \mathbf{x}_N \end{bmatrix} \in \mathbb{R}^{10304 \times 360}$$

Similarly, the test dataset matrix $X_{\text{test}} \in \mathbb{R}^{d \times M}$ with $M = 40$ images is represented as:
$$X_{\text{test}} = \begin{bmatrix} \mathbf{y}_1 & \mathbf{y}_2 & \dots & \mathbf{y}_M \end{bmatrix} \in \mathbb{R}^{10304 \times 40}$$

---

## 2. Mean Face Computation & Mean Centering

### 2.1 Mean Face
The mean face vector $\mathbf{\mu} \in \mathbb{R}^{10304}$ represents the pixel-wise average across all training face samples:
$$\mathbf{\mu} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{x}_i$$

### 2.2 Mean-Centered Matrix $P$
Subtracting the mean face from each training vector centers the dataset at the origin:
$$\mathbf{p}_i = \mathbf{x}_i - \mathbf{\mu}$$
$$P = \begin{bmatrix} \mathbf{p}_1 & \mathbf{p}_2 & \dots & \mathbf{p}_N \end{bmatrix} = X_{\text{train}} - \mathbf{\mu} \mathbf{1}^T \in \mathbb{R}^{10304 \times 360}$$

---

## 3. High-Dimensional Eigenproblem Reduction ($P^T P$ Trick)

### 3.1 The Full Covariance Matrix Problem
The standard spatial covariance matrix is defined as:
$$C_{\text{full}} = \frac{1}{N} P P^T \in \mathbb{R}^{10304 \times 10304}$$

Solving for the eigenvectors of $C_{\text{full}}$ requires diagonalizing a $10,304 \times 10,304$ matrix, which has $(10,304)^2 \approx 1.06 \times 10^8$ entries and is computationally expensive.

### 3.2 Gram Matrix Derivative ($P^T P$)
Instead, we compute the smaller Gram matrix $G \in \mathbb{R}^{N \times N}$:
$$G = P^T P \in \mathbb{R}^{360 \times 360}$$

Let $\mathbf{u}_i \in \mathbb{R}^{360}$ be an eigenvector of $G$ with corresponding non-zero eigenvalue $\lambda_i$:
$$P^T P \mathbf{u}_i = \lambda_i \mathbf{u}_i$$

Multiplying both sides on the left by $P$:
$$P (P^T P \mathbf{u}_i) = P (\lambda_i \mathbf{u}_i)$$
$$(P P^T) (P \mathbf{u}_i) = \lambda_i (P \mathbf{u}_i)$$

This proves that $\mathbf{v}_i = P \mathbf{u}_i \in \mathbb{R}^{10304}$ is an eigenvector of $P P^T$ with the exact same eigenvalue $\lambda_i$.

---

## 4. Eigenfaces Orthonormalization

To ensure that the eigenface basis vectors $\mathbf{v}_i$ form an orthonormal basis ($\mathbf{v}_i^T \mathbf{v}_j = \delta_{ij}$), we normalize each eigenface vector:
$$\mathbf{v}_i = \frac{P \mathbf{u}_i}{\|P \mathbf{u}_i\|_2}$$

Let $V = \begin{bmatrix} \mathbf{v}_1 & \mathbf{v}_2 & \dots & \mathbf{v}_k \end{bmatrix} \in \mathbb{R}^{10304 \times k}$ denote the top $k = 50$ principal components matrix.

Orthonormality condition:
$$V^T V \approx I_k \in \mathbb{R}^{50 \times 50}$$

---

## 5. PCA Projection & Dimensionality Reduction

Each face vector $\mathbf{x} \in \mathbb{R}^{10304}$ is projected into the low-dimensional $k$-dimensional PCA subspace ($k = 50$):
$$\mathbf{w} = V^T (\mathbf{x} - \mathbf{\mu}) \in \mathbb{R}^{50}$$

For all training images:
$$W_{\text{train}} = V^T P \in \mathbb{R}^{50 \times 360}$$

For all testing images:
$$W_{\text{test}} = V^T (X_{\text{test}} - \mathbf{\mu} \mathbf{1}^T) \in \mathbb{R}^{50 \times 40}$$

---

## 6. Nearest Neighbor Recognition via Euclidean Distance

Given a query test face $\mathbf{x}_{\text{test}}$ with PCA feature vector $\mathbf{w}_{\text{test}} \in \mathbb{R}^{50}$, we compute the Euclidean distance to every training face signature $\mathbf{w}_{\text{train}, j} \in \mathbb{R}^{50}$:
$$d_j = \|\mathbf{w}_{\text{test}} - \mathbf{w}_{\text{train}, j}\|_2 = \sqrt{\sum_{r=1}^{50} (w_{\text{test}, r} - w_{\text{train}, j, r})^2}$$

The predicted identity corresponds to the index of the closest training vector:
$$j^* = \arg\min_{j \in \{1, \dots, 360\}} d_j$$
$$\text{Predicted Label} = y_{\text{train}, j^*}$$

---

## 7. Face Image Reconstruction

A face vector $\mathbf{x}$ can be reconstructed from its $k$-dimensional PCA representation $\mathbf{w}$ using:
$$\mathbf{\hat{x}} = \mathbf{\mu} + V \mathbf{w} = \mathbf{\mu} + \sum_{r=1}^{k} w_r \mathbf{v}_r$$

The reconstruction error (residual norm) is defined as:
$$e = \|\mathbf{x} - \mathbf{\hat{x}}\|_2$$

With $k = 50$ components, fine facial details are smoothed out while essential structural features (eyes, nose, mouth geometry) are retained, explaining $\sim 82.03\%$ of total dataset variance.
