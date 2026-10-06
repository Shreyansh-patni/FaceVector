# FaceVector: MFAD Viva Questions & Answers

This document provides a comprehensive list of likely viva / oral examination questions for the **UE25MA242A — Mathematical Foundation for AI & Data Science (MFAD)** mini-project.

---

## Section 1: Linear Algebra & Matrix Foundations

### Q1: What is the shape of the training face matrix in FaceVector, and what do the rows and columns represent?
**Answer**:  
The training face matrix $X_{\text{train}}$ has a shape of $(10304, 360)$.
- **Rows (10,304)**: Represent individual pixel features ($92 \times 112 = 10,304$ pixels per image).
- **Columns (360)**: Represent individual training face samples ($9 \text{ images/subject} \times 40 \text{ subjects} = 360$).

---

### Q2: Why do we mean-center the data matrix before computing PCA?
**Answer**:  
Mean centering shifts the dataset origin to the dataset mean vector $\mathbf{\mu}$. Without mean centering:
1. The first principal component would simply point towards the center of mass of the data rather than the direction of maximum variance.
2. Covariance matrix calculations require zero-mean variables: $C = \frac{1}{N} P P^T$ where $P = X - \mathbf{\mu} \mathbf{1}^T$.

---

### Q3: What is the mean face, and how is it calculated?
**Answer**:  
The mean face $\mathbf{\mu} \in \mathbb{R}^{10304}$ is the arithmetic mean vector computed across all columns of the training matrix:
$$\mathbf{\mu} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{x}_i$$
Each element of $\mathbf{\mu}$ represents the average pixel intensity at that specific spatial location across all 360 training images.

---

### Q4: Why do we solve the eigenvalue problem on $P^T P$ ($360 \times 360$) instead of $P P^T$ ($10304 \times 10304$)?
**Answer**:  
1. **Computational Complexity**: Diagonalizing a $10,304 \times 10,304$ matrix requires $O(d^3) \approx 1.1 \times 10^{12}$ operations. Diagonalizing a $360 \times 360$ matrix requires $O(N^3) \approx 4.6 \times 10^7$ operations, which is $\sim 24,000\times$ faster.
2. **Mathematical Equivalence**: If $\mathbf{u}_i$ is an eigenvector of $P^T P$ with eigenvalue $\lambda_i$, then $\mathbf{v}_i = P \mathbf{u}_i$ is an eigenvector of $P P^T$ with the exact same non-zero eigenvalue $\lambda_i$, because $P P^T (P \mathbf{u}_i) = P (P^T P \mathbf{u}_i) = \lambda_i (P \mathbf{u}_i)$.

---

### Q5: How many non-zero eigenvalues can $P^T P$ have at most?
**Answer**:  
At most $\min(d, N) - 1 = 360 - 1 = 359$ non-zero eigenvalues. Because the centered matrix $P$ has rank at most $N-1 = 359$ (due to mean subtraction reducing rank by 1).

---

## Section 2: Eigenfaces & Orthonormality

### Q6: What is an eigenface?
**Answer**:  
An eigenface is an eigenvector $\mathbf{v}_i \in \mathbb{R}^{10304}$ of the spatial covariance matrix $P P^T$, reshaped back into a $92 \times 112$ 2D image matrix for visual inspection. Eigenfaces represent the principal directions of variance across the facial dataset.

---

### Q7: Are eigenfaces actual human faces?
**Answer**:  
No. Eigenfaces are mathematical basis vectors (principal directions) in the $\mathbb{R}^{10304}$ image space. They highlight areas of maximum visual variance (e.g., eye sockets, nose bridge, jawline, lighting gradients).

---

### Q8: How do we verify that the eigenfaces are orthonormal?
**Answer**:  
We compute the matrix product $V^T V$ where $V \in \mathbb{R}^{10304 \times 50}$ contains the top 50 normalized eigenfaces. If $V$ is orthonormal, $V^T V = I_{50}$ (the $50 \times 50$ identity matrix). In FaceVector, $\|V^T V - I_{50}\|_{\infty} < 10^{-12}$, confirming orthonormality.

---

## Section 3: PCA, Projections & Variance

### Q9: What is the difference between explained variance and recognition accuracy?
**Answer**:  
- **Explained Variance (~82.03%)**: Measures the proportion of total pixel energy/information retained by the 50 PCA components relative to all 359 components.
- **Recognition Accuracy (95.00%)**: Measures the percentage of test query images correctly assigned to their ground-truth subject identity (38 out of 40 correct).

---

### Q10: How do we project a new face image into the PCA subspace?
**Answer**:  
Given a query image vector $\mathbf{x} \in \mathbb{R}^{10304}$:
1. Center the vector: $\mathbf{p} = \mathbf{x} - \mathbf{\mu}$.
2. Multiply by the transposed eigenface matrix $V^T \in \mathbb{R}^{50 \times 10304}$:
$$\mathbf{w} = V^T \mathbf{p} \in \mathbb{R}^{50}$$
$\mathbf{w}$ contains the 50 PCA projection coefficients (the signature of the face).

---

### Q11: How many PCA components does FaceVector use, and why?
**Answer**:  
FaceVector uses $k = 50$ principal components.
- **Dimensionality Reduction**: Reduces dimensions from $10,304 \to 50$ (a 99.51% feature compression).
- **Variance Retained**: Retains ~82.03% of cumulative variance, eliminating high-frequency noise while retaining essential facial structure.
- **High Accuracy**: Reaches a 95.00% recognition rate.

---

## Section 4: Classification & Reconstruction

### Q12: How does FaceVector classify a test face?
**Answer**:  
Using 1-Nearest Neighbor (1-NN) classification based on Euclidean distance in the 50-dimensional PCA space:
1. Project test face into $\mathbf{w}_{\text{test}} \in \mathbb{R}^{50}$.
2. Compute Euclidean distance $d_j = \|\mathbf{w}_{\text{test}} - \mathbf{w}_{\text{train}, j}\|_2$ for all 360 training projections.
3. Assign the identity label of the training sample $j^*$ that yields the minimum distance.

---

### Q13: How is a face reconstructed from its PCA coefficients?
**Answer**:  
Using the formula:
$$\mathbf{\hat{x}} = \mathbf{\mu} + V \mathbf{w} = \mathbf{\mu} + \sum_{i=1}^{50} w_i \mathbf{v}_i$$
Where $\mathbf{\mu}$ is the mean face, $V$ is the $10304 \times 50$ eigenface matrix, and $\mathbf{w}$ is the 50-dimensional coefficient vector.

---

### Q14: Why is the reconstructed face image slightly blurry compared to the original image?
**Answer**:  
Because only the top 50 principal components out of 359 available components are used. The discarded higher-order components capture fine high-frequency details (wrinkles, skin texture, noise). Retaining 50 components preserves smooth lower-frequency facial structures (~82.03% variance).

---

## Section 5: Experimental Results & Performance

### Q15: What are the exact experimental results achieved by FaceVector?
**Answer**:  
- **Dataset**: AT&T / ORL Face Dataset (40 subjects, 400 images).
- **Training Set**: 360 images (9 per subject).
- **Testing Set**: 40 images (1 per subject).
- **Components ($k$)**: 50.
- **Variance Retained**: 82.03%.
- **Correct Predictions**: 38 / 40.
- **Accuracy**: 95.00%.

---

### Q16: What are the primary limitations of Eigenface-based recognition?
**Answer**:  
1. **Sensitivity to Illumination & Expression**: PCA assumes linear variance; extreme shadows or facial expressions alter pixel values significantly.
2. **Alignment Requirement**: Faces must be centered and aligned.
3. **Linear Subspace Assumption**: Real facial manifold variation is non-linear (handled better by modern CNNs/deep learning).
