# FaceVector: Eigenvector & PCA-Based Face Recognition System
**Course**: UE25MA242A — Mathematical Foundation for AI & Data Science (MFAD)  
**Project Type**: Academic Mini-Project  

---

## Abstract

FaceVector is a linear-algebraic facial recognition and reconstruction system built upon Principal Component Analysis (PCA), spectral decomposition of high-dimensional matrix operators, and orthogonal basis projections. Utilizing the AT&T / ORL Face Dataset (40 subjects, 400 total images of size $92 \times 112$), the project transforms $10,304$-dimensional facial pixel spaces into a compact $50$-dimensional eigenface subspace. By computing eigenvalues and eigenvectors of the reduced $360 \times 360$ Gram matrix $P^T P$ rather than the prohibitive $10,304 \times 10,304$ spatial covariance matrix, computational complexity is reduced dramatically. Experimental results demonstrate that $50$ principal components preserve **82.03%** of cumulative variance while achieving **95.00% recognition accuracy** (38/40 correct classifications) using minimum Euclidean distance matching.

---

## 1. Problem Statement

The project addresses the core MFAD topic:
> *"Projections, eigenvectors, Principal Component Analysis, and face recognition algorithms."*

High-dimensional image data ($10,304$ pixels per face) suffers from the curse of dimensionality, high computational redundancy, and noise sensitivity. The objective is to formulate an optimal linear subspace projection that minimizes information loss while enabling rapid, accurate identity classification and image reconstruction.

---

## 2. Project Objectives

1. **Dimensionality Reduction**: Reduce face image vectors from $10,304$ dimensions to $50$ dimensions.
2. **Computational Optimization**: Implement the $P^T P$ matrix trick to solve a $360 \times 360$ eigenvalue problem instead of a $10,304 \times 10,304$ problem.
3. **Eigenface Basis Generation**: Extract and normalize orthonormal eigenface vectors.
4. **Subspace Projection**: Project training and test sets into feature spaces.
5. **Pattern Classification**: Classify test samples using nearest-neighbor Euclidean distance.
6. **Facial Reconstruction**: Reconstruct facial images from low-dimensional projections and evaluate reconstruction fidelity.

---

## 3. Verified Benchmark Results

| Metric / Parameter | Value / Finding |
|---|---|
| Dataset Name | AT&T / ORL Face Dataset |
| Total Subjects | 40 subjects |
| Total Images | 400 images |
| Training Images | 360 images (9 per subject) |
| Testing Images | 40 images (1 per subject) |
| Image Dimensions | $92 \times 112$ pixels (grayscale) |
| Original Feature Dimensions | $10,304$ pixels |
| Training Matrix $X_{\text{train}}$ Shape | $(10304, 360)$ |
| Testing Matrix $X_{\text{test}}$ Shape | $(10304, 40)$ |
| Mean Centered Matrix $P$ Shape | $(10304, 360)$ |
| Gram Matrix $P^T P$ Shape | $(360, 360)$ |
| PCA Components Selected ($k$) | 50 components |
| Reduced Dimension | 50 features per image |
| Explained Variance Retained ($k=50$) | **~82.03%** |
| Correct Test Classifications | **38 / 40** |
| Incorrect Test Classifications | **2 / 40** |
| Recognition Accuracy | **95.00%** |
| Eigenface Basis Orthogonality Error $\|V^T V - I_k\|_{\infty}$ | $< 1 \times 10^{-12}$ (Orthonormal) |

---

## 4. Mathematical Pipeline & Algorithm Summary

```text
Image (92x112) -> Vector (10304) -> Data Matrix X (10304x360) -> Mean Face μ -> Centered Matrix P (10304x360)
  -> Gram Matrix PᵀP (360x360) -> Eigenvalues λ & Eigenvectors U -> Eigenfaces V = P U -> Orthonormalization V / ||V||
  -> Subspace Projection W = Vᵀ(X - μ) -> Euclidean Distance Classification -> Face Reconstruction X̂ = μ + V W
```

---

## 5. Mathematical Mechanics

### 5.1 Covariance Matrix vs. Gram Matrix Optimization
Calculating eigenvectors for $C = \frac{1}{N} P P^T \in \mathbb{R}^{10304 \times 10304}$ is intractable for interactive systems. By leveraging the identity:
$$P^T P \mathbf{u}_i = \lambda_i \mathbf{u}_i \implies P P^T (P \mathbf{u}_i) = \lambda_i (P \mathbf{u}_i)$$

The eigenvectors of $P P^T$ are obtained by left-multiplying the eigenvectors $\mathbf{u}_i$ of $P^T P$ by $P$.

### 5.2 Variance Retention vs. Dimensionality Reduction
The cumulative variance ratio $v_k$ for $k$ components is given by:
$$v_k = \frac{\sum_{i=1}^{k} \lambda_i}{\sum_{i=1}^{N-1} \lambda_i}$$

For $k=50$, $v_{50} \approx 82.03\%$, which captures the main structural features (head shape, eye positioning, nose width, jawline) while removing high-frequency noise.

---

## 6. Visual Outputs & System Visualizations

The generated plots and analytical visualizations are stored in `outputs/`:
- `outputs/eigenfaces.png`: Visualizes the top 9 eigenfaces representing primary visual principal components.
- `outputs/explained_variance.png`: Displays cumulative variance retained as a function of component count.
- `outputs/eigenvalue_spectrum.png`: Plots the exponential decay of eigenvalues.
- `outputs/face_reconstruction.png`: Compares original test faces against reconstructions using $50$ components.

---

## 7. Performance Analysis & Limitations

### 7.1 Strengths
- Fast inference time (< 1 ms per query face projection and distance lookup).
- Extreme compression ratio ($10,304 \to 50$, representing a $99.51\%$ memory footprint reduction).
- High classification rate ($95.00\%$) on standardized facial datasets.

### 7.2 Limitations
- Sensitive to extreme lighting variations and shadows.
- Requires face alignment and uniform scaling.
- Nearest-neighbor classification degrades with very large, unconstrained populations.

---

## 8. Conclusion

The FaceVector project demonstrates that fundamental linear algebra principles—specifically orthogonal basis projections and spectral matrix decomposition—provide a highly effective, mathematically elegant framework for face recognition. By retaining only 50 principal components out of 10,304 original features, the system retains ~82.03% of visual variance and achieves a 95.00% classification accuracy.

---

## References
1. Turk, M., & Pentland, A. (1991). Eigenfaces for recognition. *Journal of Cognitive Neuroscience*, 3(1), 71-86.
2. Strang, G. (2016). *Introduction to Linear Algebra* (5th ed.). Wellesley-Cambridge Press.
