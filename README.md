# MNIST Support Vector Classifier (From Scratch)

A custom implementation of a **One-Versus-All (OvA) Support Vector Classifier** built entirely from scratch using Python and NumPy. This project solves the classic MNIST handwritten digit classification problem without relying on black-box optimization libraries like `sklearn.svm`.

Instead, it explicitly implements **Mini-Batch Stochastic Gradient Descent (SGD)** to minimize the **Regularized Hinge Loss** objective function.

## 👻 Feature Highlight: The "Ghost Digits"
One of the most compelling aspects of linear classifiers is their interpretability. Unlike deep neural networks, where features are hidden in abstract layers, a Linear SVC learns a weight vector $w$ that has a direct geometric relationship to the input image.

By reshaping the learned weight vectors ($1 \times 784$) back into image dimensions ($28 \times 28$), we can visualize exactly what the model "sees." 

### How to Read the Plots
The visualization uses a custom **Red-White-Black** colormap to display the learned weights for each digit class (0–9):
* **Black (Positive Weights):** The model rewards pixel intensity here. (e.g., "If there is ink here, it's likely this digit.")
* **Red (Negative Weights):** The model penalizes pixel intensity here. (e.g., "If there is ink here, it is definitely NOT this digit.")
* **White (Zero Weights):** These pixels are irrelevant to the decision boundary.

### Interpretation
The resulting images look like "ghostly" versions of the numbers. For example, the classifier for **Digit 2** learns positive weights (black) along the top and bottom curves, but learns negative weights (red) in the center-left, where a "3" or "0" might have ink. This visualization proves the model is not just memorizing data, but learning the structural "essence" of each digit.

*(See `SVC_Github.ipynb` for the full visualization code and output.)*

## Technical Methodology

### 1. Optimization Problem
The classifier minimizes the unconstrained **Regularized Hinge Loss** objective function:

$$
C(w, b) = \frac{\lambda}{2}||w||^2 + \frac{1}{N} \sum_{i=1}^{N} \max(0, 1 - y_i(w \cdot x_i + b))
$$

* **Regularization ($\frac{\lambda}{2}||w||^2$):** Maximizes the margin to prevent overfitting.
* **Hinge Loss:** Penalizes misclassifications and points inside the margin.

### 2. Algorithm
* **Strategy:** One-Versus-All (OvA) ensemble. 10 independent binary classifiers were trained (one for each digit vs. the rest).
* **Optimizer:** Mini-Batch Stochastic Gradient Descent (SGD).
* **Sub-Gradients:** Since Hinge Loss is non-differentiable at the margin, sub-gradients were used for parameter updates.
* **Hyperparameters:** Tuned via Grid Search (Best configuration: $\lambda=0.0001$, Decay Rate=0.05, Initial LR=0.1).

## Results
* **Mean Test Accuracy:** ~91.7%
* **Convergence:** The model converges rapidly, reaching near-optimal performance within just **20 epochs**, showing that simple linear models are highly effective for this high-dimensional dataset.

## Quick Start

### Prerequisites
* Python 3.8+
* Jupyter Notebook
* NumPy, Matplotlib, Seaborn, Scikit-Learn (used only for data fetching/splitting)

### Installation
1.  Clone the repo:
    ```bash
    git clone [https://github.com/Eschtayl/1-Vs-All_SVC_MNIST_from_scratch.git](https://github.com/Eschtayl/1-Vs-All_SVC_MNIST_from_scratch.git)
    cd 1-Vs-All_SVC_MNIST_from_scratch
    ```
2.  Install dependencies:
    ```bash
    pip install numpy matplotlib seaborn scikit-learn notebook
    ```
3.  Run the notebook:
    ```bash
    jupyter notebook SVC_Github.ipynb
    ```

## References
This implementation was informed by the mathematics and theory from:
1.  **Zisserman, A.** (2015). *The SVM classifier*. University of Oxford.
2.  **Lu, W.-S.** (2024). *Optimization for Machine Learning*. University of Victoria.
3.  **James, G., et al.** (2023). *An Introduction to Statistical Learning*. Springer.
