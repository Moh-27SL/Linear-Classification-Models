# Linear-Classification-Models

This repository contains implementations of linear classification models developed
from scratch using PyTorch, as part of Assignment 2.

The assignment focuses on understanding discriminative linear classifiers, loss
functions, optimization using gradient descent, and performance evaluation for both
binary and multiclass problems.

------------------------------------------------------------

Assignment Objectives

- Implement linear classifiers from scratch using PyTorch.
- Understand logistic regression for binary classification.
- Extend logistic regression to multiclass classification using softmax.
- Implement cross‑entropy loss functions manually.
- Train models using gradient descent.
- Evaluate and compare binary vs. multiclass performance.

------------------------------------------------------------

Models Implemented

1. Binary Logistic Regression
   - Task: Digit 1 vs. Rest
   - Activation: Sigmoid
   - Loss Function: Binary Cross‑Entropy
   - Output: Probability in [0, 1]
   - Decision Rule: Threshold at 0.5

2. Multiclass Logistic Regression (Softmax Regression)
   - Task: 10‑class digit classification
   - Activation: Softmax
   - Loss Function: Categorical Cross‑Entropy
   - Output: Probability distribution over 10 classes
   - Decision Rule: Argmax over class probabilities

------------------------------------------------------------

Dataset & Preprocessing

- Dataset: MNIST handwritten digits
- Images normalized to range [0, 1]
- Stratified data split:
  - Training set: 36,000 samples
  - Validation set: 12,000 samples
  - Test set: 12,000 samples
- Labels:
  - Binary task: digit 1 vs all others
  - Multiclass task: one‑hot encoded labels

------------------------------------------------------------

Training Procedure

- Bias term added explicitly to input features.
- Weights initialized randomly.
- Optimization performed using full‑batch gradient descent.
- Hyperparameters:
  - Learning rate (α)
  - Number of epochs
- Training and validation loss tracked for each epoch.
- Accuracy computed during training for monitoring performance.

------------------------------------------------------------

Evaluation Metrics

- Classification Accuracy
- Confusion Matrix
- Training & Validation Loss Curves
- Training & Validation Accuracy Curves

------------------------------------------------------------

Results

- Binary Logistic Regression:
  - Test Accuracy: 88.77%
  - Strong separation between digit 1 and non‑1 digits
  - Faster convergence and higher accuracy

- Multiclass Softmax Regression:
  - Test Accuracy: 64.51%
  - Increased difficulty due to overlapping class distributions
  - Demonstrates limitations of linear decision boundaries

------------------------------------------------------------

Analysis & Observations

- Binary classification is significantly easier than multiclass classification.
- Softmax regression struggles with complex, non‑linear digit relationships.
- Linear decision boundaries limit classification performance on MNIST.
- Loss curves confirm stable convergence using gradient descent.
- Confusion matrices highlight frequent misclassification between similar digits.

------------------------------------------------------------

Implementation Notes

- All forward passes, losses, and gradient updates are implemented manually.
- PyTorch is used only for tensor operations and automatic differentiation.
- scikit‑learn is used only for:
  - Data splitting
  - Accuracy calculation
  - Confusion matrix generation
- No high‑level ML models or optimizers are used.

------------------------------------------------------------

Technologies Used

- Python
- PyTorch
- NumPy
- scikit‑learn
- Matplotlib
- Seaborn

------------------------------------------------------------

Authors

Developed by Mohammed Yasser Mohammed
with classmates for a course assignment
email: es-mohamed.yasser2027@alexu.edu.eg
