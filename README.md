# Digit Recognition using a Multi-Layer Perceptron (MLP)


A notebook focused on implementing a neural network for a real-world task: handwritten digit recognition using the `MLPClassifier` from scikit-learn.

---

## 📖 Lab Overview

In this lab, we will build, train, and evaluate a neural network to classify handwritten digits from the famous MNIST dataset. This exercise provides a practical introduction to applying neural networks using a powerful, high-level library. The primary tool used is the **Multi-layer Perceptron (MLP)** classifier from scikit-learn.

### 🎯 Objectives
* Implement a neural network for a real-world classification task.
* Understand the workflow of loading data, preprocessing, training, and evaluation.
* Experiment with the hyperparameters of the `MLPClassifier`.
* Evaluate the model's performance using metrics like accuracy and a confusion matrix.



---

## 🛠️ Tools and Libraries

* **Python 3.x**
* **Scikit-learn:** For the MLP classifier and evaluation metrics.
* **NumPy:** For efficient numerical operations.
* **Matplotlib / Seaborn:** For data visualization, such as plotting the confusion matrix.

---


### Example Code Snippet
```python
from sklearn.neural_network import MLPClassifier

# Initialize the MLP Classifier
# Example architecture: two hidden layers with 50 neurons each
mlp = MLPClassifier(hidden_layer_sizes=(50, 50), max_iter=10, alpha=1e-4,
                    solver='sgd', verbose=10, random_state=1,
                    learning_rate_init=.1)

# Train the model
mlp.fit(X_train, y_train)
```
