# DominantColorNeural
Make sure you have python3 installed.

**1.Clone repository.**  
**2.Create virtual environment: python -m venv venv**  
**3.Activate the venv: venv\Scripts\activate**  
**4.Install the dependencies: pip install -r requirements.txt**  

# 📄 Technical Documentation

**Project:** Dominant Color Detection using Neural Network

---

## 1. Overview

This project implements a simple feedforward neural network to detect the **dominant color** (Red, Green, or Blue) in a set of images. The system uses **heuristic labeling** to automatically assign labels based on pixel intensities, allowing training without manually labeled data.

---

## 2. Application Flow

### 2.1 Input

* A folder containing image files (`input/`).
* Images can be in standard formats (e.g., `.jpg`, `.png`).

### 2.2 Preprocessing

* All images are resized to **50×50 pixels**.
* Converted to NumPy arrays and normalized to values between `0.0` and `1.0`.
* Flattened to 1D vectors of size **7500** (`50×50×3` RGB values).

### 2.3 Heuristic Labeling

* For each image, the average values of **Red**, **Green**, and **Blue** channels are computed.
* The channel with the highest average is selected as the dominant color.
* Labels are represented as one-hot vectors:

  * `[1, 0, 0]` → Red
  * `[0, 1, 0]` → Green
  * `[0, 0, 1]` → Blue

---

## 3. Neural Network Architecture

| Layer        | Size / Type     | Activation |
| ------------ | --------------- | ---------- |
| Input Layer  | 7500 features   | —          |
| Hidden Layer | 32 neurons      | ReLU       |
| Output Layer | 3 neurons (RGB) | Softmax    |

### Parameters

* **Weights and biases** are initialized using:

  * He initialization for ReLU layer.
  * Xavier initialization for Softmax output.
* **Loss Function:** Cross-Entropy
* **Optimizer:** Gradient Descent

---

## 4. Training Process

### 4.1 Forward Propagation

* Compute `Z1 = X · W1 + b1`
* Apply ReLU activation: `A1 = ReLU(Z1)`
* Compute `Z2 = A1 · W2 + b2`
* Apply Softmax: `A2 = softmax(Z2)` — this yields predicted class probabilities.

### 4.2 Loss Calculation

* Use **cross-entropy** to measure prediction error:

  $$
  \text{Loss} = -\sum(Y \cdot \log(A2))
  $$

### 4.3 Backward Propagation

* Compute gradients of weights and biases via backpropagation.
* Update weights using gradient descent:

  $$
  W := W - \text{learning_rate} \cdot \frac{\partial \text{Loss}}{\partial W}
  $$

### 4.4 Epochs

* Repeat forward + backward pass for a fixed number of epochs (e.g., 100).
* Optionally, display loss every N epochs.
* Store and optionally plot loss values per epoch.

---

## 5. Prediction and Output

After training:

* For each image, the network outputs a probability distribution across `[Red, Green, Blue]`.
* The predicted dominant color is the one with the highest probability.
* Predictions are printed alongside true (heuristic) labels and image filenames.

Example Output:

```
File: image_01.jpg | Prediction: [0.972, 0.027, 0.000] | Label: [1, 0, 0]
File: image_02.jpg | Prediction: [0.053, 0.927, 0.020] | Label: [0, 1, 0]
```

---

## 6. Visualization

* A line plot of training loss is displayed using `matplotlib`, showing how the model learns over time.

---

