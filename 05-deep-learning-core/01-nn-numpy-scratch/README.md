# 01 - Neural Network from Scratch (NumPy)

## Project Goal
This project is the first foundational module in the Deep Learning section of the **AI Accelerated 1 Year Plan**. The objective was to implement a complete, fully functional, two-layer (input, hidden, output) neural network using **only the NumPy library**—i.e., building it entirely from scratch without frameworks like PyTorch or TensorFlow.

## Core Accomplishment: Solving XOR
The implemented network successfully learns and solves the non-linear **XOR (Exclusive OR) logic problem**, proving the correctness of the core algorithms.

| Input (A, B) | True Value | Prediction |
| :----------: | :--------: | :--------: |
| [0 0] | 0 | $\approx 0.04$ |
| [0 1] | 1 | $\approx 0.94$ |
| [1 0] | 1 | $\approx 0.96$ |
| [1 1] | 0 | $\approx 0.06$ |

## Implementation Details

The `simple_nn.py` file contains the implementation of:
1.  **Forward Propagation:** Calculation of linear combinations and application of the Sigmoid activation function.
2.  **Loss Function:** Mean Squared Error (MSE).
3.  **Backpropagation:** Calculation of gradients using the chain rule.
4.  **Optimization:** Weight and bias updates via Gradient Descent.

## Setup and Execution

### Prerequisites
* Python 3.x
* The `numpy` library.

### 1. Activate Environment (as shown in your setup)
```bash
source venv_scratch/bin/activate
pip install numpy
python simple_nn.py
