# 1. Load dataset
# 2. the sigmoid function
# 3. initialize parameters and the forward pass
# 4. the loss function
# 5. the gradient
# 6. parameter update and the training loop

# set up
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#load data
df = pd.read_csv("diabetes_risk.csv")
y = (df['diabetes_risk'] == 'High').astype(int).values # target labels
X = df[['age', 'bmi', 'hours_sleep_per_night']] # features

# the sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Parameters and the forward pass
n_features = X.shape[1] 
w = np.zeros(n_features)
b = 0.0

# the forward pass
def predict_proba(X, w, b):
    z = X @ w + b # linear combination
    return sigmoid(z)
p = predict_proba(X, w, b)

# The loss function
def binary_cross_entropy_loss(y, p):
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

# The gradient
def gradient_descent(X, y, p):
    n = X.shape[0]
    error = p - y
    dw =  (1 / n) * X.T @ error
    db = (1 / n) * np.sum(error)
    return dw, db
dw, db = gradient_descent(X, y, p)





