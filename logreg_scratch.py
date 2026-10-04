import numpy as np
import pandas as pd
# STEP 1: THE PROBLEM
# load dataset
data = pd.read_csv("diabetes_risk.csv")

# Binary target: 1 if high risk, else 0
y = (data['diabetes_risk'] == 'High').astype(int).values

# Using two numeric features to start
X = data[['bmi', 'hba1c_level']].values

#Check the shapes of the features and the targets
#print(X.shape, y.shape)

#Check if the target has the binary category in a sane ratio
#print(y[:10])

# Check for class balance
#print(y.mean())


# STEP 2: The sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

#STEP 3: parameter initialization and the forward pass
n_features = X.shape[1] 
w = np.zeros(n_features) # weights, one per feature
b = 0.0 # bias

# The forward pass
def predict_proba(X, w, b):
    z = X @ w + b # linear combination
    return sigmoid(z)
p = predict_proba(X, w, b)

# STEP 4: The Loss function
def binary_cross_entropy(y, p):
    eps = 1e-15
    p = np.clip(p, eps, 1 - eps) # A numerical stability guard. Prevents log(0), which is numerically problematic.
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
#print(binary_cross_entropy(y, p))

# STEP 5: The Gradient 
def compute_gradients(X, y, p):
    n = X.shape[0]
    error = p - y # How wrong each prediction is
    dw =  (1 / n) * X.T @ error
    db = (1 / n) * np.sum(error)
    return dw, db
dw, db = compute_gradients(X, y, p)
#print(dw)
#print(db)

# STEP 6: PARAMETER RESET AND THE TRAINING LOOP
w, b
X_mean = X.mean(axis= 0)
X_std = X.std(axis= 0)
X_scaled = (X - X_mean) / X_std #Standardizes the features. Feature scaling
#print(X_scaled.mean(axis = 0))

learning_rate= 0.01
epochs= 3000
losses = []

for i in range(epochs):
    p = predict_proba(X_scaled, w, b)
    loss = binary_cross_entropy(y, p)
    losses.append(loss)

    dw, db = compute_gradients(X_scaled, y, p)
    w = w - learning_rate * dw
    b = b - learning_rate * db
    if i % 100 == 0:
        print(f"epoch {i}, loss = {loss: 4f}")
print("FINAL LOSS:", losses[-1])
print('FINAL w:', w)
print('FINAL b:', b)




#DAY6: Turning probabilities into predictions

# STEP 1: Threshold and predict
def predict_class(p, threshold=0.5):
    return (p >= threshold).astype(int)
p = predict_proba(X_scaled, w, b)
y_pred = predict_class(p)
#print(y_pred[:20])
#print([y[:20]])




# step 2: confusion matrix
TP = np.sum((y_pred == 1) & (y == 1)) #predicted 'High risk', actually 'High risk'
TN = np.sum((y_pred == 0) & (y == 0)) #predicted 'not High risk', actually 'not High risk'
FP = np.sum((y_pred == 1) & (y == 0)) #predicted 'High risk', actually 'not High risk'
FN = np.sum((y_pred == 0) & (y == 1)) #predicted 'not High risk', actually 'High risk'
print("TP:", TP, 'TN:', TN, 'FP:', FP, 'FN:', FN)


# Plotting the loss curve
import matplotlib.pyplot as plt
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()
