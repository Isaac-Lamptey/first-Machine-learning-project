import pandas as pd 
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score

# load dataset
data = pd.read_csv('diabetes_risk.csv')

# set features and labels from dataset
y = (data['diabetes_risk'] == 'High').astype(int).values # labels/target
X = (data[['bmi', 'hba1c_level']]).values # Using two featues

# X_train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# set the model
model = LogisticRegression()
model.fit(X_train, y_train)

# predict
y_pred = model.predict(X_test)

# Evaluation metrics
print("Accuracy:",  accuracy_score(y_test, y_pred))
print('Precision:', precision_score(y_test, y_pred))
print('Recal:', recall_score(y_test, y_pred))
print('F1:', f1_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# save the model
import joblib as jb
jb.dump(model, 'logreg_model.pkl')