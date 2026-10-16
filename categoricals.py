#libraries
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

# load the dataset
data = pd.read_csv('diabetes_risk.csv')
y = (data['diabetes_risk'] == 'High').astype(int)

# which columns are text?
cat_cols = data.select_dtypes(exclude='number').columns.drop('diabetes_risk')
print(list(cat_cols))

#idea 1: missing text becomes its own category
data[cat_cols] = data[cat_cols].fillna('missing')

#idea 2: one-hot encoding
X = pd.get_dummies(data.drop(columns= ['diabetes_risk', 'patient_id']), columns=cat_cols)
print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model = RandomForestClassifier(max_depth=5, random_state=42)
proba = cross_val_predict(model, X_train, y_train, cv=cv, method='predict_proba')[:, 1]
pred = (proba > 0.3).astype(int)

print(f'precision: {precision_score(y_train, pred):.3f}, recall: {recall_score(y_train, pred):.3f}, f1: {f1_score(y_train, pred):.3f}')