# libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score

#load data
data = pd.read_csv('diabetes_risk.csv')
y = (data['diabetes_risk'] == 'High'). astype(int).values
X = data[['bmi', 'hba1c_level', 'age', 'fasting_blood_sugar', 'waist_circumference_cm']].values
#X = data[['bmi', 'age', 'waist_circumference_cm']].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
for depth in [2, 3, 5, 10,20, None]:
    models = [
        ('Tree', DecisionTreeClassifier(max_depth=depth, random_state=42)),
        ('Forest', RandomForestClassifier(max_depth=depth, random_state=42)),
      ]
    for name, model in models:
        model.fit(X_train, y_train)
        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))
        test_rec = recall_score(y_test, model.predict(X_test))
        print(f'{name:6} depth={str(depth):4} train= {train_acc:.3f} test={test_acc:.3f} gap={train_acc - test_acc:.3f} recall={test_rec: .3f}')

#cross validation
from sklearn.model_selection import cross_val_score, StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for depth in [2, 3, 5, 10, None]:
    model = RandomForestClassifier(max_depth=depth, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1")
    print(f"depth={str(depth):4} f1 mean={scores.mean():.3f} std={scores.std():.3f}")


    #choosing threshold using cross-validation
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_score, recall_score, f1_score

model = RandomForestClassifier(max_depth=5, random_state=42)
proba = cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba")[:, 1]

for t in [0.2, 0.3, 0.4, 0.5, 0.6]:
    pred = (proba >= t).astype(int)
    print(f"t={t} precision={precision_score(y_train, pred):.3f} "
          f"recall={recall_score(y_train, pred):.3f} f1={f1_score(y_train, pred):.3f}")

# using test dataset
print('the test set')
from sklearn.metrics import confusion_matrix, precision_score, recall_score

final = RandomForestClassifier(max_depth=5, random_state=42)
final.fit(X_train, y_train)
proba_test = final.predict_proba(X_test)[:, 1]

for t in [0.2, 0.5]:
    pred = (proba_test >= t).astype(int)
    print(f"t={t} precision={precision_score(y_test, pred):.3f} recall={recall_score(y_test, pred):.3f}")
    print(confusion_matrix(y_test, pred))