# libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# load the dataset
data = pd.read_csv('diabetes_risk.csv')
y = (data['diabetes_risk'] == 'High').astype(int).values # target/label
X = data[['bmi', 'hba1c_level', 'age', 'fasting_blood_sugar', 'waist_circumference_cm']].values #features

# train, test, split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# evaluation metric function 
def evaluate(name, model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f'...{name}....')
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print('precision:', precision_score(y_test, y_pred))
    print('Recall:', recall_score(y_test, y_pred))
    print('F1:', f1_score(y_test, y_pred))

# call the function
evaluate('Decision Tree', DecisionTreeClassifier(random_state=42))
evaluate('Random Forest', RandomForestClassifier(random_state=42))