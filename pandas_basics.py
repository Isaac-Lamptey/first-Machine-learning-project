
import pandas as pd
data = pd.read_csv("diabetes_risk.csv")
print(data.shape)
#print(data.dtypes)
#print(data.isnull().sum())
#select SOME COLUMNS
high_income_patients = data[data['income_bracket'] == 'high']
print(high_income_patients)