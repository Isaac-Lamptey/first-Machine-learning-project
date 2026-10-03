
import pandas as pd
data = pd.read_csv("diabetes_risk.csv")
print(data.shape)
#print(data.dtypes)
#print(data.isnull().sum())
#filter
high_income_patients = data[data['income_bracket'] == 'High']
print(high_income_patients)