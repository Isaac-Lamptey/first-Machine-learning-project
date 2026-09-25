import pandas as pd
# load data
df = pd.read_csv('diabetes_risk.csv')
# the sfhape of the data
#print(df)

# get the data types of the column
print(df.dtypes)

# missing values
print(df.isnull().sum())

#Lets now filter usin blood_pressure_diastolic
print(df[df["blood_pressure_diastolic"]> 95])

# GROUP BY A CONDITION
print(df.groupby("diabetes_risk")["hours_sleep_per_night"].max())
# lets plot the 
import matplotlib.pyplot as plt
# plot the age and
# = df['age'].max()
#y = df['diabetes_risk']
#plt.plot(x, y)
df['stress_level'].hist()
plt.show()