
# version A

# version B
# the above was just a test of the merge conflict resolution process


# lets now work witth the actual code
# starting with pandas
import pandas as pd
#load
df = pd.read_csv('diabetes_risk.csv')
# shape: rows and columns
print(df.shape)
# column names + data types
print(df.dtypes)
# first 5 rows, to just explore the data
print(df.head())
# summary statistics(mean, std, min, max, etc per column)
print(df.describe())
# select one column (a series)
print(df['Glucose'])
# filter rows where a condition is true
print(df[df["Glucose"] > 150])

# group by a column, get mean of another
print(df.groupby("Outcome")["Age"].mean())


# missing values per column
print(df.isnull().sum())

# filter: rows where age > 50
print(df[df["Age"] > 50])

# groupby: mean of bmi by diabetes_risk
print(df.groupby("Diabetes_risk")["BMI"].mean())

# plot
import matplotlib.pyplot as plt
df["Age"].hist()
plt.show()