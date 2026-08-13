import pandas as pd

# Load the dataset
df = pd.read_csv("../data/StudentPerformanceFactors.csv")

# Look at the first 5 rows
print(df.head())

# Look at column names and data types
print(df.info())