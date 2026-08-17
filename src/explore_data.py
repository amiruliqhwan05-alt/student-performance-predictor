import pandas as pd

# Load the dataset
df = pd.read_csv("../data/StudentPerformanceFactors.csv")

# Look at the first 5 rows
print(df.head())

# Look at column names and data types
print(df.info())

# Check missing values per column
print(df.isnull().sum()) # ← Block 1: BEFORE cleaning (still dirty)

# Handle missing values in categorical columns
df['Teacher_Quality'] = df['Teacher_Quality'].fillna('Unknown')
df['Parental_Education_Level'] = df['Parental_Education_Level'].fillna('Unknown')
df['Distance_from_Home'] = df['Distance_from_Home'].fillna('Unknown')
# Verify no missing values remain
print(df.isnull().sum())  # ← Block 2: AFTER cleaning (now clean)

# Save the cleaned dataset to a new file
df.to_csv("../data/cleaned_students.csv", index=False)
print("Cleaned data saved to ../data/cleaned_students.csv")