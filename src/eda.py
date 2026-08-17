import pandas as pd

# Load the CLEANED dataset (not the raw one!)
df = pd.read_csv("../data/cleaned_students.csv")

# Basic statistics for all numeric columns
print(df.describe())

# Find rows where Exam_Score is above 100
suspicious = df[df['Exam_Score'] > 100]
print(suspicious)

# Cap Exam_Score at 100 (assuming 100 is the true maximum)
df.loc[df['Exam_Score'] > 100, 'Exam_Score'] = 100
print(df['Exam_Score'].max())

# Save the updated cleaned dataset
df.to_csv("../data/cleaned_students.csv", index=False)
print("Updated cleaned data saved.")