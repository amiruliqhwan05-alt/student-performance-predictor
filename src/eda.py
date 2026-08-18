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

import matplotlib.pyplot as plt
plt.scatter(df['Attendance'], df['Exam_Score'], alpha=0.3)
plt.xlabel('Attendance (%)')
plt.ylabel('Exam Score')
plt.title('Attendance vs Exam Score')
plt.savefig('../notebooks/attendance_vs_score.png')
plt.show()

plt.scatter(df['Hours_Studied'], df['Exam_Score'], alpha=0.3)
plt.xlabel('Hours Studied per Week')
plt.ylabel('Exam Score')
plt.title('Hours Studied vs Exam Score')
plt.savefig('../notebooks/hours_vs_score.png')
plt.show()



# Correlation matrix for numeric columns
print(df.corr(numeric_only=True))