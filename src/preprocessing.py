import pandas as pd

# Load the cleaned dataset
df = pd.read_csv("../data/cleaned_students.csv")

# List all categorical (text) columns
categorical_cols = df.select_dtypes(include='str').columns
print(categorical_cols)

# See the unique values in each categorical column
for col in categorical_cols:
    print(f"\n{col}: {df[col].unique()}")

# Encode binary (Yes/No, Public/Private, etc.) columns
df['Extracurricular_Activities'] = df['Extracurricular_Activities'].map({'No': 0, 'Yes': 1})
df['Internet_Access'] = df['Internet_Access'].map({'No': 0, 'Yes': 1})
df['Learning_Disabilities'] = df['Learning_Disabilities'].map({'No': 0, 'Yes': 1})
df['School_Type'] = df['School_Type'].map({'Public': 0, 'Private': 1})
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
print(df[['Extracurricular_Activities', 'Internet_Access', 'Learning_Disabilities', 'School_Type', 'Gender']].head())

# Encode ordinal columns (Low < Medium < High)
ordinal_map = {'Low': 0, 'Medium': 1, 'High': 2}

df['Parental_Involvement'] = df['Parental_Involvement'].map(ordinal_map)
df['Access_to_Resources'] = df['Access_to_Resources'].map(ordinal_map)
df['Motivation_Level'] = df['Motivation_Level'].map(ordinal_map)
df['Family_Income'] = df['Family_Income'].map(ordinal_map)
print(df[['Parental_Involvement', 'Access_to_Resources', 'Motivation_Level', 'Family_Income']].head())

# Encode ordinal columns that include "Unknown"
teacher_map = {'Low': 0, 'Medium': 1, 'High': 2, 'Unknown': -1}
df['Teacher_Quality'] = df['Teacher_Quality'].map(teacher_map)

education_map = {'High School': 0, 'College': 1, 'Postgraduate': 2, 'Unknown': -1}
df['Parental_Education_Level'] = df['Parental_Education_Level'].map(education_map)

distance_map = {'Near': 0, 'Moderate': 1, 'Far': 2, 'Unknown': -1}
df['Distance_from_Home'] = df['Distance_from_Home'].map(distance_map)
print(df[['Teacher_Quality', 'Parental_Education_Level', 'Distance_from_Home']].head())

# One-hot encode Peer_Influence (no natural order)
df = pd.get_dummies(df, columns=['Peer_Influence'])
print(df.filter(like='Peer_Influence').head())

# Save the fully preprocessed (encoded) dataset
df.to_csv("../data/preprocessed_students.csv", index=False)
print("Preprocessed data saved.")