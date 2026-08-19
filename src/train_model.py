import pandas as pd

# Load the fully preprocessed dataset
df = pd.read_csv("../data/preprocessed_students.csv")

# X = everything except Exam_Score (our inputs/features)
X = df.drop('Exam_Score', axis=1)

# y = just Exam_Score (what we want to predict)
y = df['Exam_Score']

print("X shape:", X.shape)
print("y shape:", y.shape)

from sklearn.model_selection import train_test_split

# Split into 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)