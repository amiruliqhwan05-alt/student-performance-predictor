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

from sklearn.linear_model import LinearRegression

# Create the model
model = LinearRegression()

# Train it on the training data
model.fit(X_train, y_train)

print("Model trained successfully.")

# Make predictions on the test set
y_pred = model.predict(X_test)

# Compare a few actual vs predicted values
comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(comparison.head(10))

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.4f}")