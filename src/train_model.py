import pandas as pd

# Load the fully preprocessed dataset
df = pd.read_csv("../data/preprocessed_students.csv")

# X = everything except Exam_Score (our inputs/features)
X = df.drop('Exam_Score', axis=1)
print(X.columns.tolist())

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
#Figure out the best weights using these 5285 training students. Store them inside model
model.fit(X_train, y_train)

print("Model trained successfully.")

# Make predictions on the test set
# "Take the weights you already found. For each of these 1322 test students, plug their Hours_Studied, Attendance, etc. into the formula, and spit out a predicted Exam_Score."
y_pred = model.predict(X_test)

# Compare a few actual vs predicted values
#y_test = the real exam scores for your 1322 test students (the answers we hid from the model).
#y_pred = the model's guesses for those same 1322 students.
#pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}) — this builds a small side-by-side table with two columns:
comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(comparison.head(10))

#This imports 3 tools from scikit-learn, each one calculates a different kind of "how good is this model, overall, across ALL 1322 test students" score.
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#MAE = "On average, how wrong was I, in real units (degrees, or exam points)?"
#MSE = "Same as MAE, but I really don't like big embarrassing mistakes, so I punish those extra hard."
#R² = "Compared to not even trying (just guessing the average), how much did my model actually help?"
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.4f}")

from sklearn.tree import DecisionTreeRegressor

# Create and train a Decision Tree model
tree_model = DecisionTreeRegressor(random_state=42)
tree_model.fit(X_train, y_train)

# Predict on the test set
tree_pred = tree_model.predict(X_test)

# Evaluate it
tree_mae = mean_absolute_error(y_test, tree_pred)
tree_mse = mean_squared_error(y_test, tree_pred)
tree_r2 = r2_score(y_test, tree_pred)

print("\n--- Decision Tree Results ---")
print(f"MAE: {tree_mae:.2f}")
print(f"MSE: {tree_mse:.2f}")
print(f"R² Score: {tree_r2:.4f}")

from sklearn.ensemble import RandomForestRegressor

# Create and train a Random Forest model
forest_model = RandomForestRegressor(random_state=42)
forest_model.fit(X_train, y_train)

# Predict on the test set
forest_pred = forest_model.predict(X_test)

# Evaluate it
forest_mae = mean_absolute_error(y_test, forest_pred)
forest_mse = mean_squared_error(y_test, forest_pred)
forest_r2 = r2_score(y_test, forest_pred)

print("\n--- Random Forest Results ---")
print(f"MAE: {forest_mae:.2f}")
print(f"MSE: {forest_mse:.2f}")
print(f"R² Score: {forest_r2:.4f}")

import joblib

# Save the trained Linear Regression model to the models/ folder
joblib.dump(model, '../models/linear_regression_model.pkl')
print("Model saved to ../models/linear_regression_model.pkl")