import joblib
import pandas as pd

# Load the saved trained model
model = joblib.load('../models/linear_regression_model.pkl')
print("Model loaded successfully.")

# Create one new, fake student's data
# Columns must match the exact order used in training (X.columns)
new_student = pd.DataFrame({
    'Hours_Studied': [25],
    'Attendance': [90],
    'Parental_Involvement': [2],       # High
    'Access_to_Resources': [2],        # High
    'Extracurricular_Activities': [1], # Yes
    'Sleep_Hours': [7],
    'Previous_Scores': [80],
    'Motivation_Level': [2],           # High
    'Internet_Access': [1],            # Yes
    'Tutoring_Sessions': [2],
    'Family_Income': [1],              # Medium
    'Teacher_Quality': [2],            # High
    'School_Type': [0],                # Public
    'Physical_Activity': [3],
    'Learning_Disabilities': [0],      # No
    'Parental_Education_Level': [1],   # College
    'Distance_from_Home': [0],         # Near
    'Gender': [0],                     # Male
    'Peer_Influence_Negative': [0],
    'Peer_Influence_Neutral': [0],
    'Peer_Influence_Positive': [1],    # Positive
})

# Predict this student's exam score
prediction = model.predict(new_student)
print(f"Predicted Exam Score: {prediction[0]:.2f}")