# student-performance-predictor
A beginner machine learning project that predicts student performance using academic and attendance data.
Day1
1. Fixed your .gitignore
2. Built the real project folder structure
Added .gitkeep files to data/, models/, notebooks/, src/ so Git would actually track these empty folders.
3. Got real hands-on experience with messy data
Your first "Maths.csv" / "Portuguese.csv" turned out to be Excel files mislabeled as .csv.
4. Found and validated a proper dataset
5. Did your first real Data Understanding step
Used df.head() and df.info() to see the actual shape of your data — and spotted that 3 columns (Teacher_Quality, Parental_Education_Level, Distance_from_Home) have

# concept
80% (training set): the model studies real examples — attendance, sleep, hours studied → paired with the actual exam score — and learns the pattern connecting them.

20% (test set): the model is given attendance, sleep, hours studied for students it's never seen before — WITHOUT the real exam score — and has to guess. We then check its guesses against the real scores to see how good it actually is.
