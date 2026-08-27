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

# concept coding in train model
Imagine you finally figured out a recipe: "2 cups flour + 1 cup sugar + 3 eggs = perfect cake." That's your .fit() moment — you experimented and found the right recipe.

Now every time you bake a NEW cake using that same recipe, you're not "discovering" the recipe again — you're just using the recipe you already found. That's .predict().

# Concept model in ML
Model Comparison

I trained two different models to predict student exam scores and compared their performance:

Model	MAE	MSE	R² Score
Linear Regression	0.47	3.28	0.7681
Decision Tree	1.62	9.60	0.3210

Result: Linear Regression performed significantly better, explaining ~77% of the variation in exam scores compared to only ~32% for the Decision Tree.

Why: The relationships in this dataset (e.g., study hours and attendance vs. exam score) are gradual and fairly linear — students don't score dramatically differently at specific "cutoff points." Linear Regression is well-suited to this kind of smooth, steady trend, while Decision Trees perform better on data with sharp, rule-based patterns (e.g., "if X > 85 AND Y > 20, score jumps"). Since this data lacks those sharp cutoffs, the Decision Tree ended up learning noise rather than the true underlying pattern.

Key takeaway: No single algorithm is universally best — different models suit different data patterns. Testing multiple models before choosing one is a standard and necessary part of the ML workflow.
