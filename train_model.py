import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib

print("Starting model training process...")

# 1. Load Data
print("Loading consolidated dataset...")
df = pd.read_csv('data/all_clauses.csv')
df.dropna(subset=['clause_text', 'clause_type'], inplace=True)

# Define our features (X) and labels (y)
X = df['clause_text']
y = df['clause_type']
print(f"Data loaded successfully with {len(df)} samples.")

# 2. Split Data into Training and Testing sets
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# 'stratify=y' is important for imbalanced datasets to keep the class distribution similar in train/test splits.

# 3. Create a Machine Learning Pipeline
print("Defining the model pipeline...")
# A pipeline chains together multiple steps. Here, it will:
#   a. Convert our text into numerical vectors (TfidfVectorizer)
#   b. Train a classification model on those vectors (LogisticRegression)
model_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 2))),
    ('clf', LogisticRegression(max_iter=1000, random_state=42))
])

# 4. Train the Model
print("Training the model... This may take a few moments.")
model_pipeline.fit(X_train, y_train)
print("Model training complete.")

# 5. Evaluate the Model
print("Evaluating model performance on the test set...")
y_pred = model_pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

# Print a detailed report of precision, recall, f1-score for each class
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 6. Save the Trained Model
print("Saving the trained model pipeline...")
model_path = 'models/clause_classifier_pipeline.pkl'
joblib.dump(model_pipeline, model_path)
print(f"Model saved successfully to {model_path}")