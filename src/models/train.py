import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# 1. Load Data
df = pd.read_csv("data/processed/cleaned_reviews.csv")

# 2. Handle missing text values (Fill NaN with empty string or drop)
df["clean_review"] = df["clean_review"].fillna("")
df = df.dropna(subset=["sentiment"])  # Ensure target labels aren't missing

X = df["clean_review"]
y = df["sentiment"]

# 3. Vectorization
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2)
)

X = vectorizer.fit_transform(X)

# 4. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 5. Model Training
model = LogisticRegression(class_weight="balanced", max_iter=1000)
model.fit(X_train, y_train)

# 6. Save Model & Vectorizer
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/sentiment_model.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

print("Training Completed Successfully")