import os
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv("data/processed/cleaned_reviews.csv")

# Fill missing values with empty strings
df["clean_review"] = df["clean_review"].fillna("")

vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X = vectorizer.fit_transform(df["clean_review"])

# Create directory if it doesn't exist
os.makedirs("models", exist_ok=True)

joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
print(X.shape)