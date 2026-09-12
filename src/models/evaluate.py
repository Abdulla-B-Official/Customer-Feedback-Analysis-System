import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# 1. Load Data
df = pd.read_csv("data/processed/cleaned_reviews.csv")

# 2. Clean missing values (Fill NaN strings & drop missing sentiment rows)
df["clean_review"] = df["clean_review"].fillna("")
df = df.dropna(subset=["sentiment"])

# 3. Load Models
model = joblib.load("models/sentiment_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# 4. Transform Text Features
X = vectorizer.transform(df["clean_review"])
y = df["sentiment"]

# 5. Train/Test Split (matches training configuration)
_, X_test, _, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 6. Predict & Evaluate
predictions = model.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, predictions))

# 7. Confusion Matrix Presentation
labels = ["negative", "neutral", "positive"]

cm = confusion_matrix(
    y_test,
    predictions,
    labels=labels
)

cm_table = pd.DataFrame(
    cm,
    index=["actual_negative", "actual_neutral", "actual_positive"],
    columns=["pred_negative", "pred_neutral", "pred_positive"]
)

print("\nConfusion Matrix (rows = actual, columns = predicted):")
print(cm_table)