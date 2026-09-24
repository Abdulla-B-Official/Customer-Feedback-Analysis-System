# Customer Feedback Analysis System

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge)
![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-TF--IDF_%26_Logistic_Regression-F7931E?style=for-the-badge)
![Pandas](https://img.shields.io/badge/Pandas-Data_Processing-150458?style=for-the-badge)
![FastAPI](https://img.shields.io/badge/FastAPI-API_Service-009688?style=for-the-badge)
![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge)

# Customer Feedback Analysis

An end-to-end NLP project that analyzes customer reviews and predicts **positive, neutral, or negative sentiment**.

The project uses VADER to create sentiment labels from review text, spaCy for text cleaning and lemmatization, TF-IDF for feature extraction, and Logistic Regression for classification. A FastAPI service exposes the trained model through a local web application.

## Local Web App

The project includes a **Nordic Frost** frontend built with plain HTML, CSS, and JavaScript. You do **not** need React, Node.js, npm, or a separate frontend server.

When the FastAPI server starts, it serves the web application at:

```text
http://127.0.0.1:8000
```

## What You Need

- Windows 10/11
- Python 3.10 recommended
- VS Code
- Internet connection for the first-time package installation and spaCy model download

> The trained model and TF-IDF vectorizer are already included in the `models/` folder in this project copy. You do not need to retrain the model just to open the local web app.

## Project Structure

```text
Customer-Feedback-Analysis/
│
├── data/
│   ├── raw/
│   │   └── Reviews.csv
│   ├── processed/
│   │   └── cleaned_reviews.csv
│   └── sample/
│
├── models/
│   ├── sentiment_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── frontend/
│   ├── index.html
│   └── static/
│       ├── css/
│       │   └── style.css
│       └── js/
│           └── app.js
│
├── src/
│   ├── api/
│   │   └── app.py
│   ├── data/
│   │   ├── load_data.py
│   │   ├── preprocess.py
│   │   └── process_dataset.py
│   ├── features/
│   │   └── vectorizer.py
│   └── models/
│       ├── train.py
│       ├── predict.py
│       └── evaluate.py
│
├── main.py
├── requirements.txt
└── README.md
```

## 1. Open the Project

Open the project folder in VS Code.

Then open the VS Code terminal:

```powershell
cd "C:\path\to\Customer-Feedback-Analysis"
```

Replace the path above with the actual folder location on your computer.

## 2. Create a Virtual Environment

Run:

```powershell
python -m venv .venv
```

Activate it in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

After activation, the terminal should show something similar to:

```text
(.venv) PS C:\...\Customer-Feedback-Analysis>
```

### PowerShell Activation Error

If Windows blocks the activation script, run this once:

```powershell
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then activate the environment again:

```powershell
.\.venv\Scripts\Activate.ps1
```

## 3. Install the Python Libraries

Run:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Install the spaCy English Model

The preprocessing code uses `en_core_web_sm` for tokenization and lemmatization.

Run:

```powershell
python -m spacy download en_core_web_sm
```

## 5. Download the NLTK VADER Lexicon

Run:

```powershell
python -m nltk.downloader vader_lexicon
```

The application also checks for the VADER lexicon automatically when the preprocessing module is imported.

## 6. Check the Model Files

Before starting the web app, make sure these two files exist:

```text
models/sentiment_model.pkl
models/tfidf_vectorizer.pkl
```

They are already included in this project copy.

If those files are missing, the API cannot make predictions. In that case, rebuild the model using the training pipeline described below.

## 7. Start the Localhost Web App

From the **project root folder**, run:

```powershell
python -m uvicorn src.api.app:app --reload
```

You should see a message similar to:

```text
Uvicorn running on http://127.0.0.1:8000
```

Open this in your browser:

```text
http://127.0.0.1:8000
```

The **Nordic Frost Customer Feedback Analysis** page should open.

## 8. How to Use the Website

1. Type a customer review into the review box.
2. Click **Analyze Feedback**.
3. The browser sends the text to the local FastAPI `/predict` endpoint.
4. The backend cleans the text using the same preprocessing used by the model.
5. The saved TF-IDF vectorizer converts the text into numerical features.
6. The saved Logistic Regression model predicts the sentiment.
7. The website displays **Positive**, **Neutral**, or **Negative**.

You can also click the sample buttons:

```text
Positive
Neutral
Negative
```

### Keyboard Shortcut

Press `Ctrl + Enter` inside the review box to run the prediction.

## 9. Test the API Directly

FastAPI also provides interactive API documentation.

Open:

```text
http://127.0.0.1:8000/docs
```

Use the `POST /predict` endpoint with:

```json
{
  "text": "The product quality is excellent and I am very happy with my purchase."
}
```

The response will look like:

```json
{
  "sentiment": "positive"
}
```

You can also check the API health endpoint:

```text
http://127.0.0.1:8000/health
```

Expected response:

```json
{
  "status": "ok",
  "model_loaded": true
}
```

## 10. If You Need to Rebuild the Dataset

The current project already contains:

```text
data/raw/Reviews.csv
data/processed/cleaned_reviews.csv
```

To process the raw dataset again, run from the project root:

```powershell
python main.py
```

This runs `src/data/process_dataset.py`, which samples the dataset, creates VADER sentiment labels, cleans the text with spaCy, and saves the result to:

```text
data/processed/cleaned_reviews.csv
```

## 11. If You Need to Retrain the Model

Run the model training script from the project root:

```powershell
python src/models/train.py
```

This updates:

```text
models/sentiment_model.pkl
models/tfidf_vectorizer.pkl
```

Then restart the FastAPI server so it loads the updated model files.

## 12. Test from the Command Line

To test the trained model without the website, run:

```powershell
python src/models/predict.py
```

Enter a review when prompted.

Press `Ctrl + C` to stop the prediction loop.

## 13. Evaluate the Model

Run:

```powershell
python src/models/evaluate.py
```

The script prints:

- Precision
- Recall
- F1-score
- Confusion matrix

The confusion matrix uses this class order:

```text
negative
neutral
positive
```

## 14. Stop the Local Server

In the terminal running Uvicorn, press:

```text
Ctrl + C
```

## Troubleshooting

### `ModuleNotFoundError: No module named 'spacy'`

Make sure the virtual environment is active, then run:

```powershell
pip install -r requirements.txt
```

### `Can't find model 'en_core_web_sm'`

Run:

```powershell
python -m spacy download en_core_web_sm
```

### `LookupError: vader_lexicon`

Run:

```powershell
python -m nltk.downloader vader_lexicon
```

### `FileNotFoundError` for a model file

Check that these files exist:

```text
models/sentiment_model.pkl
models/tfidf_vectorizer.pkl
```

### Browser shows connection refused

Make sure Uvicorn is running and use:

```text
http://127.0.0.1:8000
```

Do not open `index.html` directly with `file:///...`. The page is designed to run through FastAPI.

### Port 8000 is already in use

Start the server on another port:

```powershell
python -m uvicorn src.api.app:app --reload --port 8080
```

Then open:

```text
http://127.0.0.1:8080
```

## Technology Stack

```text
Python
Pandas
NumPy
NLTK / VADER
spaCy
Scikit-learn
TF-IDF
Logistic Regression
Joblib
FastAPI
Uvicorn
HTML
CSS
JavaScript
```

## Project Flow

```text
Customer Review
      ↓
Nordic Frost Web App
      ↓
FastAPI /predict
      ↓
Text Cleaning + Lemmatization
      ↓
Saved TF-IDF Vectorizer
      ↓
Saved Logistic Regression Model
      ↓
Sentiment Prediction
      ↓
Positive / Neutral / Negative
```

## Important Note

Run all Python commands from the **project root folder** — the folder that contains `main.py`, `requirements.txt`, `models/`, `data/`, `src/`, and `frontend/`.

The main web command is:

```powershell
python -m uvicorn src.api.app:app --reload
```

Then open:

```text
http://127.0.0.1:8000
```
