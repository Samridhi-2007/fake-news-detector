# 📰 Fake News Detector

An end-to-end Machine Learning application that detects whether a news article is likely to be **Real News** or **Fake News** using Natural Language Processing (NLP) techniques.

The project includes:

- Text preprocessing using NLTK
- TF-IDF feature extraction
- Logistic Regression classifier
- FastAPI backend API
- Streamlit frontend interface
- Docker containerization
- Explainability using LIME (Work in Progress)

---

## 🚀 Features

- Detects fake and real news articles
- Confidence score for each prediction
- Text cleaning and preprocessing pipeline
- FastAPI REST API
- Interactive Streamlit UI
- Dockerized deployment
- Modular project structure

---

## 🛠 Tech Stack

### Machine Learning

- Scikit-Learn
- Logistic Regression
- TF-IDF Vectorizer

### NLP

- NLTK
- Stopword Removal
- Tokenization
- Text Cleaning

### Backend

- FastAPI
- Uvicorn

### Frontend

- Streamlit

### Deployment

- Docker

---

## 📂 Project Structure

```text
fake-news-detector/
│
├── app/
│   ├── frontend.py
│   ├── main.py
│   ├── schemas.py
│   ├── static/
│   └── templates/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│
├── src/
│   ├── explainability/
│   ├── feature_engineering/
│   ├── models/
│   ├── preprocessing/
│   ├── utils/
│   ├── predict.py
│   └── train.py
│
├── trained_models/
│   ├── logistic_model.pkl
│   └── tfidf.pkl
│
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### Clone Repository

```bash
git clone <repository-url>
cd fake-news-detector
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Download NLTK Resources

```bash
python download_nltk.py
```

---

## ▶️ Run Streamlit Application

```bash
streamlit run app/frontend.py
```

Open:

```text
http://localhost:8501
```

---

## ▶️ Run FastAPI Server

```bash
uvicorn app.main:app --reload
```

Open Swagger Documentation:

```text
http://127.0.0.1:8000/docs
```

---

## 🐳 Docker Setup

### Build Docker Image

```bash
docker build -t fake-news-detector .
```

### Run Docker Container

```bash
docker run -p 8501:8501 fake-news-detector
```

Open:

```text
http://localhost:8501
```

---

## 📊 Model Pipeline

```text
News Article
      ↓
Text Cleaning
      ↓
Tokenization
      ↓
Stopword Removal
      ↓
TF-IDF Vectorization
      ↓
Logistic Regression
      ↓
Prediction
      ↓
Confidence Score
```

---

## 🧪 Example

### Input

```text
Scientists have developed a new vaccine that showed promising results in trials.
```

### Output

```json
{
  "prediction": "Real News",
  "confidence": 97.34
}
```

---

## 🔍 Explainability

LIME is integrated to identify important words contributing to the prediction.

Example:

```text
Prediction: Fake News

Important Words:
- shocking
- secret
- miracle
- revealed
```

---

## 📈 Future Improvements

- Transformer-based models (BERT)
- Advanced fact-checking pipeline
- News source credibility scoring
- Explainability dashboard
- Cloud deployment
- CI/CD integration

---

## 👩‍💻 Author

Samridhi Prakash

Built as an NLP and Machine Learning project to explore Fake News Detection using modern ML workflows and deployment practices.
