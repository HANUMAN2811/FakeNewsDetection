# 📰 Fake News Detection System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.x-black?logo=flask)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.x-F7931E?logo=scikit-learn&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-NLP-green)
![License](https://img.shields.io/badge/License-MIT-blue)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

> An end-to-end Machine Learning web application that classifies news articles as **REAL ✅** or **FAKE 🚨** using NLP + TF-IDF + Multiple ML Classifiers.

---

## 📌 Table of Contents
- [Overview](#-overview)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Tech Stack](#-tech-stack)
- [Dataset](#-dataset)
- [ML Pipeline](#-ml-pipeline)
- [Installation & Setup](#-installation--setup)
- [Usage](#-usage)
- [API Reference](#-api-reference)
- [Model Performance](#-model-performance)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🔍 Overview

Fake news spreads rapidly in the digital age and causes real-world harm. This project provides a complete solution:

- **NLP Preprocessing** — Cleaning, tokenisation, stop-word removal, stemming
- **TF-IDF Vectorisation** — Converts text to numerical features
- **Multiple ML Models** — Passive Aggressive, Logistic Regression, Naive Bayes
- **Flask REST API** — `/api/predict` endpoint returns JSON
- **Web UI** — Paste any article and get an instant verdict

---

## ✨ Features

| Feature | Status |
|---|---|
| Text cleaning pipeline (URL, HTML, punctuation, digits) | ✅ |
| TF-IDF vectorisation with bi-grams | ✅ |
| 3 ML models compared at training time | ✅ |
| Best model auto-selected and saved | ✅ |
| Flask REST API with JSON responses | ✅ |
| Responsive Web UI | ✅ |
| Confidence score in prediction | ✅ |
| Unit tests (pytest) | ✅ |
| Jupyter Notebook for EDA + Training walkthrough | ✅ |
| `.env` based config | ✅ |

---

## 📁 Project Structure

```
FakeNewsDetection/               ← GitHub repo root
│
├── data/
│   ├── raw/                     ← Place Fake.csv & True.csv here
│   │   ├── Fake.csv             ← (download from Kaggle — not tracked by Git)
│   │   ├── True.csv             ← (download from Kaggle — not tracked by Git)
│   │   └── .gitkeep
│   └── processed/
│       ├── news_dataset.csv     ← Auto-generated after training
│       └── .gitkeep
│
├── models/                      ← Auto-created; saved .pkl files go here
│   └── .gitkeep
│
├── notebooks/
│   └── EDA_and_Training.ipynb   ← Full walkthrough notebook
│
├── src/
│   ├── __init__.py
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── text_cleaner.py      ← NLP preprocessing functions
│   ├── models/
│   │   ├── __init__.py
│   │   └── train_model.py       ← Training + evaluation script
│   └── utils/
│       ├── __init__.py
│       └── helpers.py           ← Model loading, prediction, validation
│
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py    ← Unit tests for text_cleaner
│   └── test_helpers.py          ← Unit tests for helpers
│
├── static/
│   ├── css/
│   │   └── style.css            ← Web UI stylesheet
│   └── js/
│       └── script.js            ← Frontend JavaScript
│
├── templates/
│   └── index.html               ← Jinja2 HTML template
│
├── docs/
│   └── report.md                ← Project report
│
├── scripts/
│   └── prepare_data.py          ← Merges raw CSVs into processed dataset
│
├── app.py                       ← Flask app entry point
├── config.py                    ← All paths & hyper-parameters
├── requirements.txt             ← pip dependencies
├── .env.example                 ← Environment variable template
├── .gitignore                   ← Git ignore rules
├── LICENSE                      ← MIT License
└── README.md                    ← You are here
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.9+ |
| NLP | NLTK (tokenise, stopwords, stemming) |
| Vectoriser | TF-IDF — `sklearn.feature_extraction.text` |
| ML Models | Passive Aggressive Classifier, Logistic Regression, Multinomial Naive Bayes |
| Serialisation | Joblib |
| Web Framework | Flask 2.x |
| Frontend | HTML5, CSS3, Vanilla JavaScript |
| Testing | pytest |
| Notebook | Jupyter |

---

## 📊 Dataset

Download the **ISOT Fake News Dataset** from Kaggle:

🔗 https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

| File | Rows | Label |
|---|---|---|
| `Fake.csv` | ~23,500 | FAKE (1) |
| `True.csv` | ~21,400 | REAL (0) |

**Columns:** `title`, `text`, `subject`, `date`

Place both files in `data/raw/` before training.

---

## 🤖 ML Pipeline

```
Raw CSV (Fake.csv + True.csv)
        │
        ▼
  Merge & Label  →  label=1 (FAKE)  /  label=0 (REAL)
        │
        ▼
  Combine title + text  →  "content" column
        │
        ▼
  Text Cleaning
    • lowercase
    • remove URLs / HTML
    • remove punctuation & digits
    • tokenise (NLTK word_tokenize)
    • remove English stop-words
    • Porter stemming
        │
        ▼
  TF-IDF Vectorisation
    max_features=5000, ngram_range=(1,2), sublinear_tf=True
        │
        ▼
  Train / Compare 3 classifiers
    • PassiveAggressiveClassifier
    • LogisticRegression
    • MultinomialNaiveBayes
        │
        ▼
  Best model saved → models/model.pkl
  Vectoriser saved → models/vectorizer.pkl
        │
        ▼
  Flask API  →  POST /api/predict  →  { prediction, confidence, label }
```

---

## ⚙️ Installation & Setup

### 1 — Clone the repo
```bash
git clone https://github.com/YOUR-USERNAME/FakeNewsDetection.git
cd FakeNewsDetection
```

### 2 — Create & activate a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### 4 — Download NLTK data
```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

### 5 — Configure environment
```bash
cp .env.example .env
# Edit .env if you want to change host/port
```

### 6 — Add dataset files
Place `Fake.csv` and `True.csv` inside `data/raw/`

### 7 — Train the model
```bash
python src/models/train_model.py
```
This creates `models/model.pkl` and `models/vectorizer.pkl`

### 8 — Start the app
```bash
python app.py
```
Open **http://127.0.0.1:5000** in your browser.

---

## 🚀 Usage

### Web UI
1. Open `http://127.0.0.1:5000`
2. Paste any news article text
3. Click **Check News**
4. See the verdict + confidence score

### REST API
```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Breaking: Scientists discover new planet in solar system..."}'
```

**Response:**
```json
{
  "prediction": "REAL",
  "label": 0,
  "confidence": 0.9231
}
```

---

## 📡 API Reference

| Endpoint | Method | Body | Response |
|---|---|---|---|
| `/` | GET | — | Web UI |
| `/api/predict` | POST | `{"text": "..."}` | `{prediction, label, confidence}` |
| `/api/health` | GET | — | `{status: "ok"}` |

---

## 📈 Model Performance

| Model | Accuracy | F1 Score |
|---|---|---|
| Passive Aggressive Classifier | ~97% | ~0.97 |
| Logistic Regression | ~95% | ~0.95 |
| Multinomial Naive Bayes | ~93% | ~0.93 |

> Results vary slightly by random seed; best model is auto-selected at training time.

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## 🤝 Contributing

1. Fork this repo
2. `git checkout -b feature/your-feature`
3. `git commit -m "Add: your feature description"`
4. `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 👨‍💻 Author

**Your Name**
- GitHub: [@your-username](https://github.com/your-username)
- LinkedIn: [your-profile](https://linkedin.com/in/your-profile)

---
> ⭐ Star this repo if it helped you!
