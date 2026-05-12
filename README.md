# MedPredict AI — Disease Prediction System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![Flask](https://img.shields.io/badge/Flask-3.1.3-black?style=flat-square&logo=flask)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18.0-orange?style=flat-square&logo=tensorflow)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-F7931E?style=flat-square&logo=scikit-learn)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=flat-square&logo=docker)
![Railway](https://img.shields.io/badge/Deployed-Railway-0B0D0E?style=flat-square&logo=railway)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

**An AI-powered web application that predicts diseases from patient symptoms using an ensemble of classical ML models and a deep learning neural network.**

[ Live Demo](https://disease-predictor-app.up.railway.app/) · [ Report](#) · [ Issues](https://github.com/MRehanMehdi/disease-prediction/issues)

</div>

---

## Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Demo](#-demo)
- [System Architecture](#-system-architecture)
- [NLP Pipeline](#-nlp-pipeline)
- [Models & Performance](#-models--performance)
- [Dataset](#-dataset)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Docker Deployment](#-docker-deployment)
- [API Reference](#-api-reference)
- [Limitations & Disclaimer](#-limitations--disclaimer)
- [Future Work](#-future-work)
- [Author](#-author)

---

## Overview

MedPredict AI is a comprehensive, cloud-deployed disease prediction web application built as a final project for the **Artificial Intelligence course (BS Computer Science)**. It accepts patient symptoms through multiple input modalities — natural language text, a structured symptom checklist, or an uploaded medical document — and predicts the most probable disease using an ensemble of four machine learning models.

The system covers **41 disease categories** across **131 binary symptom features**, trained on 4,920 patient records from a curated Kaggle dataset. All four models achieved **100% accuracy** on the held-out test set, reflecting the clean, well-structured and linearly separable nature of the dataset.

---

## Features

### User-Facing
| Feature | Description |
|---|---|
| Natural Language Input | Describe symptoms in plain English — the NLP pipeline handles the rest |
| Symptom Checklist | Select from a structured list of 131 standardized symptoms |
| Document Upload | Upload a PDF or image medical report for automated symptom extraction (OCR) |
| Dual Model Inference | Every prediction runs through both Random Forest and Neural Network simultaneously |
| Top-3 Predictions | Both models return the top 3 most probable diseases with confidence percentages |
| Severity Assessment | Weighted risk score (Low / Medium / High) computed from reported symptoms |
| Disease Info Panel | Clinical description of the top predicted disease displayed after prediction |
| Precautionary Advice | Up to 4 evidence-based precautionary recommendations per prediction |
| PDF Report Export | Downloadable, professionally formatted health assessment PDF report |

### Technical
- RESTful API architecture (Flask)
- Alias-enhanced multi-stage NLP pipeline (200+ symptom phrase mappings)
- Tesseract OCR integration for scanned medical documents
- Containerized deployment via Docker
- Railway cloud hosting with automatic SSL and DNS

---

## Demo

> **Live at:** [https://disease-predictor-app.up.railway.app/](https://disease-predictor-app.up.railway.app/)

The web interface supports three input modes:

1. **Text input** — type symptoms naturally (e.g., *"I have a high fever, headache, and joint pain"*)
2. **Checklist** — select from the full list of 131 standard symptoms
3. **Document upload** — upload a PDF or image; symptoms are extracted via OCR

After prediction, the system shows disease name, confidence scores from both models, severity level, clinical description, and precautionary recommendations. A PDF report can be exported on demand.

---

## System Architecture

```
┌────────────────────────────────────────────────────────┐
│               HTML / CSS / JavaScript Frontend          │
│         (Text Input / Checklist / File Upload)          │
└──────────────────────┬─────────────────────────────────┘
                       │ HTTP POST
┌──────────────────────▼─────────────────────────────────┐
│                  Flask Backend (app.py)                  │
│   /predict    /predict-doc    /export-pdf               │
└──────────────────────┬─────────────────────────────────┘
                       │
┌──────────────────────▼─────────────────────────────────┐
│               Core Utility Module (utils.py)            │
│  ┌──────────────┐  ┌────────────┐  ┌────────────────┐  │
│  │ NLP Pipeline │  │ OCR / PDF  │  │ Severity Score │  │
│  │ (3 stages)   │  │ Extraction │  │ & Disease Info │  │
│  └──────┬───────┘  └─────┬──────┘  └───────┬────────┘  │
│         └────────────────▼──────────────────┘           │
│                   Model Inference                        │
│  ┌────────────┐  ┌──────────┐  ┌────────┐  ┌────────┐  │
│  │Random Forest│ │Dec. Tree │  │Naive   │  │Neural  │  │
│  │(Primary)   │  │          │  │Bayes   │  │Network │  │
│  └────────────┘  └──────────┘  └────────┘  └────────┘  │
└────────────────────────────────────────────────────────┘
                       │
              Docker Container → Railway Cloud
```

All three tiers are packaged in a single Docker container. Gunicorn serves the app on port 8080 in production.

---

## NLP Pipeline

The multi-stage pipeline in `extract_symptoms_from_text()` (utils.py) allows users to describe symptoms in natural language:

### Stage 1 — Alias Dictionary Lookup
A hand-crafted `SYMPTOM_ALIASES` dictionary maps **200+ natural language phrases** to standardized symptom identifiers, covering fever, body pain, respiratory, urinary, skin, gastrointestinal, and neurological domains.

```
"running a fever"  →  high_fever
"loose motions"    →  diarrhoea
"burning up"       →  high_fever
"watery stool"     →  diarrhoea
```

### Stage 2 — Direct String Matching
User input is compared against all 131 symptom identifiers in both underscore form (`skin_rash`) and space-separated form (`skin rash`), catching exact vocabulary matches.

### Stage 3 — RapidFuzz Fuzzy Matching
Input is tokenized into unigrams, bigrams, and trigrams. Each n-gram is scored against the symptom vocabulary using `fuzz.ratio` with a similarity threshold of **72**, handling typos, spelling variations, and partial matches.

### Document Extraction
`extract_symptoms_from_document()` extends the pipeline for uploaded files:
- **PDFs** → PyMuPDF (primary) + pdfplumber (fallback)
- **Images** → pytesseract OCR with PIL preprocessing

---

## Models & Performance

### Training Results

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Decision Tree | 100.00% | 100.00% | 100.00% | 100.00% |
| Random Forest | 100.00% | 100.00% | 100.00% | 100.00% |
| Naive Bayes | 100.00% | 100.00% | 100.00% | 100.00% |
| Neural Network | 100.00% | 100.00% | 100.00% | 100.00% |

 Perfect accuracy reflects the clean, structured, and linearly separable nature of the dataset — not a claim of real-world clinical generalizability. See [Limitations](#-limitations--disclaimer).

### Neural Network Architecture

```
Input: 131 binary symptom features
────────────────────────────────────────
Dense(256) → BatchNorm → Dropout → ReLU
Dense(128) → BatchNorm → Dropout → ReLU
Dense(64)  → Dropout → ReLU
Dense(41)  → Softmax
────────────────────────────────────────
Total Parameters: 79,145 (Trainable: 78,377)
```

- **Optimizer:** Adam | **Loss:** Categorical Cross-Entropy
- Validation accuracy reached **100% by Epoch 2**; early stopping triggered at **Epoch 17**

### Model Files

| File | Size | Description |
|---|---|---|
| `best_model.pkl` | 6,990 KB | Random Forest — primary prediction engine |
| `decision_tree.pkl` | 58.5 KB | Decision Tree classifier |
| `naive_bayes.pkl` | 94.0 KB | Gaussian Naive Bayes classifier |
| `neural_network.keras` | 969.4 KB | TensorFlow/Keras Neural Network |
| `symptom_list.pkl` | 2.3 KB | Ordered 131-symptom feature list |
| `label_encoder.pkl` | 1.1 KB | Class index → disease name mapping |

---

## Dataset

| Attribute | Details |
|---|---|
| Source | [Kaggle — Disease Symptom Description Dataset](https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset) |
| Total Samples | 4,920 patient records |
| Total Features | 131 binary symptom features |
| Target Classes | 41 distinct diseases |
| Symptom Columns | 17 columns (Symptom_1 through Symptom_17) |
| Missing Values | Present in Symptom_4–Symptom_17; filled with 0 |
| Train / Test Split | 80% training (3,936) / 20% testing (984) |

- **Class balance:** Each of the 41 diseases has exactly 120 samples — perfectly balanced
- **Most frequent symptoms:** fatigue, vomiting, high_fever, loss_of_appetite, nausea, headache

**Supplementary files used at inference time:**
- `symptom_Description.csv` — Clinical descriptions for all 41 diseases
- `symptom_precaution.csv` — Up to 4 precautionary recommendations per disease
- `Symptom-severity.csv` — Numerical severity weight (1–7) per symptom

---

## Tech Stack

| Category | Technology | Version |
|---|---|---|
| Language | Python | 3.11 |
| Web Framework | Flask | 3.1.3 |
| ML Libraries | scikit-learn | 1.6.1 |
| Deep Learning | TensorFlow / Keras | 2.18.0 |
| Data Processing | pandas, NumPy | 2.2.3 / 2.0.2 |
| NLP / Fuzzy | RapidFuzz | 3.14.5 |
| PDF Generation | ReportLab | 4.5.0 |
| PDF Parsing | PyMuPDF, pdfplumber | 1.25.5 / 0.11.9 |
| OCR | pytesseract, Pillow | 0.3.13 / 11.1.0 |
| Model Persistence | joblib | 1.4.2 |
| Deployment | Gunicorn + Docker + Railway | — |

---

## Project Structure

```
disease-prediction/
│
├── app.py                    # Flask entry point & API route definitions
├── utils.py                  # Core module: NLP pipeline, inference, severity, PDF generation
├── requirements.txt          # Python dependencies
├── Dockerfile                # Docker container build instructions
├── Procfile                  # Gunicorn startup command for Railway
├── .python-version           # Pins Python 3.11 runtime
│
├── templates/
│   └── medpredict_ui.html    # Frontend HTML/CSS/JavaScript interface
│
├── models/
│   ├── best_model.pkl        # Random Forest (primary model)
│   ├── decision_tree.pkl     # Decision Tree
│   ├── naive_bayes.pkl       # Gaussian Naive Bayes
│   ├── neural_network.keras  # TensorFlow Neural Network
│   ├── symptom_list.pkl      # Ordered symptom feature list
│   └── label_encoder.pkl     # Label encoder for Neural Network output
│
├── data/
│   ├── symptom_Description.csv
│   ├── symptom_precaution.csv
│   └── Symptom-severity.csv
│
└── notebooks/
    ├── 01_EDA_and_Preprocessing.ipynb
    ├── 02_Model_Training.ipynb
    └── 03_Neural_Network.ipynb
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- Tesseract OCR installed on your system
  - **Windows:** [Tesseract installer](https://github.com/UB-Mannheim/tesseract/wiki)
  - **Ubuntu/Debian:** `sudo apt install tesseract-ocr`
  - **macOS:** `brew install tesseract`

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/MRehanMehdi/disease-prediction.git
cd disease-prediction

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate       # Linux/macOS
venv\Scripts\activate          # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the application
python app.py
```

The app will be available at `http://localhost:5000`.

---

## Docker Deployment

```bash
# Build the Docker image
docker build -t medpredict-ai .

# Run the container
docker run -p 8080:8080 medpredict-ai
```

The app will be available at `http://localhost:8080`.

**Docker configuration:**
- Base image: `python:3.11-slim`
- System packages: `tesseract-ocr`, `tesseract-ocr-eng`, `libgl1`
- Production server: Gunicorn on port 8080

---

## API Reference

### `POST /predict`
Accepts natural language symptom text and returns prediction results.

**Request body:**
```json
{
  "symptoms": "I have high fever, headache, and joint pain"
}
```

**Response:**
```json
{
  "disease": "Dengue",
  "rf_predictions": [
    { "disease": "Dengue", "confidence": 0.95 },
    ...
  ],
  "nn_predictions": [...],
  "severity": "High",
  "severity_score": 14,
  "description": "...",
  "precautions": ["Drink plenty of fluids", "..."]
}
```

---

### `POST /predict-doc`
Accepts a multipart file upload (PDF or image) with optional supplementary text.

**Form data:**
- `file` — PDF or image file
- `extra_text` *(optional)* — Additional symptom description

---

### `POST /export-pdf`
Accepts a full prediction result payload and returns a formatted PDF health assessment report.

---

## Limitations & Disclaimer

> **This application is intended for educational and preliminary screening purposes only. It is NOT a diagnostic tool and must NOT be used as a substitute for professional medical consultation.**

The 100% test accuracy is a direct result of the dataset's clean, structured, non-overlapping symptom signatures — each disease maps to a distinct set of binary features, making the classes trivially linearly separable. Real-world clinical deployment would require:

- Diverse, noisy, real patient data with comorbidities and incomplete symptom reporting
- Expanded disease coverage beyond the 41 categories in this dataset
- Rigorous clinical validation by licensed medical professionals

This disclaimer is embedded in both the web interface and all generated PDF reports.

---

## Future Work

- [ ] **Real-world dataset integration** — noisier clinical data with comorbidities for more generalizable models
- [ ] **Expanded disease coverage** — rare diseases, chronic conditions, mental health disorders
- [ ] **Conversational AI interface** — LLM-powered chatbot for natural symptom collection
- [ ] **Lab result integration** — include basic lab parameters alongside symptom data
- [ ] **Mobile application** — native iOS and Android apps
- [ ] **Multilingual support** — Urdu and other South Asian regional languages
- [ ] **Clinical validation** — partner with medical professionals to validate against real patient cases
- [ ] **Explainability module** — SHAP / LIME visualizations to explain which symptoms drove each prediction

---

## Author

**Muhammad Rehan Mehdi**
BS Computer Science — Artificial Intelligence Course

- GitHub: [@MRehanMehdi](https://github.com/MRehanMehdi)
- Fiverr: [View Gigs](https://www.fiverr.com)

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

- [Kaggle — Disease Symptom Description Dataset](https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset)
- [scikit-learn](https://scikit-learn.org/) · [TensorFlow](https://www.tensorflow.org/) · [Flask](https://flask.palletsprojects.com/) · [RapidFuzz](https://github.com/rapidfuzz/RapidFuzz) · [ReportLab](https://www.reportlab.com/) · [Railway](https://railway.app/)

---

<div align="center">
  <sub>Built with  as a Final AI Course Project · BS Computer Science · 2026</sub>
</div>
