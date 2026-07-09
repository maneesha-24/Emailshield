# 🛡️ EmailShield — AI-Powered Email Risk Detection

EmailShield analyzes any email and classifies it into one of 7 categories
Legitimate, Spam/Newsletter, Suspicious, Scam, Phishing, Spoofed, Phishing+Spoofed, with a plain-English explanation of why.

Built with a BERT + SVM ensemble trained on 70,423 real emails (CEAS 2008, Enron, Nazario Phishing Corpus)

---

## Features

- **7-category verdict system** — goes beyond simple safe/spam classification
- **BERT + SVM ensemble** (60/40 weighted) — 99.5%+ F1 score
- **Spoofing detection** — parses SPF/DKIM/DMARC headers and checks for brand impersonation
- **Natural language explanations** — every verdict includes a 3-paragraph plain-English breakdown
- **File upload support** — paste raw email text or upload `.eml` / `.txt` files

---

## Quick Start

```bash
git clone
cd EmailShield
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Then open `http://localhost:5000` in your browser.

The SVM model is included in this repo (`models/svm_model.pkl`, `models/vectorizer.pkl`), so the app works immediately after install. BERT is optional — see below.

---

## Optional: Enable BERT

The app runs on SVM alone out of the box. To enable the higher-accuracy BERT ensemble:

1. Train your own BERT model using the training pipeline (see `bert_detector.py` for the expected format)
2. Place the resulting model files in `models/bert_phishing_model/`
3. Restart the app — it will detect and load BERT automatically

Without this folder, the app automatically falls back to SVM-only and still works.
---

## Retraining From Scratch

The training datasets aren't included in this repo (size + licensing). To reproduce the full pipeline:

1. Download the dataset from Kaggle:
   [Phishing Email Dataset](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset)
   *(combines Enron, CEAS, Nazario, Ling, Nigerian Fraud, and SpamAssassin corpora)*
2. Extract `CEAS_08.csv`, `Enron.csv`, and `Nazario.csv` into `data/email dataset/`
3. Run the pipeline in order:
```bash
   python explore_data.py
   python preprocess.py
   python train_model.py
```
4. This regenerates `models/svm_model.pkl`, `vectorizer.pkl`, `lr_model.pkl`, `rf_model.pkl`

---

## Project Structure

```
EmailShield/
├── .gitignore
├── app.py                              # Flask app & routes
├── detector.py                         # Core analysis engine (verdict logic, explanations)
├── bert_detector.py                    # BERT model loading & inference
├── explore_data.py                     # Dataset merging & exploration
├── preprocess.py                       # Text cleaning pipeline
├── train_model.py                      # Trains SVM / LR / RF models
├── results_log.txt                     # Classical ML training results
├── requirements.txt
├── README.md
├── EmailShield_detailreport_file.pdf   # Full project report
├── templates/
│   └── index.html                      # Frontend UI
└── models/
    ├── svm_model.pkl                   # Included — required to run
    └── vectorizer.pkl                  # Included — required to run
```

---

## Results

Full methodology and detailed results are in `EmailShield_detailreport_file.pdf`. 
---

## How It Works

1. User pastes a raw email or uploads a file
2. Email parser splits headers from body
3. Feature extractor identifies URLs, urgency words, phishing phrases, sender domain
4. **Spoofing engine** reads SPF/DKIM/DMARC headers and checks display name against trusted brands
5. **ML engine** runs BERT (60% weight) + SVM (40% weight) on cleaned email text
6. Decision fusion combines both signals to assign one of 7 verdicts
7. Explainer generates a natural language explanation for the result

---

## Limitations

- Spoofing detection requires full raw email headers — pasting only the body shows "N/A" for auth checks
- Trained on English-language emails only
- Image-heavy marketing emails give the model less text to analyze
- Sophisticated, low-signal spear-phishing may score with lower confidence


## Disclaimer

This is an educational/portfolio project demonstrating applied NLP and ML for email security. It is not a production-grade security product and should not be relied on as a sole defense against phishing.

---
