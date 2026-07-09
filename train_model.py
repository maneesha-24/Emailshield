import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix,
                             classification_report)

# ── Load data ──────────────────────────────────────────────────────

print("Loading processed dataset...")
df = pd.read_csv("data/processed_emails.csv")
print(f"Total emails: {len(df)}")
print(f"Label distribution:\n{df['label'].value_counts()}\n")

X = df["clean_text"].astype(str)
y = df["label"]

# ── Split into train and test ──────────────────────────────────────
# 80% train, 20% test — standard split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training emails : {len(X_train)}")
print(f"Testing emails  : {len(X_test)}\n")

# ── Convert text to numbers using TF-IDF ──────────────────────────
# TF-IDF turns each email into a vector of word importance scores
# max_features=50000 means we track the 50,000 most important words

print("Converting text to TF-IDF vectors...")
vectorizer = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),   # single words AND pairs like "click here"
    min_df=2,             # ignore words appearing in only 1 email
    sublinear_tf=True     # compress very high word counts
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
print(f"Feature matrix shape: {X_train_vec.shape}\n")

# ── Train and evaluate each model ─────────────────────────────────


def evaluate(name, model, X_tr, y_tr, X_te, y_te):
    print(f"Training {name}...")
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)

    acc = accuracy_score(y_te, preds)
    prec = precision_score(y_te, preds)
    rec = recall_score(y_te, preds)
    f1 = f1_score(y_te, preds)
    cm = confusion_matrix(y_te, preds)

    print(f"\n{'='*40}")
    print(f"  {name} Results")
    print(f"{'='*40}")
    print(f"  Accuracy  : {acc*100:.2f}%")
    print(f"  Precision : {prec*100:.2f}%")
    print(f"  Recall    : {rec*100:.2f}%")
    print(f"  F1 Score  : {f1*100:.2f}%")
    print(f"\n  Confusion Matrix:")
    print(f"  True Negatives  (legit   → legit  ) : {cm[0][0]}")
    print(f"  False Positives (legit   → phishing) : {cm[0][1]}")
    print(f"  False Negatives (phishing→ legit  ) : {cm[1][0]}")
    print(f"  True Positives  (phishing→ phishing) : {cm[1][1]}")
    print()

    return model, acc


models = {}
scores = {}

# Model 1 — Logistic Regression (fast baseline)
lr_model, lr_acc = evaluate(
    "Logistic Regression",
    LogisticRegression(max_iter=1000, C=1.0),
    X_train_vec, y_train, X_test_vec, y_test
)
models["lr"] = lr_model
scores["Logistic Regression"] = lr_acc

# Model 2 — Linear SVM (best classical model per your papers)
svm_model, svm_acc = evaluate(
    "SVM (LinearSVC)",
    LinearSVC(C=1.0, max_iter=2000),
    X_train_vec, y_train, X_test_vec, y_test
)
models["svm"] = svm_model
scores["SVM"] = svm_acc

# Model 3 — Random Forest
rf_model, rf_acc = evaluate(
    "Random Forest",
    RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
    X_train_vec, y_train, X_test_vec, y_test
)
models["rf"] = rf_model
scores["Random Forest"] = rf_acc

# ── Summary ────────────────────────────────────────────────────────

print("\n" + "="*40)
print("  SUMMARY — All Models")
print("="*40)
for name, acc in sorted(scores.items(), key=lambda x: x[1], reverse=True):
    bar = "█" * int(acc * 40)
    print(f"  {name:<22} {acc*100:.2f}%  {bar}")

# ── Save best model and vectorizer ────────────────────────────────

os.makedirs("models", exist_ok=True)

best_name = max(scores, key=scores.get)
best_model = models[{"Logistic Regression": "lr",
                     "SVM": "svm",
                     "Random Forest": "rf"}[best_name]]

print(f"\n✓ Best model: {best_name} ({scores[best_name]*100:.2f}%)")

# Save vectorizer (needed to process new emails later)
with open("models/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

# Save all three models
with open("models/lr_model.pkl",  "wb") as f:
    pickle.dump(lr_model, f)
with open("models/svm_model.pkl", "wb") as f:
    pickle.dump(svm_model, f)
with open("models/rf_model.pkl",  "wb") as f:
    pickle.dump(rf_model, f)

print("✓ All models saved to models/ folder")
print("✓ Vectorizer saved to models/vectorizer.pkl")
print("\nPhase 2 complete. Ready for Phase 3 — Deep Learning.")
