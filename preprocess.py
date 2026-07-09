from nltk.corpus import stopwords
import pandas as pd
import re
import nltk
import os

# Download required NLTK data (only runs once)
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)


STOP_WORDS = set(stopwords.words("english"))

# ── Text cleaning ──────────────────────────────────────────────────


def clean_text(text):
    """
    Takes raw email text and returns clean lowercase words only.
    Removes URLs, special characters, numbers, extra spaces.
    """
    if pd.isnull(text):
        return ""

    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " url ", text)

    # Remove email addresses
    text = re.sub(r"\S+@\S+", " email ", text)

    # Remove special characters and numbers
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Remove stopwords
    words = text.split()
    words = [w for w in words if w not in STOP_WORDS and len(w) > 2]

    return " ".join(words)


def combine_subject_body(row):
    """
    Combines subject and body into one text field.
    Subject gets repeated 3 times because it carries
    strong signal (urgency words, suspicious phrases).
    """
    subject = str(row["subject"]) if not pd.isnull(row["subject"]) else ""
    body = str(row["body"]) if not pd.isnull(row["body"]) else ""

    # Repeat subject 3x to give it more weight
    combined = f"{subject} {subject} {subject} {body}"
    return combined


# ── Main preprocessing pipeline ───────────────────────────────────

def preprocess(input_path="data/merged_emails.csv",
               output_path="data/processed_emails.csv"):

    print("Loading merged dataset...")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} emails")

    # Fill missing subjects
    df["subject"] = df["subject"].fillna("no subject")

    # Combine subject + body
    print("Combining subject and body...")
    df["text"] = df.apply(combine_subject_body, axis=1)

    # Clean text
    print("Cleaning text (this takes ~30 seconds)...")
    df["clean_text"] = df["text"].apply(clean_text)

    # Keep only what we need
    df_final = df[["clean_text", "label", "source"]].copy()

    # Remove any rows where clean_text is empty
    before = len(df_final)
    df_final = df_final[df_final["clean_text"].str.len() > 10]
    after = len(df_final)
    print(f"Removed {before - after} empty rows")

    # Save
    df_final.to_csv(output_path, index=False)
    print(f"\n✓ Processed dataset saved to {output_path}")
    print(f"✓ Total emails ready for training: {len(df_final)}")

    # Show sample
    print("\n--- Sample cleaned phishing email ---")
    sample = df_final[df_final["label"] == 1].iloc[0]["clean_text"]
    print(sample[:200])

    print("\n--- Sample cleaned legitimate email ---")
    sample = df_final[df_final["label"] == 0].iloc[0]["clean_text"]
    print(sample[:200])

    return df_final


# ── Run ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    preprocess()
