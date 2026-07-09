import pandas as pd
import os

DATA_PATH = "data/email dataset"

# ── Load each dataset ──────────────────────────────────────────────


def load_ceas():
    df = pd.read_csv(f"{DATA_PATH}/CEAS_08.csv", encoding="latin-1")
    df = df[["subject", "body", "label"]].copy()
    df["source"] = "CEAS"
    return df


def load_enron():
    df = pd.read_csv(f"{DATA_PATH}/Enron.csv", encoding="latin-1")
    df = df[["subject", "body", "label"]].copy()
    df["source"] = "Enron"
    return df


def load_nazario():
    df = pd.read_csv(f"{DATA_PATH}/Nazario.csv", encoding="latin-1")
    df = df[["subject", "body", "label"]].copy()
    df["source"] = "Nazario"
    return df

# ── Merge all datasets ─────────────────────────────────────────────


def merge_all():
    print("Loading datasets...")
    ceas = load_ceas()
    enron = load_enron()
    nazario = load_nazario()

    combined = pd.concat([ceas, enron, nazario], ignore_index=True)
    return combined

# ── Explore ────────────────────────────────────────────────────────


def explore(df):
    print("\n========== DATASET OVERVIEW ==========")
    print(f"Total emails     : {len(df)}")
    print(f"Columns          : {df.columns.tolist()}")

    print("\n--- Label distribution ---")
    print(df["label"].value_counts())

    print("\n--- Per source ---")
    print(df.groupby(["source", "label"]).size().reset_index(name="count"))

    print("\n--- Missing values ---")
    print(df.isnull().sum())

    print("\n--- Sample phishing email ---")
    phish = df[df["label"] == 1].iloc[0]
    print(f"Subject : {phish['subject']}")
    print(f"Body    : {str(phish['body'])[:200]}...")

    print("\n--- Sample legitimate email ---")
    legit = df[df["label"] == 0].iloc[0]
    print(f"Subject : {legit['subject']}")
    print(f"Body    : {str(legit['body'])[:200]}...")

# ── Run ────────────────────────────────────────────────────────────


if __name__ == "__main__":
    df = merge_all()
    explore(df)

    # Save merged dataset for next phase
    output_path = "data/merged_emails.csv"
    df.to_csv(output_path, index=False)
    print(f"\n✓ Merged dataset saved to {output_path}")
    print(f"✓ Total emails ready for training: {len(df)}")
