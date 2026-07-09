import torch
import os
from transformers import BertTokenizer, BertForSequenceClassification

BERT_MODEL_PATH = "models/bert_phishing_model"
MAX_LEN = 128

# These will be loaded once when the app starts
bert_model = None
bert_tokenizer = None
bert_device = None


def load_bert():
    """
    Loads BERT model from models/bert_phishing_model folder.
    Called once when app starts.
    """
    global bert_model, bert_tokenizer, bert_device

    if not os.path.exists(BERT_MODEL_PATH):
        print("⚠ BERT model not found — using SVM only")
        return False

    print("Loading BERT model...")
    bert_device = torch.device("cpu")
    bert_tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH)
    bert_model = BertForSequenceClassification.from_pretrained(
        BERT_MODEL_PATH)
    bert_model = bert_model.to(bert_device)
    bert_model.eval()
    print("✓ BERT model loaded")
    return True


def predict_bert(text):
    """
    Takes cleaned email text, returns phishing probability 0-1.
    Returns None if BERT not loaded.
    """
    if bert_model is None:
        return None

    encoding = bert_tokenizer(
        text,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(bert_device)
    attention_mask = encoding["attention_mask"].to(bert_device)

    with torch.no_grad():
        outputs = bert_model(input_ids=input_ids,
                             attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1)
        phishing_prob = probs[0][1].item()

    return phishing_prob
