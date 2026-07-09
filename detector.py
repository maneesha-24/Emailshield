from bert_detector import load_bert, predict_bert
from nltk.corpus import stopwords
import pickle
import re
import nltk
from email import message_from_string

nltk.download("stopwords", quiet=True)

BERT_AVAILABLE = load_bert()

STOP_WORDS = set(stopwords.words("english"))

# ── Load saved model and vectorizer ───────────────────────────────

with open("models/svm_model.pkl", "rb") as f:
    MODEL = pickle.load(f)

with open("models/vectorizer.pkl", "rb") as f:
    VECTORIZER = pickle.load(f)

# ── Word lists ─────────────────────────────────────────────────────

PHISHING_WORDS = [
    "verify your account", "confirm your identity",
    "unusual activity", "suspicious login", "account suspended",
    "account locked", "click here to verify", "update your information",
    "your account will be", "immediate action required",
    "validate your", "unauthorized access", "security alert",
    "your password", "enter your credentials", "billing information",
    "payment details", "bank account", "credit card number",
    "social security", "login attempt"
]

SCAM_WORDS = [
    "you have won", "winner", "lottery", "prize", "claim your",
    "million dollars", "inheritance", "nigerian prince",
    "wire transfer", "western union", "money gram",
    "work from home", "make money fast", "earn dollars",
    "investment opportunity", "double your money",
    "secret shopper", "gift card", "itunes card"
]

SPAM_WORDS = [
    "unsubscribe", "click here", "buy now", "limited offer",
    "sale", "discount", "deal", "offer expires", "free shipping",
    "shop now", "order now", "best price", "lowest price",
    "payday", "wellness", "newsletter", "promotional",
    "marketing", "advertisement"
]

URGENCY_WORDS = [
    "urgent", "immediately", "act now", "expires",
    "last chance", "final notice", "warning", "alert",
    "suspended", "blocked", "locked", "risk"
]

TRUSTED_BRANDS = [
    "paypal", "amazon", "google", "microsoft", "apple",
    "netflix", "facebook", "instagram", "twitter", "linkedin",
    "bank", "chase", "wells fargo", "citibank", "barclays",
    "hsbc", "hdfc", "icici", "sbi", "upi"
]

# ── Text cleaning ──────────────────────────────────────────────────


def clean_text(text):
    if not text:
        return ""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " url ", text)
    text = re.sub(r"\S+@\S+", " email ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    words = [w for w in words if w not in STOP_WORDS and len(w) > 2]
    return " ".join(words)


# ── Feature extraction ─────────────────────────────────────────────

def extract_features(raw_email):
    features = {
        "subject": "",
        "body": "",
        "sender": "",
        "sender_domain": "",
        "has_urls": False,
        "url_count": 0,
        "urls": [],
        "urgency_words": [],
        "phishing_words": [],
        "scam_words": [],
        "spam_words": [],
        "has_unsubscribe": False,
        "is_html_heavy": False,
        "raw_headers": {}
    }

    try:
        msg = message_from_string(raw_email)
        features["subject"] = msg.get("Subject", "") or ""
        features["sender"] = msg.get("From", "") or ""

        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                ct = part.get_content_type()
                if ct == "text/plain":
                    payload = part.get_payload(decode=True)
                    if payload:
                        body += payload.decode(errors="ignore")
        else:
            payload = msg.get_payload(decode=True)
            if payload:
                body = payload.decode(errors="ignore")
            else:
                body = str(msg.get_payload())

        features["body"] = body

        for key, val in msg.items():
            features["raw_headers"][key] = val

    except Exception:
        features["body"] = raw_email

    # Sender domain
    sender = features["sender"]
    domain_match = re.search(r"@([\w\.\-]+)", sender)
    if domain_match:
        features["sender_domain"] = domain_match.group(1).lower()

    full_text = (features["subject"] + " " +
                 features["body"]).lower()

    # URLs
    urls = re.findall(
        r"http[s]?://\S+|www\.\S+", full_text)
    features["urls"] = urls
    features["url_count"] = len(urls)
    features["has_urls"] = len(urls) > 0

    # Unsubscribe link — strong spam indicator
    features["has_unsubscribe"] = "unsubscribe" in full_text

    # HTML heavy — lots of HTML with little text
    html_tags = len(re.findall(r"<[^>]+>", raw_email))
    text_length = len(re.sub(r"<[^>]+>", "", raw_email))
    features["is_html_heavy"] = (
        html_tags > 20 and text_length < 500
    )

    # Word matching
    features["urgency_words"] = [
        w for w in URGENCY_WORDS if w in full_text]
    features["phishing_words"] = [
        w for w in PHISHING_WORDS if w in full_text]
    features["scam_words"] = [
        w for w in SCAM_WORDS if w in full_text]
    features["spam_words"] = [
        w for w in SPAM_WORDS if w in full_text]

    return features


# ── ML prediction ──────────────────────────────────────────────────

def detect_phishing(features):
    subject = features["subject"]
    body = features["body"]
    combined = f"{subject} {subject} {subject} {body}"
    cleaned = clean_text(combined)

    import numpy as np

    vec = VECTORIZER.transform([cleaned])
    svm_decision = MODEL.decision_function(vec)[0]
    svm_prob = 1 / (1 + np.exp(-svm_decision))

    bert_prob = predict_bert(cleaned) if BERT_AVAILABLE else None

    if bert_prob is not None:
        final_prob = (bert_prob * 0.6) + (svm_prob * 0.4)
        method_used = "BERT + SVM Ensemble"
    else:
        final_prob = svm_prob
        method_used = "SVM"

    is_phishing = final_prob >= 0.5
    confidence = min(99, max(1,
                             int(final_prob * 100) if is_phishing
                             else int((1 - final_prob) * 100)
                             ))

    return {
        "is_phishing": is_phishing,
        "raw_prob": final_prob,
        "confidence": confidence,
        "prediction": "PHISHING" if is_phishing else "LEGITIMATE",
        "method": method_used
    }


# ── Spoofing detection ─────────────────────────────────────────────

def detect_spoofing(features):
    results = {
        "spf_result": "not found",
        "dkim_result": "not found",
        "dmarc_result": "not found",
        "display_spoof": False,
        "is_spoofed": False,
        "spoof_signals": []
    }

    headers = features["raw_headers"]

    for key, val in headers.items():
        kl = key.lower()
        vl = val.lower()

        if "received-spf" in kl:
            if "pass" in vl:
                results["spf_result"] = "pass"
            elif "fail" in vl:
                results["spf_result"] = "fail"
                results["spoof_signals"].append("SPF authentication failed")
            elif "softfail" in vl:
                results["spf_result"] = "softfail"
                results["spoof_signals"].append(
                    "SPF softfail — sender may be forged")

        if "authentication-results" in kl:
            if "dkim=pass" in vl:
                results["dkim_result"] = "pass"
            elif "dkim=fail" in vl:
                results["dkim_result"] = "fail"
                results["spoof_signals"].append(
                    "DKIM signature verification failed")

            if "dmarc=pass" in vl:
                results["dmarc_result"] = "pass"
            elif "dmarc=fail" in vl:
                results["dmarc_result"] = "fail"
                results["spoof_signals"].append("DMARC policy check failed")

    # Display name spoofing
    sender = features["sender"]
    if sender:
        name_match = re.search(r'"?([^"<]+)"?\s*<', sender)
        domain_match = re.search(r"@([\w\.\-]+)>?$", sender)

        if name_match and domain_match:
            display_name = name_match.group(1).lower().strip()
            domain = domain_match.group(1).lower()

            for brand in TRUSTED_BRANDS:
                if brand in display_name and brand not in domain:
                    results["display_spoof"] = True
                    results["spoof_signals"].append(
                        f"Display name claims to be "
                        f"'{display_name.title()}' but "
                        f"email comes from '{domain}'"
                    )

    results["is_spoofed"] = len(results["spoof_signals"]) > 0
    return results


# ── Smart verdict classifier ───────────────────────────────────────

def classify_verdict(features, phishing_result, spoofing_result):
    raw_prob = phishing_result["raw_prob"]
    is_spoofed = spoofing_result["is_spoofed"]

    spf = spoofing_result["spf_result"]
    dkim = spoofing_result["dkim_result"]
    dmarc = spoofing_result["dmarc_result"]

    all_auth_pass = (spf == "pass" and
                     dkim == "pass" and
                     dmarc == "pass")

    has_phishing = len(features["phishing_words"]) > 0
    has_scam = len(features["scam_words"]) > 0
    has_spam = (features["has_unsubscribe"] or
                len(features["spam_words"]) >= 2)
    has_urgency = len(features["urgency_words"]) > 0

    # ── Spoofing takes priority ────────────────────────────────
    if is_spoofed and raw_prob >= 0.5:
        return "PHISHING + SPOOFED", "red"

    if is_spoofed and raw_prob < 0.5:
        return "SPOOFED", "orange"

    # ── Authenticated commercial email ─────────────────────────
    if all_auth_pass and has_spam and not has_phishing and not has_scam:
        return "SPAM / NEWSLETTER", "orange"

    # ── Scam detection ─────────────────────────────────────────
    if has_scam and raw_prob >= 0.5:
        return "SCAM", "red"

    # ── Definite phishing — needs BOTH high score + attack words
    if raw_prob >= 0.75 and has_phishing:
        return "PHISHING", "red"

    # ── Very high score alone ──────────────────────────────────
    if raw_prob >= 0.88:
        return "PHISHING", "red"

    # ── No attack signals at all → legitimate ─────────────────
    # Even if ML score is medium, if there are zero phishing
    # words, zero scam words, zero urgency — treat as safe
    if not has_phishing and not has_scam and not has_urgency:
        if raw_prob < 0.80:
            return "LEGITIMATE", "green"

    # ── Suspicious middle ground ───────────────────────────────
    if raw_prob >= 0.60 and (has_urgency or has_phishing):
        return "SUSPICIOUS", "orange"

    # ── Auth passed ────────────────────────────────────────────
    if all_auth_pass and raw_prob < 0.65:
        return "LEGITIMATE", "green"

    # ── Default legitimate ─────────────────────────────────────
    if raw_prob < 0.60:
        return "LEGITIMATE", "green"

    return "SUSPICIOUS", "orange"


# ── Natural language explainer ─────────────────────────────────────

def explain_result(features, phishing_result,
                   spoofing_result, verdict):
    """
    Generates a detailed, natural language explanation.
    """
    raw_prob = phishing_result["raw_prob"]
    conf = phishing_result["confidence"]
    sender = features["sender"] or "unknown sender"
    subject = features["subject"] or "no subject"
    domain = features["sender_domain"] or "unknown domain"

    spf = spoofing_result["spf_result"]
    dkim = spoofing_result["dkim_result"]
    dmarc = spoofing_result["dmarc_result"]

    all_auth_pass = (spf == "pass" and
                     dkim == "pass" and
                     dmarc == "pass")
    no_auth = (spf == "not found" and
               dkim == "not found" and
               dmarc == "not found")

    paragraphs = []

    # ── Opening paragraph based on verdict ────────────────────
    if verdict == "PHISHING + SPOOFED":
        paragraphs.append(
            f"This email is highly dangerous and shows clear signs "
            f"of both phishing and sender identity forgery. "
            f"Our AI models analysed the content and flagged it "
            f"with {conf}% confidence as a malicious email "
            f"designed to steal your information. "
            f"You should delete this email immediately and avoid "
            f"clicking any links or downloading any attachments."
        )

    elif verdict == "PHISHING":
        paragraphs.append(
            f"This email has been identified as a phishing attempt "
            f"with {conf}% confidence. "
            f"Phishing emails are designed to trick you into "
            f"revealing sensitive information such as passwords, "
            f"bank details, or personal data. "
            f"Do not click any links in this email or provide "
            f"any information requested."
        )

    elif verdict == "SCAM":
        paragraphs.append(
            f"This email appears to be a scam. "
            f"Our analysis detected language patterns commonly "
            f"associated with fraudulent schemes such as fake "
            f"prize winnings, lottery notifications, or "
            f"financial fraud attempts. "
            f"These emails are designed to manipulate you into "
            f"sending money or sharing personal details. "
            f"Do not respond or engage with the sender."
        )

    elif verdict == "SPOOFED":
        paragraphs.append(
            f"The content of this email appears relatively benign, "
            f"but the sender's identity shows signs of forgery. "
            f"Someone may be impersonating a trusted organisation "
            f"or individual. Treat this email with caution and "
            f"verify the sender through official channels before "
            f"taking any action."
        )

    elif verdict == "SPAM / NEWSLETTER":
        paragraphs.append(
            f"This email is a commercial marketing or newsletter "
            f"email. It is not a phishing or spoofing attack — "
            f"it was sent by a verified sender and passed all "
            f"email authentication checks. "
            f"However, it is promotional in nature and was "
            f"likely sent to a large number of recipients. "
            f"You can safely unsubscribe if you no longer wish "
            f"to receive it."
        )

    elif verdict == "SUSPICIOUS":
        paragraphs.append(
            f"This email could not be conclusively classified "
            f"as safe or dangerous. "
            f"Our AI models detected some patterns that are "
            f"mildly associated with suspicious activity, "
            f"but the signals are not strong enough to confirm "
            f"a phishing or scam attempt. "
            f"Exercise caution — do not share personal "
            f"information and verify the sender if unsure."
        )

    else:  # LEGITIMATE
        paragraphs.append(
            f"This email appears to be legitimate and safe. "
            f"Our AI models found no significant indicators of "
            f"phishing, scam, or spoofing activity. "
            f"The content and writing style are consistent with "
            f"genuine communication."
        )

    # ── Sender analysis paragraph ──────────────────────────────
    sender_para = f"The email was sent from {sender}. "

    if spoofing_result["spoof_signals"]:
        sender_para += (
            f"The sender identity raises concerns: "
            + " ".join(spoofing_result["spoof_signals"]) + ". "
        )
    elif all_auth_pass:
        sender_para += (
            f"The sending domain '{domain}' passed all three "
            f"email authentication checks — SPF, DKIM, and "
            f"DMARC — which means the email genuinely originated "
            f"from this domain and was not tampered with in transit."
        )
    elif no_auth:
        sender_para += (
            f"No email authentication headers were found. "
            f"This may mean the email headers were not included "
            f"in the pasted content, or the sender's domain does "
            f"not have authentication configured."
        )
    else:
        failed = []
        if spf == "fail":
            failed.append("SPF")
        if dkim == "fail":
            failed.append("DKIM")
        if dmarc == "fail":
            failed.append("DMARC")
        if failed:
            sender_para += (
                f"Authentication checks partially failed: "
                f"{', '.join(failed)} did not pass, which may "
                f"indicate the sender's address was forged."
            )

    paragraphs.append(sender_para)

    # ── Content signals paragraph ──────────────────────────────
    signals = []

    if features["phishing_words"]:
        signals.append(
            f"phishing-related phrases such as "
            f"'{features['phishing_words'][0]}'"
            + (f" and '{features['phishing_words'][1]}'"
               if len(features["phishing_words"]) > 1 else "")
        )

    if features["scam_words"]:
        signals.append(
            f"scam-associated language including "
            f"'{features['scam_words'][0]}'"
        )

    if features["urgency_words"]:
        signals.append(
            f"urgency triggers like "
            f"'{', '.join(features['urgency_words'][:3])}'"
        )

    if features["url_count"] > 0:
        signals.append(
            f"{features['url_count']} embedded URL"
            f"{'s' if features['url_count'] > 1 else ''}"
        )

    if features["has_unsubscribe"]:
        signals.append(
            "an unsubscribe link typical of mass marketing emails"
        )

    if signals:
        content_para = (
            f"In terms of content, the email contains "
            + ", ".join(signals[:-1])
            + (f", and {signals[-1]}" if len(signals) > 1
               else signals[0])
            + ". "
        )

        if verdict in ["LEGITIMATE", "SPAM / NEWSLETTER"]:
            content_para += (
                "While some of these signals can appear in "
                "suspicious emails, in this case they are "
                "consistent with normal commercial communication "
                "from an authenticated sender."
            )
        elif verdict in ["PHISHING", "SCAM", "PHISHING + SPOOFED"]:
            content_para += (
                "These signals, combined with the AI model "
                "analysis, strongly suggest malicious intent."
            )
        else:
            content_para += (
                "These signals warrant caution even if the "
                "email cannot be definitively classified as "
                "an attack."
            )

        paragraphs.append(content_para)

    # ── Recommendation paragraph ───────────────────────────────
    if verdict in ["PHISHING", "PHISHING + SPOOFED", "SCAM"]:
        paragraphs.append(
            "Recommended action: Do not click any links, "
            "download attachments, or reply to this email. "
            "If this email claims to be from a service you use, "
            "contact that organisation directly through their "
            "official website. Report this email as phishing "
            "to your email provider."
        )
    elif verdict == "SPOOFED":
        paragraphs.append(
            "Recommended action: Verify the sender's identity "
            "by contacting them through a known, trusted channel "
            "before taking any action requested in this email."
        )
    elif verdict == "SUSPICIOUS":
        paragraphs.append(
            "Recommended action: Do not share any personal or "
            "financial information. If the email requests action "
            "on an account, log in directly through the official "
            "website rather than clicking any links provided."
        )
    elif verdict == "SPAM / NEWSLETTER":
        paragraphs.append(
            "Recommended action: If you did not sign up for "
            "this mailing list, use the unsubscribe link at "
            "the bottom of the email or mark it as spam in "
            "your email client."
        )
    else:
        paragraphs.append(
            "Recommended action: No immediate action required. "
            "As always, remain cautious with any unexpected "
            "requests for personal information, even from "
            "apparently legitimate senders."
        )

    return " \n\n".join(paragraphs)


# ── Master analysis function ───────────────────────────────────────

def analyze_email(raw_email):
    features = extract_features(raw_email)
    phishing_result = detect_phishing(features)
    spoofing_result = detect_spoofing(features)

    final_verdict, verdict_color = classify_verdict(
        features, phishing_result, spoofing_result
    )

    explanation = explain_result(
        features, phishing_result, spoofing_result, final_verdict
    )

    return {
        "verdict": final_verdict,
        "verdict_color": verdict_color,
        "confidence": phishing_result["confidence"],
        "explanation": explanation,
        "subject": features["subject"],
        "sender": features["sender"],
        "url_count": features["url_count"],
        "urgent_words": (features["urgency_words"] +
                         features["phishing_words"] +
                         features["scam_words"])[:6],
        "spf": spoofing_result["spf_result"],
        "dkim": spoofing_result["dkim_result"],
        "dmarc": spoofing_result["dmarc_result"],
        "spoof_signals": spoofing_result["spoof_signals"],
        "method": phishing_result["method"]
    }
