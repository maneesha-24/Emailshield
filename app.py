from flask import Flask, render_template, request, jsonify
from detector import analyze_email

app = Flask(__name__)

SAMPLE_EMAILS = {
    "bank_phishing": {
        "label": "Bank Aadhar Request",
        "content": """From: Security Team <alerts@sbi-secure-update.com>
Subject: Urgent: Upload Your Aadhaar Details to Avoid Account Suspension

Dear Customer,

We have detected that your KYC verification is incomplete.
As per RBI guidelines, your account will be suspended within 48 hours
unless you upload your Aadhaar card details immediately.

Please click the link below to complete your KYC:
http://sbi-kyc-update.fake-portal.com/verify

You will need to provide:
- Aadhaar Card Number
- PAN Card Number  
- Date of Birth
- Net Banking Password

Failure to comply will result in permanent account suspension.

SBI Security Team"""
    },
    "food_invoice": {
        "label": "Food Delivery Invoice",
        "content": """From: Swiggy Orders <orders@swiggy.com>
Subject: Your order invoice #SWG-2024-98231

Hi there,

Thank you for your order! Here is your invoice summary.

Order #SWG-2024-98231
Date: April 4, 2026

Items ordered:
- Butter Chicken x1 - Rs. 320
- Garlic Naan x2 - Rs. 80
- Mango Lassi x1 - Rs. 90

Subtotal: Rs. 490
Delivery fee: Rs. 40
Total paid: Rs. 530

Your order was delivered to your saved address.
For any issues, contact support@swiggy.com

Thank you for ordering with Swiggy!
Team Swiggy"""
    },
    "new_year_offer": {
        "label": "New Year Offer",
        "content": """From: Amazon India <deals@amazon.in>
Subject: New Year Sale - Up to 70% off on Electronics!

Hello,

Wishing you a Happy New Year from Amazon India!

We have amazing deals just for you this New Year season.

FEATURED DEALS:
- Samsung 4K TV - 45% off - Rs. 32,999
- Apple AirPods Pro - 30% off - Rs. 17,999
- Boat Rockerz Headphones - 60% off - Rs. 1,299
- Laptop Bags & Accessories - Up to 70% off

Sale ends January 5, 2026. Limited stock available.

Shop now at amazon.in/newyearsale

To unsubscribe from promotional emails, click here.

Best wishes,
Amazon India Team"""
    },
    "lottery_scam": {
        "label": "Lottery Winning",
        "content": """From: International Lottery Commission <winner@global-lottery-claim.net>
Subject: CONGRATULATIONS!!! You Have Won $2,500,000 USD

CONGRATULATIONS LUCKY WINNER!!!

Your email address has been selected as the GRAND PRIZE WINNER
of the International Email Lottery Draw 2026.

YOU HAVE WON: TWO MILLION FIVE HUNDRED THOUSAND US DOLLARS ($2,500,000)

To claim your prize money IMMEDIATELY you must:
1. Send your Full Name, Address, Phone Number
2. Send your Bank Account Number and IFSC Code
3. Pay a processing fee of $150 via Western Union

Contact our claims agent immediately:
Email: agent.james@lottery-claims-international.com
WhatsApp: +1-555-987-6543

WARNING: You must claim within 72 hours or prize will be FORFEITED.
Keep this notification CONFIDENTIAL until claim is processed.

International Lottery Commission
Geneva, Switzerland"""
    },
    "meeting_scheduled": {
        "label": "Meeting Scheduled",
        "content": """From: Priya Sharma <priya.sharma@techcorp.com>
Subject: Team sync scheduled for Monday 10am

Hi everyone,

I have scheduled our weekly team sync for Monday, April 7 at 10:00 AM IST.

Agenda:
1. Sprint review - Q1 deliverables
2. Blockers and updates from each team member
3. Planning for next sprint

Meeting link: meet.google.com/abc-defg-hij
Duration: 45 minutes

Please come prepared with your updates. Let me know if you
have any agenda items to add.

See you Monday!
Priya Sharma
Senior Project Manager
TechCorp India"""
    }
}


@app.route("/")
def home():
    samples = {k: v["label"] for k, v in SAMPLE_EMAILS.items()}
    return render_template("index.html", samples=samples)


@app.route("/sample/<key>")
def get_sample(key):
    if key in SAMPLE_EMAILS:
        return jsonify({"content": SAMPLE_EMAILS[key]["content"]})
    return jsonify({"error": "Sample not found"}), 404


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        raw_email = ""
        if "email_file" in request.files:
            file = request.files["email_file"]
            if file.filename != "":
                raw_email = file.read().decode("utf-8", errors="ignore")
        if not raw_email:
            raw_email = request.form.get("email_text", "")
        if not raw_email.strip():
            return jsonify({"error": "No email content provided"}), 400
        result = analyze_email(raw_email)
        return jsonify(result)

    except Exception:
        app.logger.exception("analyze failed")
        return jsonify({"error": "Something went wrong while analyzing this email."}), 500


if __name__ == "__main__":
    app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024  # 2MB upload limit
    app.run(debug=False)
