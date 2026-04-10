"""
Communication Risk Analyzer - Backend API

This FastAPI backend provides AI-powered phishing detection for emails and URLs.
It uses a hybrid approach combining:
1. Rule-based detection (trusted domains + PhishTank blacklist)
2. Machine Learning models (Logistic Regression for emails, Random Forest for websites)
3. Pattern-based scam detection

The API accepts POST requests to /analyze and returns risk assessments with explanations.
"""

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import re
from pathlib import Path

# Initialize FastAPI application
app = FastAPI(
    title="Communication Risk Analyzer API",
    description="AI-powered phishing detection for emails and URLs",
    version="2.0"
)

# ======================
# CORS CONFIGURATION
# ======================
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://communication-risk-analyzer.vercel.app",
        "https://communication-risk-analyzer-frontend.vercel.app",
        "http://localhost:3000",
        "http://localhost:8000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ======================
# HEALTH CHECK ENDPOINTS
# ======================

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "Communication Risk Analyzer API",
        "version": "2.0",
        "description": "AI-powered phishing detection for emails and URLs",
        "endpoints": {
            "/": "GET - This information",
            "/health": "GET - Simple health check for monitoring",
            "/analyze": "POST - Analyze email or URL for phishing risks"
        },
        "example_request": {
            "method": "POST",
            "url": "/analyze",
            "body": {
                "input": "www.google.com",
                "mode": "url"
            }
        }
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "communication-risk-analyzer",
        "message": "API is operational"
    }


# ======================
# HELPER FUNCTIONS
# ======================

def load_lines(file_path: Path) -> set[str]:
    if not file_path.exists():
        return set()
    with open(file_path, "r", encoding="utf-8") as f:
        return {line.strip().lower() for line in f if line.strip()}


def normalize_domain(url: str) -> str:
    if not url.startswith(('http://', 'https://')):
        url = 'http://' + url
    
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    domain = domain.replace("www.", "")
    domain = domain.split(':')[0]
    
    return domain


def normalize_url(url: str) -> str:
    if not url.startswith(('http://', 'https://')):
        url = 'http://' + url
    
    return url.strip().lower().rstrip('/')


def normalize_risk(prob: float) -> str:
    """
    Convert probability score to risk category.
    BALANCED THRESHOLDS - Only clear phishing is dangerous.
    """
    # Clear phishing (85%+ confidence) - account locked, lottery wins, etc.
    if prob >= 0.85:
        return "dangerous"
    # Suspicious (50-85%) - unknown numbers, generic alerts, pressure tactics
    elif prob >= 0.50:
        return "suspicious"
    # Safe (below 50%) - normal conversations, educational content
    return "safe"


def normalize_action(risk: str) -> str:
    if risk == "dangerous":
        return "report"
    elif risk == "suspicious":
        return "verify"
    return "ignore"


def clean_email_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"www\.\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def detect_input_type(user_input: str, mode: str) -> str:
    if mode in {"email", "url"}:
        return mode

    text = user_input.strip().lower()
    if text.startswith("http://") or text.startswith("https://") or ("." in text and " " not in text):
        return "url"
    return "email"


def has_ip_address(url: str) -> int:
    if not url.startswith(('http://', 'https://')):
        url = 'http://' + url
    
    parsed = urlparse(url)
    domain = parsed.netloc
    ip_pattern = r"^(?:\d{1,3}\.){3}\d{1,3}$"
    return int(bool(re.match(ip_pattern, domain)))


def has_suspicious_keyword(url: str) -> int:
    url_lower = url.lower()
    return int(any(word in url_lower for word in suspicious_keywords))


def extract_url_features(url: str) -> dict:
    original_url = url
    if not url.startswith(('http://', 'https://')):
        url = 'http://' + url
    
    parsed = urlparse(url)
    domain = parsed.netloc.split(':')[0]

    return {
        "url_length": len(original_url),
        "num_dots": original_url.count("."),
        "has_at_symbol": int("@" in original_url),
        "has_https": int(original_url.startswith("https")),
        "num_dash": original_url.count("-"),
        "num_slash": original_url.count("/"),
        "num_digits": sum(ch.isdigit() for ch in original_url),
        "has_ip_address": has_ip_address(original_url),
        "has_suspicious_keyword": has_suspicious_keyword(original_url),
        "domain_length": len(domain),
    }


def extract_html_features(html: str) -> dict:
    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception:
        soup = BeautifulSoup("", "html.parser")

    html_lower = html.lower() if isinstance(html, str) else ""

    return {
        "html_length": len(html) if isinstance(html, str) else 0,
        "num_forms": len(soup.find_all("form")),
        "num_links": len(soup.find_all("a")),
        "num_scripts": len(soup.find_all("script")),
        "num_iframes": len(soup.find_all("iframe")),
        "num_inputs": len(soup.find_all("input")),
        "num_buttons": len(soup.find_all("button")),
        "has_password_input": int(bool(soup.find("input", {"type": "password"}))),
        "has_login_keyword": int(
            any(word in html_lower for word in ["login", "signin", "verify", "password", "account"])
        ),
    }


def fetch_html_from_url(url: str) -> str:
    if not url.startswith(('http://', 'https://')):
        url = 'http://' + url
    
    try:
        response = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        return response.text
    except Exception:
        return ""


def is_clearly_legitimate(text: str) -> tuple[bool, str]:
    """
    Check if text is clearly legitimate (not phishing).
    Returns (is_legitimate, reason)
    """
    text_lower = text.lower()
    
    # Educational/Professional content
    education_patterns = [
        "training", "academy", "course", "learning", "education",
        "students", "participants", "knowledge", "skills", "development",
        "focus on providing", "quality", "capabilities", "enrolling",
        "right course level", "exact knowledge"
    ]
    
    for pattern in education_patterns:
        if pattern in text_lower:
            return (True, f"Educational/Professional content detected: '{pattern}'")
    
    # Transactional patterns
    transactional_patterns = [
        "order #", "order number", "tracking", "shipped", "delivered",
        "receipt", "purchase", "appointment", "reminder", "meeting",
        "lunch", "dinner", "coffee", "doctor", "dentist",
        "thanks for your", "thank you for your"
    ]
    
    for pattern in transactional_patterns:
        if pattern in text_lower:
            return (True, f"Transactional message detected: '{pattern}'")
    
    return (False, "")


def has_dangerous_keywords(text: str) -> tuple[bool, list[str]]:
    """
    Check for EXTREMELY dangerous keywords that indicate confirmed scams.
    """
    text_lower = text.lower()
    
    dangerous_patterns = [
        (["won", "$", "bank details"], "Prize scam requesting bank information"),
        (["send", "bank details", "money"], "Direct request for bank details"),
        (["account suspended", "click here", "verify"], "Account suspension phishing link"),
        (["free", "iphone", "click"], "Free product phishing scam"),
        (["lottery", "winner", "claim", "fee"], "Lottery fee scam"),
        (["social security", "verify"], "Identity theft attempt"),
        (["western union", "money gram"], "Wire transfer scam"),
        (["irs", "tax", "refund"], "Tax refund scam"),
        (["paypal", "limited", "access"], "PayPal account scam"),
    ]
    
    found_patterns = []
    for keywords, description in dangerous_patterns:
        if all(keyword in text_lower for keyword in keywords):
            found_patterns.append(description)
    
    return len(found_patterns) > 0, found_patterns


def has_suspicious_patterns(text: str) -> tuple[bool, list[str]]:
    """
    Check for suspicious but not necessarily dangerous patterns.
    """
    text_lower = text.lower()
    
    suspicious_patterns = []
    
    # Unknown sender/caller patterns
    if "unknown number" in text_lower or "unknown caller" in text_lower:
        suspicious_patterns.append("Unknown phone number - potential spam")
    
    # Generic account mentions without urgency/phishing links
    if "about your account" in text_lower and not any(word in text_lower for word in ["click", "verify", "suspended", "locked"]):
        suspicious_patterns.append("Generic account reference without specific action")
    
    # Phone numbers in suspicious contexts
    phone_pattern = r'call \d{3}[-.]?\d{3}[-.]?\d{4}'
    if re.search(phone_pattern, text_lower) and any(word in text_lower for word in ["account", "verify", "confirm"]):
        suspicious_patterns.append("Phone number provided for account verification")
    
    # Shortened URLs
    if "bit.ly" in text_lower or "tinyurl" in text_lower:
        suspicious_patterns.append("Contains shortened URL (often used to hide malicious links)")
    
    # Urgency without clear action
    urgency_words = ["urgent", "immediately", "asap", "action required"]
    if any(word in text_lower for word in urgency_words):
        suspicious_patterns.append("Urgent language used - potential pressure tactic")
    
    return len(suspicious_patterns) > 0, suspicious_patterns


def explain_email(text: str) -> list[str]:
    found = [word for word in email_suspicious_words if word in text.lower()]
    explanations = []

    if found:
        explanations.append(f"Suspicious wording detected: {', '.join(found[:4])}")

    if "http" in text or "www." in text:
        explanations.append("Contains link or URL inside message")

    if not explanations:
        explanations.append("Message appears normal based on text analysis")

    return explanations


def explain_website(url: str, html_features: dict, url_features: dict) -> list[str]:
    explanations = []
    
    if not url_features["has_https"]:
        explanations.append("CRITICAL: Website does not use HTTPS (insecure connection)")
    elif url_features["has_https"]:
        explanations.append("Secure connection (HTTPS detected)")
    
    if url_features["has_at_symbol"]:
        explanations.append("URL contains '@' symbol (common phishing technique)")
    
    if url_features["has_ip_address"]:
        explanations.append("URL uses IP address instead of domain name")
    
    if url_features["has_suspicious_keyword"]:
        explanations.append("URL contains suspicious keywords (login, verify, secure, account)")
    
    if html_features["has_password_input"]:
        explanations.append("Page contains password input field (potential credential harvesting)")
    
    if html_features["has_login_keyword"]:
        explanations.append("Page contains login-related wording")
    
    if html_features["num_iframes"] > 0:
        explanations.append(f"Page contains {html_features['num_iframes']} iframe(s)")
    
    if not explanations:
        explanations.append("No obvious phishing indicators detected")
    
    return explanations


# ======================
# MODEL AND DATA LOADING
# ======================
base_dir = Path(__file__).resolve().parent

trusted_domains = load_lines(base_dir / "trusted_domains.txt")
phishtank_urls = load_lines(base_dir / "phishtank_urls.txt")

email_model = joblib.load(base_dir / "email_model.pkl")
email_vectorizer = joblib.load(base_dir / "email_vectorizer.pkl")
website_model = joblib.load(base_dir / "website_model.pkl")

website_feature_cols = [
    "url_length", "num_dots", "has_at_symbol", "has_https", "num_dash",
    "num_slash", "num_digits", "has_ip_address", "has_suspicious_keyword",
    "domain_length", "html_length", "num_forms", "num_links", "num_scripts",
    "num_iframes", "num_inputs", "num_buttons", "has_password_input", "has_login_keyword",
]

suspicious_keywords = [
    "login", "verify", "secure", "account", "update",
    "bank", "confirm", "signin", "password", "paypal"
]

email_suspicious_words = [
    "urgent", "verify", "account", "suspended",
    "click", "login", "password", "confirm", "bank"
]


def is_trusted_domain(url: str) -> bool:
    domain = normalize_domain(url)
    return domain in trusted_domains


def is_known_phishtank_url(url: str) -> bool:
    return normalize_url(url) in phishtank_urls


class RequestData(BaseModel):
    input: str
    mode: str = "auto"


# ======================
# EMAIL ANALYSIS FUNCTION
# ======================
def analyze_email(user_input: str):
    # FIRST: Check if clearly legitimate (educational, transactional, etc.)
    is_legit, legit_reason = is_clearly_legitimate(user_input)
    if is_legit:
        return {
            "risk": "safe",
            "type": "legitimate",
            "confidence": 0.90,
            "model_used": "Legitimate Content Detection",
            "input_type": "email",
            "explanation": [f"✓ {legit_reason} - This appears to be legitimate content"],
            "action": "ignore",
        }
    
    # SECOND: Check for dangerous scam patterns
    is_dangerous, dangerous_patterns = has_dangerous_keywords(user_input)
    
    if is_dangerous:
        explanations = [
            "🚨 CONFIRMED SCAM PATTERNS DETECTED:",
            *[f"  • {pattern}" for pattern in dangerous_patterns[:5]],
            "⚠️ This is a known fraud attempt - DO NOT RESPOND"
        ]
        return {
            "risk": "dangerous",
            "type": "phishing",
            "confidence": 0.95,
            "model_used": "Scam Pattern Detection",
            "input_type": "email",
            "explanation": explanations,
            "action": "report",
        }
    
    # THIRD: ML Model
    try:
        clean_text = clean_email_text(user_input)
        X = email_vectorizer.transform([clean_text])
        
        pred_raw = email_model.predict(X)[0]
        
        if isinstance(pred_raw, str):
            if pred_raw.lower() in ['phishing email', 'phishing', 'spam', '1', 'true']:
                pred = 1
            else:
                pred = 0
        else:
            pred = int(pred_raw)
        
        if hasattr(email_model, "predict_proba"):
            probs = email_model.predict_proba(X)[0]
            prob = float(probs[1])  # phishing probability
            prob = max(0.0, min(1.0, prob))
        else:
            prob = 1.0 if pred == 1 else 0.0

        risk = normalize_risk(prob)
        
    except Exception as e:
        print(f"Email ML error: {e}")
        suspicious_words = ["urgent", "verify", "account", "suspended", "click", "login", "password"]
        found_words = [w for w in suspicious_words if w in user_input.lower()]
        
        if found_words:
            risk = "suspicious"
            prob = 0.6
            explanation = [f"Contains suspicious words: {', '.join(found_words)}"]
        else:
            risk = "safe"
            prob = 0.7
            explanation = ["No suspicious patterns detected"]
        
        return {
            "risk": risk,
            "type": "legitimate" if risk == "safe" else "unknown",
            "confidence": prob,
            "model_used": "Fallback Rule-based",
            "input_type": "email",
            "explanation": explanation,
            "action": normalize_action(risk),
        }
    
    # Check for suspicious patterns
    is_suspicious, suspicious_patterns = has_suspicious_patterns(user_input)
    
    if is_suspicious and risk == "safe":
        risk = "suspicious"
        prob = max(prob, 0.55)
    
    explanation = explain_email(user_input)
    
    if is_suspicious and suspicious_patterns:
        explanation = ["⚠️ " + pattern for pattern in suspicious_patterns] + explanation
    
    # Set type based on risk
    if risk == "dangerous":
        email_type = "phishing"
    elif risk == "suspicious":
        email_type = "suspicious"
    else:
        email_type = "legitimate"
    
    return {
        "risk": risk,
        "type": email_type,
        "confidence": prob,
        "model_used": "Logistic Regression",
        "input_type": "email",
        "explanation": explanation,
        "action": normalize_action(risk),
    }


# ======================
# WEBSITE ANALYSIS FUNCTION
# ======================
def analyze_website(user_input: str):
    # RULE 1: trusted domain
    if is_trusted_domain(user_input):
        return {
            "risk": "safe",
            "type": "legitimate",
            "confidence": 0.99,
            "model_used": "Trusted Domain Rule",
            "input_type": "url",
            "explanation": ["✓ Domain matched trusted top-sites list", "✓ This website is verified as legitimate"],
            "action": "ignore",
        }

    # RULE 2: known phishing URL
    if is_known_phishtank_url(user_input):
        return {
            "risk": "dangerous",
            "type": "phishing",
            "confidence": 0.99,
            "model_used": "PhishTank Rule",
            "input_type": "url",
            "explanation": ["🚨 URL matched known verified phishing database", "🚨 This is a confirmed malicious site"],
            "action": "report",
        }

    html = fetch_html_from_url(user_input)

    url_features = extract_url_features(user_input)
    html_features = extract_html_features(html)

    row = {**url_features, **html_features}
    X_df = pd.DataFrame([row])[website_feature_cols]

    pred = int(website_model.predict(X_df)[0])

    if hasattr(website_model, "predict_proba"):
        prob = float(website_model.predict_proba(X_df)[0][1])
    else:
        prob = 1.0 if pred == 1 else 0.0

    risk = normalize_risk(prob)
    return {
        "risk": risk,
        "type": "phishing" if pred == 1 else "legitimate",
        "confidence": prob,
        "model_used": "Random Forest",
        "input_type": "url",
        "explanation": explain_website(user_input, html_features, url_features),
        "action": normalize_action(risk),
    }


# ======================
# MAIN API ENDPOINT
# ======================
@app.post("/analyze")
def analyze(data: RequestData):
    input_type = detect_input_type(data.input, data.mode)

    if input_type == "email":
        return analyze_email(data.input)

    return analyze_website(data.input)
