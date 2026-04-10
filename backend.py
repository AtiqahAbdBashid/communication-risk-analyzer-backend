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
# Allows frontend applications to communicate with this API
# ======================
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://communication-risk-analyzer.vercel.app",      # Production frontend
        "https://communication-risk-analyzer-frontend.vercel.app",  # Alternative frontend URL
        "http://localhost:3000",                               # Local development
        "http://localhost:8000"                                # Local backend
    ],
    allow_credentials=True,
    allow_methods=["*"],        # Allow all HTTP methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],        # Allow all headers
)


# ======================
# HEALTH CHECK ENDPOINTS
# These are useful for monitoring and keeping the service alive
# Judges can use these to verify the API is running
# ======================

@app.get("/")
def root():
    """
    Root endpoint - Returns API information and available endpoints.
    Useful for quick verification that the service is running.
    
    Returns:
        dict: API metadata including version and available endpoints
    """
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
    """
    Health check endpoint - Returns simple status for uptime monitoring.
    Can be used by services like UptimeRobot to keep the Railway instance awake.
    
    Returns:
        dict: Health status of the service
    """
    return {
        "status": "healthy",
        "service": "communication-risk-analyzer",
        "message": "API is operational"
    }


# ======================
# HELPER FUNCTIONS
# These utility functions support the main analysis logic
# ======================

def load_lines(file_path: Path) -> set[str]:
    """
    Load a text file into a set of lowercase strings.
    Used for loading trusted domains and phishing URL lists.
    
    Args:
        file_path: Path to the text file
        
    Returns:
        set: Set of stripped, lowercase lines from the file
    """
    if not file_path.exists():
        return set()
    with open(file_path, "r", encoding="utf-8") as f:
        return {line.strip().lower() for line in f if line.strip()}


def normalize_domain(url: str) -> str:
    """
    Extract and normalize domain name from a URL.
    Removes scheme (http://), www. prefix, and port numbers.
    
    Args:
        url: URL string (may or may not include scheme)
        
    Returns:
        str: Normalized domain name (e.g., "google.com")
    """
    # Add scheme if missing for proper parsing
    if not url.startswith(('http://', 'https://')):
        url = 'http://' + url
    
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    
    # Remove www. prefix
    domain = domain.replace("www.", "")
    
    # Remove port number if present
    domain = domain.split(':')[0]
    
    return domain


def normalize_url(url: str) -> str:
    """
    Normalize a URL for consistent comparison.
    Adds scheme if missing, converts to lowercase, removes trailing slashes.
    
    Args:
        url: URL string
        
    Returns:
        str: Normalized URL
    """
    if not url.startswith(('http://', 'https://')):
        url = 'http://' + url
    
    return url.strip().lower().rstrip('/')


def normalize_risk(prob: float) -> str:
    """
    Convert probability score to risk category.
    Thresholds tuned for phishing detection sensitivity.
    
    Args:
        prob: Probability score (0-1) from ML model
        
    Returns:
        str: 'safe', 'suspicious', or 'dangerous'
    """
    # More aggressive thresholds for better phishing detection
    if prob >= 0.55:      # 55%+ confidence = dangerous
        return "dangerous"
    elif prob >= 0.30:    # 30-55% = suspicious
        return "suspicious"
    return "safe"          # Below 30% = safe


def normalize_action(risk: str) -> str:
    """
    Map risk level to recommended user action.
    
    Args:
        risk: Risk category ('safe', 'suspicious', 'dangerous')
        
    Returns:
        str: Recommended action ('ignore', 'verify', 'report')
    """
    if risk == "dangerous":
        return "report"
    elif risk == "suspicious":
        return "verify"
    return "ignore"


def clean_email_text(text: str) -> str:
    """
    Preprocess email text for ML model input.
    Removes URLs, numbers, special characters, and converts to lowercase.
    
    Args:
        text: Raw email text
        
    Returns:
        str: Cleaned text ready for vectorization
    """
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)      # Remove URLs
    text = re.sub(r"www\.\S+", " ", text)     # Remove www links
    text = re.sub(r"\d+", " ", text)          # Remove numbers
    text = re.sub(r"[^a-zA-Z\s]", " ", text)  # Remove special characters
    text = re.sub(r"\s+", " ", text).strip()  # Collapse multiple spaces
    return text


def detect_input_type(user_input: str, mode: str) -> str:
    """
    Determine if input is email or URL based on content and user mode.
    
    Args:
        user_input: Raw input string
        mode: User-selected mode ('auto', 'email', 'url')
        
    Returns:
        str: 'email' or 'url'
    """
    if mode in {"email", "url"}:
        return mode

    text = user_input.strip().lower()
    # Heuristic: URLs typically start with http:// or contain dots without spaces
    if text.startswith("http://") or text.startswith("https://") or ("." in text and " " not in text):
        return "url"
    return "email"


def has_ip_address(url: str) -> int:
    """
    Check if URL uses an IP address instead of a domain name.
    Legitimate sites rarely use IP addresses directly.
    
    Args:
        url: URL string
        
    Returns:
        int: 1 if IP address is used, 0 otherwise
    """
    if not url.startswith(('http://', 'https://')):
        url = 'http://' + url
    
    parsed = urlparse(url)
    domain = parsed.netloc
    ip_pattern = r"^(?:\d{1,3}\.){3}\d{1,3}$"
    return int(bool(re.match(ip_pattern, domain)))


def has_suspicious_keyword(url: str) -> int:
    """
    Check if URL contains suspicious keywords commonly used in phishing.
    
    Args:
        url: URL string
        
    Returns:
        int: 1 if suspicious keywords found, 0 otherwise
    """
    url_lower = url.lower()
    return int(any(word in url_lower for word in suspicious_keywords))


def extract_url_features(url: str) -> dict:
    """
    Extract 10 URL-based features for the Random Forest model.
    Features include length, dot count, HTTPS usage, etc.
    
    Args:
        url: URL string
        
    Returns:
        dict: Feature dictionary for ML model
    """
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
    """
    Extract 9 HTML-based features from webpage content.
    Features include form count, input fields, password presence, etc.
    
    Args:
        html: HTML content of the webpage
        
    Returns:
        dict: Feature dictionary for ML model
    """
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
    """
    Fetch HTML content from a URL for feature extraction.
    Timeout is set to 8 seconds to avoid hanging.
    
    Args:
        url: URL to fetch
        
    Returns:
        str: HTML content or empty string if fetch fails
    """
    if not url.startswith(('http://', 'https://')):
        url = 'http://' + url
    
    try:
        response = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        return response.text
    except Exception:
        return ""


def has_dangerous_keywords(text: str) -> tuple[bool, list[str]]:
    """
    Check for EXTREMELY dangerous keywords that indicate confirmed scams.
    These patterns are highly specific to avoid false positives.
    
    Args:
        text: Input text to analyze
        
    Returns:
        tuple: (is_dangerous, list_of_patterns_found)
    """
    text_lower = text.lower()
    
    # Only the most dangerous patterns (confirmed scams)
    dangerous_patterns = [
        (["won", "$", "congratulation", "bank details"], "Prize scam requesting bank information"),
        (["send", "bank details", "money"], "Direct request for bank details"),
        (["account suspended", "click here", "verify"], "Account suspension phishing link"),
        (["free", "iphone", "click here"], "Free product phishing scam"),
        (["lottery", "winner", "claim", "fee"], "Lottery fee scam"),
        (["social security", "verify", "immediately"], "Identity theft attempt"),
        (["western union", "money gram", "transfer"], "Wire transfer scam"),
        (["irs", "tax", "refund", "immediately"], "Tax refund scam"),
        (["paypal", "limited", "access", "verify"], "PayPal account scam"),
    ]
    
    found_patterns = []
    for keywords, description in dangerous_patterns:
        if all(keyword in text_lower for keyword in keywords):
            found_patterns.append(description)
    
    return len(found_patterns) > 0, found_patterns


def has_suspicious_patterns(text: str) -> tuple[bool, list[str]]:
    """
    Check for suspicious but not necessarily dangerous patterns.
    These trigger 'suspicious' classification instead of 'dangerous'.
    
    Args:
        text: Input text to analyze
        
    Returns:
        tuple: (is_suspicious, list_of_patterns_found)
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
    
    # Multiple suspicious words without clear scam indicators
    suspicious_words_count = sum(1 for word in email_suspicious_words if word in text_lower)
    if suspicious_words_count >= 3 and not any(word in text_lower for word in ["click", "link", "http", "bank details", "send money"]):
        suspicious_patterns.append(f"Multiple suspicious words detected ({suspicious_words_count}) without clear scam indicators")
    
    # Shortened URLs (often used to hide malicious links)
    if "bit.ly" in text_lower or "tinyurl" in text_lower or "short.url" in text_lower:
        suspicious_patterns.append("Contains shortened URL (often used to hide malicious links)")
    
    # Urgency without clear action
    urgency_words = ["urgent", "immediately", "asap", "action required"]
    if any(word in text_lower for word in urgency_words) and not any(word in text_lower for word in ["click", "verify", "link"]):
        suspicious_patterns.append("Urgent language used without clear action")
    
    return len(suspicious_patterns) > 0, suspicious_patterns


def explain_email(text: str) -> list[str]:
    """
    Generate human-readable explanations for email classification.
    Identifies suspicious words and provides context.
    
    Args:
        text: Email text
        
    Returns:
        list: Explanation strings
    """
    found = [word for word in email_suspicious_words if word in text.lower()]
    explanations = []

    # Check if this is a legitimate message first
    legitimate_patterns = [
        "order #", "order number", "tracking", "shipped", "delivered",
        "receipt", "purchase", "appointment", "reminder", "meeting",
        "lunch", "dinner", "coffee", "doctor", "dentist", "haircut",
        "thanks for your", "thank you for your"
    ]
    is_legitimate = any(pattern in text.lower() for pattern in legitimate_patterns)
    
    if is_legitimate:
        explanations.append("✓ Legitimate transactional or personal message")
        if "tracking" in text.lower() or "shipped" in text.lower() or "delivered" in text.lower():
            explanations.append("✓ Order/shipping notification (normal e-commerce activity)")
        if "order #" in text.lower() or "order number" in text.lower():
            explanations.append("✓ Purchase confirmation or update")
        if "appointment" in text.lower() or "reminder" in text.lower():
            explanations.append("✓ Calendar appointment or reminder")
        if "thanks" in text.lower() or "thank you" in text.lower():
            explanations.append("✓ Thank you message or receipt")
        return explanations

    if found:
        explanations.append(f"Suspicious wording detected: {', '.join(found[:4])}")

    if "http" in text or "www." in text:
        explanations.append("Contains link or URL inside message")
    
    if "bit.ly" in text.lower() or "tinyurl" in text.lower():
        explanations.append("⚠️ Contains shortened URL - potential link hiding")

    if not explanations:
        explanations.append("Prediction based on text classification patterns")

    return explanations


def explain_website(url: str, html_features: dict, url_features: dict) -> list[str]:
    """
    Generate detailed explanations for website risk assessment.
    Shows ALL red flags detected, not just the first one.
    
    Args:
        url: The website URL
        html_features: Extracted HTML features
        url_features: Extracted URL features
        
    Returns:
        list: Detailed explanation strings
    """
    explanations = []
    
    # URL-BASED RED FLAGS
    if not url_features["has_https"]:
        explanations.append("CRITICAL: Website does not use HTTPS (insecure connection - data can be intercepted)")
    elif url_features["has_https"]:
        explanations.append("Secure connection (HTTPS detected)")
    
    if url_features["has_at_symbol"]:
        explanations.append("URL contains '@' symbol (common phishing technique to hide real destination)")
    
    if url_features["has_ip_address"]:
        explanations.append("URL uses IP address instead of domain name (suspicious - legitimate sites rarely do this)")
    
    if url_features["has_suspicious_keyword"]:
        explanations.append("URL contains suspicious keywords (login, verify, secure, account, etc.)")
    
    if url_features["url_length"] > 100:
        explanations.append(f"Unusually long URL ({url_features['url_length']} characters) - often hides malicious intent")
    elif url_features["url_length"] > 75:
        explanations.append(f"Long URL ({url_features['url_length']} characters) - could be suspicious")
    
    if url_features["num_dash"] > 5:
        explanations.append(f"Multiple hyphens ({url_features['num_dash']}) in URL - often used in phishing domains")
    elif url_features["num_dash"] > 3:
        explanations.append(f"Several hyphens ({url_features['num_dash']}) in URL")
    
    if url_features["num_dots"] > 4:
        explanations.append(f"Many dots ({url_features['num_dots']}) in URL - possible subdomain trickery")
    
    if url_features["num_digits"] > 10:
        explanations.append(f"Many numbers ({url_features['num_digits']}) in URL - suspicious pattern")
    
    if url_features["domain_length"] > 30:
        explanations.append(f"Unusually long domain name ({url_features['domain_length']} characters)")
    
    # HTML-BASED RED FLAGS
    if html_features["has_password_input"]:
        explanations.append("Page contains password input field (potential credential harvesting)")
    
    if html_features["has_login_keyword"]:
        explanations.append("Page contains login-related wording (login, signin, verify, password, account)")
    
    if html_features["num_forms"] > 3:
        explanations.append(f"Unusually high number of forms ({html_features['num_forms']}) on page")
    elif html_features["num_forms"] > 0:
        explanations.append(f"Page contains {html_features['num_forms']} form(s)")
    
    if html_features["num_inputs"] > 10:
        explanations.append(f"Many input fields ({html_features['num_inputs']}) - possible data harvesting")
    
    if html_features["num_iframes"] > 0:
        explanations.append(f"Page contains {html_features['num_iframes']} iframe(s) - can load hidden malicious content")
    
    if html_features["num_scripts"] > 30:
        explanations.append(f"Many scripts ({html_features['num_scripts']}) - potential obfuscation")
    
    if html_features["num_links"] > 100:
        explanations.append(f"Unusually many links ({html_features['num_links']}) on page")
    
    if html_features["html_length"] < 500 and html_features["html_length"] > 0:
        explanations.append("Very small HTML page - possible redirect or minimal content")
    
    # LEGITIMATE INDICATORS
    red_flags_count = sum([
        not url_features["has_https"],
        url_features["has_at_symbol"],
        url_features["has_ip_address"],
        url_features["has_suspicious_keyword"],
        url_features["url_length"] > 75,
        html_features["has_password_input"],
        html_features["num_iframes"] > 0
    ])
    
    if red_flags_count == 0 and url_features["has_https"]:
        explanations.insert(0, "Uses secure HTTPS connection")
    
    if red_flags_count == 0 and not explanations:
        explanations.append("No obvious phishing indicators detected")
        explanations.append("Prediction based on URL and HTML feature patterns")
    
    if not explanations:
        explanations.append("Prediction based on URL and HTML features analysis")
    
    return explanations


# ======================
# MODEL AND DATA LOADING
# ======================
base_dir = Path(__file__).resolve().parent

# Load rule-based detection lists
trusted_domains = load_lines(base_dir / "trusted_domains.txt")      # Whitelist of safe domains
phishtank_urls = load_lines(base_dir / "phishtank_urls.txt")        # Blacklist of known phishing URLs

# Load ML models
# Email model: Logistic Regression with TF-IDF vectorization
email_model = joblib.load(base_dir / "email_model.pkl")
email_vectorizer = joblib.load(base_dir / "email_vectorizer.pkl")

# Website model: Random Forest with 19 URL + HTML features
website_model = joblib.load(base_dir / "website_model.pkl")

# Feature columns for the Random Forest model
website_feature_cols = [
    "url_length",
    "num_dots",
    "has_at_symbol",
    "has_https",
    "num_dash",
    "num_slash",
    "num_digits",
    "has_ip_address",
    "has_suspicious_keyword",
    "domain_length",
    "html_length",
    "num_forms",
    "num_links",
    "num_scripts",
    "num_iframes",
    "num_inputs",
    "num_buttons",
    "has_password_input",
    "has_login_keyword",
]

# Keyword lists for detection
suspicious_keywords = [
    "login", "verify", "secure", "account", "update",
    "bank", "confirm", "signin", "password", "paypal"
]

email_suspicious_words = [
    "urgent", "verify", "account", "suspended",
    "click", "login", "password", "confirm", "bank"
]


def is_trusted_domain(url: str) -> bool:
    """
    Check if URL domain is in trusted whitelist.
    Returns True for safe, legitimate domains like google.com.
    """
    domain = normalize_domain(url)
    return domain in trusted_domains


def is_known_phishtank_url(url: str) -> bool:
    """
    Check if URL is in PhishTank blacklist.
    Returns True for verified phishing URLs.
    """
    return normalize_url(url) in phishtank_urls


# ======================
# REQUEST DATA MODEL
# ======================
class RequestData(BaseModel):
    """Expected request body structure for /analyze endpoint"""
    input: str           # The email content or URL to analyze
    mode: str = "auto"   # 'auto', 'email', or 'url'


# ======================
# EMAIL ANALYSIS FUNCTION
# ======================
def analyze_email(user_input: str):
    # FIRST: Check for EXTREMELY dangerous patterns (confirmed scams)
    is_dangerous, dangerous_patterns = has_dangerous_keywords(user_input)
    
    if is_dangerous:
        explanations = [
            "🚨 CONFIRMED SCAM PATTERNS DETECTED:",
            *[f"  • {pattern}" for pattern in dangerous_patterns[:5]],
            "⚠️ This is a known fraud attempt - DO NOT RESPOND or CLICK ANY LINKS"
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
    
    # SECOND: Check for legitimate patterns
    legitimate_patterns = [
        "order #", "order number", "tracking", "shipped", "delivered",
        "receipt", "purchase", "appointment", "reminder", "meeting",
        "lunch", "dinner", "coffee", "doctor", "dentist", "haircut",
        "thanks for your", "thank you for your"
    ]
    
    text_lower = user_input.lower()
    is_clearly_legitimate = any(pattern in text_lower for pattern in legitimate_patterns)
    
    if is_clearly_legitimate:
        return {
            "risk": "safe",
            "type": "legitimate",
            "confidence": 0.85,
            "model_used": "Legitimate Pattern Detection",
            "input_type": "email",
            "explanation": ["✓ Legitimate personal or transactional message"],
            "action": "ignore",
        }
    
    # THIRD: ML Model with proper label handling
    try:
        clean_text = clean_email_text(user_input)
        X = email_vectorizer.transform([clean_text])
        
        # Get prediction (handles both string and numeric labels)
        pred_raw = email_model.predict(X)[0]
        
        # Convert string labels to numeric if needed
        if isinstance(pred_raw, str):
            if pred_raw.lower() in ['phishing email', 'phishing', 'spam', '1', 'true']:
                pred = 1
            elif pred_raw.lower() in ['safe email', 'legitimate', 'safe', '0', 'false']:
                pred = 0
            else:
                pred = 0  # default to safe
        else:
            pred = int(pred_raw)
        
        # Get probability
        if hasattr(email_model, "predict_proba"):
            prob = float(email_model.predict_proba(X)[0][1])
        else:
            prob = 1.0 if pred == 1 else 0.0

        risk = normalize_risk(prob)
        
    except Exception as e:
        print(f"Email ML error: {e}")
        # Fallback to rule-based classification
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
        prob = max(prob, 0.45)
    
    explanation = explain_email(user_input)
    
    if is_suspicious and suspicious_patterns:
        explanation = ["⚠️ " + pattern for pattern in suspicious_patterns] + explanation
    
    return {
        "risk": risk,
        "type": "phishing" if pred == 1 else "legitimate",
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
    """
    Analyze URL for phishing indicators.
    Uses: trusted domain whitelist -> PhishTank blacklist -> ML model with HTML features
    """
    # RULE 1: trusted domain (immediate safe classification)
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

    # RULE 2: known phishing URL (immediate dangerous classification)
    if is_known_phishtank_url(user_input):
        return {
            "risk": "dangerous",
            "type": "phishing",
            "confidence": 0.99,
            "model_used": "PhishTank Rule",
            "input_type": "url",
            "explanation": ["🚨 URL matched known verified phishing database (PhishTank)", "🚨 This is a confirmed malicious site - do not interact"],
            "action": "report",
        }

    # Fetch HTML content for feature extraction
    html = fetch_html_from_url(user_input)

    # Extract features for ML model
    url_features = extract_url_features(user_input)
    html_features = extract_html_features(html)

    # Combine features into a single row for prediction
    row = {**url_features, **html_features}
    X_df = pd.DataFrame([row])[website_feature_cols]

    # Run Random Forest prediction
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
    """
    Main analysis endpoint - accepts email or URL input and returns risk assessment.
    
    Request body:
        input (str): The email content or URL to analyze
        mode (str): 'auto', 'email', or 'url' - defaults to 'auto'
    
    Returns:
        dict: Risk assessment with classification, confidence, explanation, and action
    """
    # Determine input type based on content and user mode
    input_type = detect_input_type(data.input, data.mode)

    # Route to appropriate analyzer
    if input_type == "email":
        return analyze_email(data.input)

    return analyze_website(data.input)
