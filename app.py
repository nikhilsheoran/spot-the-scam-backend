from flask import Flask, jsonify, request
import requests
import ssl
import socket
from urllib.parse import urlparse
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from bs4 import BeautifulSoup
from urllib.parse import urlencode
from urllib.request import urlretrieve
import uuid
import boto3
import os
from google_play_scraper import app as play_store_app
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import NotFittedError
import pickle
import re


# Import feature extraction functions
from feature import *

app = Flask(__name__)

# Initialize the fraud detection model
try:
    with open('fraud_detection_model.pkl', 'rb') as file:
        fraud_model = pickle.load(file)
    print("Loaded pre-trained fraud detection model.")
except FileNotFoundError:
    fraud_model = LogisticRegression()
    print("Created new fraud detection model.")

# Initialize training data for fraud detection
X_train_fraud = []
y_train_fraud = []

def check_ssl(url):
    try:
        hostname = urlparse(url).hostname
        context = ssl.create_default_context()
        with socket.create_connection((hostname, 443)) as sock:
            with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                return 100  # SSL valid
    except Exception:
        return 0  # SSL invalid

def get_prediction(url, model_path):
    url_features = extract_features(url)
    url_features_array = np.array([url_features])
    keras_model = keras.models.load_model(model_path)
    prediction = keras_model.predict(url_features_array)
    return round(prediction[0][0] * 100, 3), url_features

def contact_details_score(url):
    extracted_numbers = extract_phone_numbers(url)
    if extracted_numbers:
        return 100
    else:
        return 0

def extract_phone_numbers(url, timeout=30):
    try:
        # Use a session to persist settings across requests
        session = requests.Session()
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36',
            'sec-ch-ua': '"Google Chrome";v="129", "Not=A?Brand";v="8", "Chromium";v="129"',
            'sec-ch-ua-arch': '"x86"',
            'sec-ch-ua-full-version': '"129.0.6668.60"',
            'sec-ch-ua-full-version-list': '"Google Chrome";v="129.0.6668.60", "Not=A?Brand";v="8.0.0.0", "Chromium";v="129.0.6668.60"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-model': '""',
            'sec-ch-ua-platform': '"Windows"',
            'sec-ch-ua-platform-version': '"15.0.0"',
            'sec-fetch-dest': 'document',
            'sec-fetch-mode': 'navigate',
            'sec-fetch-site': 'same-origin',
            'sec-fetch-user': '?1',
            'upgrade-insecure-requests': '1'
        }
        session.headers.update(headers)

        response = session.get(url, timeout=timeout)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        soup = BeautifulSoup(response.text, 'html.parser')
        text_content = ' '.join(soup.stripped_strings)

        # Improved phone number pattern to capture international numbers like +91-80-61561999
        phone_pattern = re.compile(r'''
            (?:(?:\+|00)[1-9]\d{0,3}[\s.-]?)?    # International prefix +91 or 0091
            (?:\(?\d{1,4}\)?[\s.-]?)?            # Optional area code in parentheses or separated by dash/space
            \d{1,4}[\s.-]?\d{1,4}[\s.-]?\d{1,9}  # Main part of the phone number (1 to 4 digits followed by optional separator)
        ''', re.VERBOSE)
        
        phone_numbers = phone_pattern.findall(text_content)

        # Clean up the extracted numbers and ensure uniqueness
        filtered_numbers = set()
        for num in phone_numbers:
            cleaned_num = re.sub(r'\s+', ' ', num).strip()  # Clean up spaces
            digits = ''.join(re.findall(r'\d', cleaned_num))  # Extract digits
            if len(digits) >= 7 and not re.match(r'^(\d)\1+$', digits):  # At least 7 digits and not all same digit
                filtered_numbers.add(cleaned_num)  # Add to set for uniqueness

        return list(filtered_numbers)
    except requests.Timeout:
        print(f"Error: The request to {url} timed out. The server might be slow or unresponsive.")
    except requests.RequestException as e:
        print(f"Error: An error occurred while fetching the webpage: {e}")
    except Exception as e:
        print(f"Error: An unexpected error occurred: {e}")
    
    return []


def analyze_content(url):
    try:
        session = requests.Session()
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36',
            'sec-ch-ua': '"Google Chrome";v="129", "Not=A?Brand";v="8", "Chromium";v="129"',
            'sec-ch-ua-arch': '"x86"',
            'sec-ch-ua-full-version': '"129.0.6668.60"',
            'sec-ch-ua-full-version-list': '"Google Chrome";v="129.0.6668.60", "Not=A?Brand";v="8.0.0.0", "Chromium";v="129.0.6668.60"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-model': '""',
            'sec-ch-ua-platform': '"Windows"',
            'sec-ch-ua-platform-version': '"15.0.0"',
            'sec-fetch-dest': 'document',
            'sec-fetch-mode': 'navigate',
            'sec-fetch-site': 'same-origin',
            'sec-fetch-user': '?1',
            'upgrade-insecure-requests': '1'
        }
        session.headers.update(headers)

        response = session.get(url, timeout=30)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        soup = BeautifulSoup(response.text, 'html.parser')
        text_content = ' '.join(soup.stripped_strings)


        inputs = nlp_tokenizer(text_content, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = nlp_model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)
        content_similarity = probabilities[0][1].item()

        form_presence = 1 if soup.find('form') else 0

        suspicious_keywords = ['login', 'password', 'credit card', 'social security', 'urgent', 'verify']
        keyword_count = sum(keyword in text_content.lower() for keyword in suspicious_keywords)
        keyword_score = max(0, 1 - keyword_count / len(suspicious_keywords))

        # Improved content score calculation
        content_length = len(text_content)
        length_score = min(1, content_length / 1000)  # Assume 1000 characters is a good length
        link_count = len(soup.find_all('a'))
        link_score = min(1, link_count / 20)  # Assume 20 links is a good number

        content_score = (content_similarity + (1 - form_presence) + keyword_score + length_score + link_score) / 5 * 100
        
        text_analysis = []
        if form_presence:
            text_analysis.append("Suspicious forms often ask for personal details like usernames, passwords, or credit card info.")
        if keyword_count > 0:
            text_analysis.append(f"Found {keyword_count} suspicious keywords that are common in phishing attempts.")
        if 'update' in text_content.lower() or 'verify' in text_content.lower():
            text_analysis.append("The page asks users to update or verify information, a common phishing tactic.")
        if 'click here' in text_content.lower():
            text_analysis.append("The page contains links commonly used in phishing attempts to redirect users to malicious websites.")
        if 'congratulations' in text_content.lower():
            text_analysis.append("The page contains 'congratulations', which is often used in scams to make users believe they have won a prize or reward.")
        if 'unsubscribe' in text_content.lower():
            text_analysis.append("'Unsubscribe' links may confirm email addresses for phishing.")
        if not text_analysis:
            text_analysis.append("No specific suspicious elements found in the text content.")

        return content_score, text_analysis
    except Exception as e:
        print(f"Content analysis error: {e}")
        return 0, ["Unable to analyze content"]

def analyze_image(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        img_t = image_transforms(img)
        batch_t = torch.unsqueeze(img_t, 0)

        with torch.no_grad():
            out = image_model(batch_t)

        probabilities = torch.nn.functional.softmax(out[0], dim=0)
        general_score = probabilities.max().item()

        img_cv = cv2.imread(image_path)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(gray, 100, 200)
        edge_ratio = np.sum(edges > 0) / edges.size

        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))
        dilation = cv2.dilate(thresh, kernel, iterations=1)
        contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        text_regions = sum(cv2.contourArea(c) for c in contours)
        text_ratio = text_regions / (img_cv.shape[0] * img_cv.shape[1])

        orb = cv2.ORB_create()
        kp = orb.detect(img_cv, None)
        logo_score = len(kp) / 1000

        image_score = (general_score + (1 - edge_ratio) + (1 - text_ratio) + logo_score) / 4 * 100

        image_analysis = []
        if edge_ratio > 0.1:
            image_analysis.append("The image contains many edges, which could indicate a complex or poorly designed interface, often seen in fraudulent websites.")
        if text_ratio > 0.6:
            image_analysis.append("The image contains a large amount of text, which is common in phishing attempts as they try to convey a lot of information quickly.")
        if logo_score < 0.1:
            image_analysis.append("No clear logo detected, which is unusual for legitimate websites. Legitimate websites usually have a distinct and recognizable logo.")
        if general_score < 0.5:
            image_analysis.append("The image doesn't strongly resemble common website layouts, which indicates a non-standard or suspicious design.")
        if edge_ratio < 0.05:
            image_analysis.append("The image contains very few edges, which might indicate a lack of detailed content or a simplistic design.")
        if text_ratio < 0.1:
            image_analysis.append("The image contains very little text, which could indicate a lack of information or an attempt to hide details.")
        if logo_score > 0.5:
            image_analysis.append("A clear and prominent logo is detected, which is a positive sign of legitimacy.")
        if general_score > 0.8:
            image_analysis.append("The image strongly resembles common website layouts, which is a positive indicator of legitimacy.")
        if not image_analysis:
            image_analysis.append("No specific suspicious elements found in the image. The image appears to be typical of legitimate websites.")

        return image_score, image_analysis
    except Exception as e:
        print(f"Image analysis error: {e}")
        return 0, ["Unable to analyze image"]

def calculate_overall_score(scores):
    weights = {
        'ssl_score': 0.18,
        'url_score': 0.24,
        'content_score': 0.29,
        'image_score': 0.29
    }
    overall_score = sum(score * weights[key] for key, score in scores.items())
    return overall_score

def train_model():
    global classification_model, feedback_data
    if len(feedback_data) > 1:
        X = feedback_data.drop('label', axis=1)
        y = feedback_data['label']

        le = LabelEncoder()
        y = le.fit_transform(y)

        classification_model.fit(X, y)
        y_pred = classification_model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        print(f"Model updated. Current accuracy: {accuracy:.2f}")
        if len(feedback_data) > 10:
            print(classification_report(y, y_pred))
    else:
        print("Not enough data to train the model yet. Using simple heuristics for now.")

def analyze_website(url, image_path):
    ssl_score = check_ssl(url)
    url_score = get_prediction(url, model_path)
    content_score, text_analysis = analyze_content(url)
    image_score, image_analysis = analyze_image(image_path)

    scores = {
        'ssl_score': ssl_score,
        'url_score': 100-url_score,
        'content_score': content_score,
        'image_score': image_score
    }

    overall_score = calculate_overall_score(scores)

    return scores, overall_score, text_analysis, image_analysis

def get_verdict(overall_score):
    if overall_score >= 90:
        return "The website is extremely likely to be legitimate. It has passed all checks with high scores."
    elif 80 <= overall_score < 90:
        return "The website is highly likely to be legitimate. It has passed most checks with good scores."
    elif 70 <= overall_score < 80:
        return "The website is likely to be legitimate. It has passed several checks with decent scores."
    elif 60 <= overall_score < 70:
        return "The website is somewhat likely to be legitimate. It has passed some checks but may need further verification."
    elif 50 <= overall_score < 60:
        return "The website is possibly legitimate but has some concerning factors. Proceed with caution."
    elif 40 <= overall_score < 50:
        return "The website is likely to be fraudulent. It has failed several checks and may pose a risk."
    elif 30 <= overall_score < 40:
        return "The website is highly likely to be fraudulent. It has failed most checks and is very suspicious."
    else:
        return "The website is extremely likely to be fraudulent. It has failed all checks and should be avoided."


# New functions for app analysis
def calculate_app_score(app_data):
    scoring_criteria = {
        "installs_score": {"weight": 0.2},
        "reviews_score": {"weight": 0.2},
        "description_length": {"weight": 0.15, "range": (100, 4000)},
        "updates": {"weight": 0.15, "range": (1, 100)},
        "privacy_policy": {"weight": 0.15, "range": (0, 1)},
        "developer_apps": {"weight": 0.15, "range": (1, 100)}
    }

    def normalize(value, min_val, max_val):
        value = float(value)
        return (value - min_val) / (max_val - min_val) if min_val < value < max_val else (0 if value < min_val else 1)

    def calculate_installs_score(installs, ratings):
        installs = int(''.join(filter(str.isdigit, str(installs))) or 0)
        ratings = int(''.join(filter(str.isdigit, str(ratings))) or 0)
        if installs == 0:
            return 0
        review_ratio = ratings / installs
        return min((review_ratio / 0.05) * 100, 100)

    def calculate_reviews_score(rating):
        rating = float(rating)
        return (rating / 5) * 100

    total_score = 0
    features = []
    scores = {}

    installs_score = calculate_installs_score(app_data["installs"], app_data["ratings"])
    reviews_score = calculate_reviews_score(app_data["score"])

    features.append(installs_score / 100)
    total_score += installs_score * scoring_criteria["installs_score"]["weight"]
    scores["installs_score"] = round(installs_score, 2)

    features.append(reviews_score / 100)
    total_score += reviews_score * scoring_criteria["reviews_score"]["weight"]
    scores["reviews_score"] = round(reviews_score, 2)

    for param, criteria in scoring_criteria.items():
        if param in ["installs_score", "reviews_score"]:
            continue

        value = app_data.get(param, 0)
        normalized_value = normalize(value, *criteria["range"])
        weighted_score = normalized_value * criteria["weight"] * 100
        features.append(normalized_value)
        total_score += weighted_score
        scores[param] = round(weighted_score / criteria["weight"], 2)

    total_score = min(total_score, 100)

    return round(total_score, 2), features, scores

def fetch_app_data(app_id):
    try:
        app_info = play_store_app(app_id)
        app_data = {
            "installs": app_info.get("installs", 0),
            "ratings": app_info.get("ratings", 0),
            "score": app_info.get("score", 0),
            "description_length": len(app_info.get("description", "")),
            "updates": len(app_info.get("recentChanges", [])),
            "privacy_policy": 1 if app_info.get("privacyPolicy") else 0,
            "developer_apps": app_info.get("developerApps", 1),
            "icon": app_info.get("icon","")
        }
        return app_data
    except Exception as e:
        print(f"Error fetching app data: {e}")
        return None

def get_app_verdict(score):
    if score >= 75:
        return "The app is highly likely legitimate."
    elif 50 <= score < 75:
        return "The app is likely legitimate."
    elif 25 <= score < 50:
        return "The app is likely fraudulent."
    else:
        return "The app is highly likely fraudulent."

def retrain_fraud_model():
    global fraud_model
    if len(set(y_train_fraud)) > 1:
        fraud_model.fit(np.array(X_train_fraud), np.array(y_train_fraud))
        print("Fraud detection model retrained with new data.")
        
        with open('fraud_detection_model.pkl', 'wb') as file:
            pickle.dump(fraud_model, file)
    else:
        print("Not enough diverse data to train the fraud detection model yet.")

@app.route('/analyze', methods=['POST'])
def analyze():
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return jsonify({"message": "Missing or invalid authorization token"}), 401

    token = auth_header.split(" ")[1]
    if token != "a3f1c1b0f4a99b8e4f95c2c56f4f63d0":
        return jsonify({"message": "Invalid authorization token"}), 401
    
    data = request.json
    analysis_type = data.get('type')
    url = data.get('url')

    if not url:
        return jsonify({"error": "URL is required"}), 400

    if analysis_type == "website":
        return analyze_website_endpoint(url)
    elif analysis_type == "app":
        return analyze_app_endpoint(url)
    else:
        return jsonify({"error": "Invalid analysis type. Must be 'website' or 'app'"}), 400

def analyze_website_endpoint(url):
    # Existing website analysis logic

    s3_access_key_id = "AKIAZFVNOOFMQBDSRGMR"
    s3_secret_key = "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"
    s3_bucket = "spot-the-scam"
    s3_region = "us-east-1"  

    analysisId = str(uuid.uuid4())
    unique_key = f"screenshots/{analysisId}.jpeg"

    params = {
        "access_key": "96243327e75c4f948ce32b236e824ef4",
        "url": url,
    }

    urlretrieve("https://api.apiflash.com/v1/urltoimage?" + urlencode(params), "screenshots/screenshot.jpeg")

    s3_client = boto3.client('s3', aws_access_key_id=s3_access_key_id, aws_secret_access_key=s3_secret_key, region_name=s3_region)
    s3_client.upload_file("screenshots/screenshot.jpeg", s3_bucket, unique_key)

    ssl_score = check_ssl(url)
    model_path = r"./model/Malicious_URL_Prediction.h5"
    url_score, url_features = get_prediction(url, model_path)
    content_score, text_analysis = analyze_content(url)
    image_score, image_analysis = analyze_image("screenshots/screenshot.jpeg")

    scores = {
        'ssl_score': ssl_score,
        'url_score': 100 - url_score,
        'content_score': content_score,
        'image_score': image_score
    }

    overall_score = calculate_overall_score(scores)
    verdict = get_verdict(overall_score)

    return jsonify({
        "analysisId": analysisId,
        "inputParameters": {
            "websiteScreenshot": f"https://spot-the-scam.s3.amazonaws.com/{unique_key}",
            "domainDetails": url_features,
            "sslDetails": ssl_score > 50,
            "contactDetails": "",
            "content": ""
        },
        "outputParameters": {
            "overallScore": {
                "score": overall_score,
                "heading": "Overall Score",
                "reason": verdict,
            },
            "sslScore": {
                "score": ssl_score,
                "heading": "SSL Score",
                "reason": "Indicates whether the website uses SSL."
            },
            "contentQualityScore": {
                "score": content_score,
                "heading": "Content Quality Score",
                "reason": text_analysis[0]
            },
            "contactDetailsScore": {
                "score": 90,
                "heading": "Contact Details Score",
                "reason": "A high score indicates the presence of contact details, which enhances the website's credibility."
            },
            "screenshotScore": {
                "score": image_score,
                "heading": "Screenshot Score",
                "reason": image_analysis[0]
            },
            "domainScore": {
                "score": 100 - url_score,
                "heading": "Domain Score",
                "reason": "Indicates the legitimacy of the URL based on various factors"
            },
        },
    })

def analyze_app_endpoint(app_id):
    app_data = fetch_app_data(app_id)
    if not app_data:
        return jsonify({"error": "Failed to fetch app data"}), 500

    score, features, individual_scores = calculate_app_score(app_data)
    verdict = get_app_verdict(score)

    try:
        prediction = fraud_model.predict([features])
        fraud_probability = fraud_model.predict_proba([features])[0][1]
        fraud_status = "fraudulent" if prediction == 1 else "not fraudulent"
    except NotFittedError:
        fraud_status = "unable to determine"
        fraud_probability = None

    return jsonify({
        "analysisId": str(uuid.uuid4()),
        "inputParameters": {
            "appId": app_id,
            "appDetails": app_data
        },
        "outputParameters": {
            "overallScore": {
                "score": score,
                "heading": "Overall Score",
                "reason": verdict,
            },
            "installsScore": {
                "score": individual_scores["installs_score"],
                "heading": "Installs Score",
                "reason": "Based on the ratio of ratings to installs"
            },
            "reviewsScore": {
                "score": individual_scores["reviews_score"],
                "heading": "Reviews Score",
                "reason": "Based on the app's rating"
            },
            "descriptionScore": {
                "score": individual_scores["description_length"],
                "heading": "Description Score",
                "reason": "Based on the length and quality of the app description"
            },
            "updatesScore": {
                "score": individual_scores["updates"],
                "heading": "Updates Score",
                "reason": "Based on the frequency of app updates"
            },
            "privacyPolicyScore": {
                "score": individual_scores["privacy_policy"],
                "heading": "Privacy Policy Score",
                "reason": "Based on the presence of a privacy policy"
            },
            "developerScore": {
                "score": individual_scores["developer_apps"],
                "heading": "Developer Score",
                "reason": "Based on the developer's history and other apps"
            },
        },
    })

@app.route('/train', methods=['POST'])
def train():
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return jsonify({"message": "Missing or invalid authorization token"}), 401

    token = auth_header.split(" ")[1]
    if token != "a3f1c1b0f4a99b8e4f95c2c56f4f63d0":
        return jsonify({"message": "Invalid authorization token"}), 401
    
    # Here you can add additional token validation logic if needed
    global feedback_data, classification_model
    data = request.json
    new_data = pd.DataFrame([data])
    feedback_data = pd.concat([feedback_data, new_data], ignore_index=True)

    if len(feedback_data) > 1:
        X = feedback_data.drop('label', axis=1)
        y = feedback_data['label']

        le = LabelEncoder()
        y = le.fit_transform(y)

        classification_model.fit(X, y)
        y_pred = classification_model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        
        return jsonify({
            "message": "Model updated successfully",
            "accuracy": accuracy,
            "data_points": len(feedback_data)
        })
    else:
        return jsonify({
            "message": "Not enough data to train the model yet. Using simple heuristics for now.",
            "data_points": len(feedback_data)
        })

if __name__ == '__main__':
    # Global variables and model initialization
    feedback_data = pd.DataFrame(columns=['ssl_score', 'url_score', 'content_score', 'image_score', 'overall_score', 'label'])
    classification_model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Load models and initialize tokenizers
    model_path = r"./model/Malicious_URL_Prediction.h5"
    nlp_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    nlp_model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
    image_model = models.resnet50(pretrained=True)
    image_model.eval()

    keras_model = keras.models.load_model(model_path)

    image_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    app.run(host='0.0.0.0', port=5000)