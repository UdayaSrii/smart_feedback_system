from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from tinydb import TinyDB, Query
from datetime import datetime
import uuid
import os

# --- INITIALIZATION ---
# Download the VADER lexicon for sentiment analysis
nltk.download('vader_lexicon')

app = Flask(__name__, static_folder='public')
CORS(app)

# Initialize VADER analyzer
analyzer = SentimentIntensityAnalyzer()

# Initialize Databases
db = TinyDB('db.json')
user_db = TinyDB('users.json')

# --- CONFIGURATION ---
# Security: Banned keywords for the profanity filter
BANNED_WORDS = [
    "fuck", "shit", "asshole", "bitch", "bastard", "dick", "piss", "cunt",
    "nigger", "faggot", "retard", "slut", "whore",
    "crypto", "bitcoin", "winner", "prize", "buy now", "free money"
]

# Business Logic: Keywords to be forced into the Neutral category
NEUTRAL_OVERRIDES = ["ok", "okay", "not bad"]

# --- ROUTES ---

@app.route('/')
def serve_index():
    """Serves the frontend interface."""
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/login', methods=['POST'])
def login():
    """Handles Admin authentication."""
    data = request.json
    User = Query()
    user = user_db.search((User.username == data.get('identifier')) & (User.password == data.get('password')))
    return jsonify({"status": "success"}) if user else (jsonify({"status": "error"}), 401)

@app.route('/api/feedback', methods=['POST'])
def add_feedback():
    """Processes feedback, performs VADER analysis, and saves to DB."""
    data = request.json
    msg_raw = data.get('message', '')
    msg_clean = msg_raw.lower().strip()
    
    # 1. LAYER 1: Profanity Filter (Exact Word Matching)
    words = msg_clean.split()
    if any(word in words for word in BANNED_WORDS):
        return jsonify({"status": "error", "message": "Inappropriate language"}), 400

    # 2. LAYER 2: Manual Neutral Overrides
    if msg_clean in NEUTRAL_OVERRIDES:
        sentiment = "Neutral"
    else:
        # 3. LAYER 3: VADER Sentiment Analysis
        # VADER calculates a 'compound' score from -1.0 to 1.0
        scores = analyzer.polarity_scores(msg_raw)
        compound = scores['compound']
        
        if compound >= 0.05:
            sentiment = "Positive"
        elif compound <= -0.05:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"

    # Save data to database
    entry = {
        'id': str(uuid.uuid4())[:8],
        'username': data.get('username', 'Anonymous'),
        'message': msg_raw,
        'sentiment': sentiment,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M")
    }
    db.insert(entry)
    
    return jsonify({"status": "success", "sentiment": sentiment})

@app.route('/api/history', methods=['GET'])
def get_history():
    """Returns all feedback in reverse chronological order."""
    return jsonify({"data": db.all()[::-1]})

@app.route('/api/feedback/<feedback_id>', methods=['DELETE'])
def delete_feedback(feedback_id):
    """Deletes a specific feedback entry."""
    Feedback = Query()
    db.remove(Feedback.id == feedback_id)
    return jsonify({"status": "success"})

@app.route('/api/feedback/clear_all', methods=['DELETE'])
def clear_all_feedback():
    """Wipes the entire feedback database."""
    db.truncate()
    return jsonify({"status": "success"})

# --- MAIN EXECUTION ---
if __name__ == '__main__':
    # Initialize a default admin user if the user_db is empty
    if not user_db.all():
        user_db.insert({'username': 'admin', 'password': 'password123'})
    
    # Run server on port 3000
    print("Server running at http://127.0.0.1:3000")
    app.run(host='0.0.0.0', port=3000, debug=True)