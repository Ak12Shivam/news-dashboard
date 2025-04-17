from flask import Flask, render_template, jsonify, send_from_directory
import threading
import time
import os
import json
import logging
from datetime import datetime
from news_updater import process_news  # Import the news processing function

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

# Initialize Flask app
app = Flask(__name__)

# Configuration
DATA_DIR = os.path.join('static', 'data')
LATEST_DATA_FILE = os.path.join(DATA_DIR, 'latest.json')
UPDATE_INTERVAL = 900  # 15 minutes in seconds
INITIAL_UPDATE_DELAY = 30  # Seconds to wait before first update

def ensure_data_directory():
    """Ensure the data directory exists"""
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(LATEST_DATA_FILE):
        with open(LATEST_DATA_FILE, 'w') as f:
            json.dump([], f)

def background_news_updater():
    """Background thread that periodically updates news data"""
    time.sleep(INITIAL_UPDATE_DELAY)  # Wait for Flask to start
    
    while True:
        try:
            logging.info("Starting scheduled news update...")
            process_news()  # Process news data
            
            # Find the latest generated file and update latest.json
            json_files = [
                f for f in os.listdir(DATA_DIR) 
                if f.startswith('news_analysis_') and f.endswith('.json')
            ]
            
            if json_files:
                latest_file = max(json_files)
                with open(os.path.join(DATA_DIR, latest_file), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                with open(LATEST_DATA_FILE, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)
                
                logging.info(f"Updated {LATEST_DATA_FILE} with latest data")
            else:
                logging.warning("No news data files found")
            
        except Exception as e:
            logging.error(f"Error in background updater: {str(e)}")
        
        time.sleep(UPDATE_INTERVAL)

@app.route('/')
def index():
    """Serve the main dashboard page"""
    return render_template('index.html')

@app.route('/static/data/latest.json')
def get_latest_data():
    """Endpoint to serve the latest news data"""
    ensure_data_directory()
    
    try:
        if not os.path.exists(LATEST_DATA_FILE):
            return jsonify([])
        
        # Add cache-busting query parameter
        timestamp = os.path.getmtime(LATEST_DATA_FILE)
        return send_from_directory(
            DATA_DIR, 
            'latest.json',
            mimetype='application/json',
            last_modified=timestamp
        )
    except Exception as e:
        logging.error(f"Error serving latest.json: {str(e)}")
        return jsonify({"error": "Could not load data"}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "last_update": str(datetime.fromtimestamp(os.path.getmtime(LATEST_DATA_FILE))) 
        if os.path.exists(LATEST_DATA_FILE) else "never"
    })

def start_background_updater():
    """Start the background news updater thread"""
    updater_thread = threading.Thread(target=background_news_updater)
    updater_thread.daemon = True  # Will stop when main thread stops
    updater_thread.start()
    logging.info("Started background news updater thread")

if __name__ == '__main__':
    ensure_data_directory()
    start_background_updater()
    
    # Run Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        use_reloader=False  # Disable Flask's reloader as we have our own updater
    )