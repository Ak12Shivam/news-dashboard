import re
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import concurrent.futures
from datetime import datetime
import time
import spacy
import json
import matplotlib.pyplot as plt
import logging
import google.generativeai as genai
from geopy.geocoders import Nominatim
import pandas as pd
import os
from collections import defaultdict

# Set up logging
logging.basicConfig(
    filename='news_updater.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Download necessary NLTK data
try:
    nltk.download(['punkt', 'stopwords', 'averaged_perceptron_tagger', 'maxent_ne_chunker', 'words'], quiet=True)
except Exception as e:
    logging.error(f"Failed to download NLTK data: {str(e)}")

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    logging.error(f"Failed to load spaCy model: {str(e)}")
    nlp = None

# Configure Gemini Pro API
GEMINI_API_KEY = "AIzaSyBbIHeQfr3HKeZmt9dS6GyeW4SDhXlLY5o"  # Replace with your Google Generative AI key
genai.configure(api_key=GEMINI_API_KEY)

# NewsAPI key
NEWS_API_KEY = "0576f865ce1e4aec8b27eb541869f6aa"  # Replace with your NewsAPI key from https://newsapi.org/

# Geocoder setup
geolocator = Nominatim(user_agent="news_updater")

# Output directory
OUTPUT_DIR = "static/data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# List of URLs to scrape (same as original)
urls = [
    "https://www.techradar.com/in", "https://www.digit.in", "https://www.inc42.com", "https://www.yourstory.com",
    "https://www.gizmodo.com", "https://www.mashable.com", "https://www.wired.com", "https://www.theverge.com",
    "https://www.techcrunch.com", "https://www.thehindubusinessline.com", "https://www.livemint.com",
    "https://www.moneycontrol.com", "https://www.economictimes.indiatimes.com", "https://www.ft.com",
    "https://www.businessinsider.com", "https://www.forbes.com", "https://www.bloomberg.com",
    "https://www.practo.com/healthfeed", "https://www.indiatoday.in/health", "https://www.curofy.com",
    "https://www.timesofindia.indiatimes.com/life-style/health-fitness", "https://www.medlineplus.gov",
    "https://www.medicalnewstoday.com", "https://www.healthline.com", "https://www.mayoclinic.org",
    "https://www.webmd.com", "https://www.thehindu.com/education", "https://www.timesofindia.indiatimes.com/education",
    "https://www.indiatoday.in/education-today", "https://www.edweek.org", "https://www.edsurge.com",
    "https://www.insidehighered.com", "https://www.edtechmagazine.com", "https://www.sportstar.thehindu.com",
    "https://www.indiatoday.in/sports", "https://www.timesofindia.indiatimes.com/sports", "https://www.thehindu.com/sport",
    "https://www.si.com", "https://www.bleacherreport.com", "https://www.skysports.com", "https://www.bbc.com/sport",
    "https://www.espn.com", "https://www.downtoearth.org.in", "https://www.timesofindia.indiatimes.com/home/science",
    "https://www.thehindu.com/sci-tech", "https://www.sciencenews.org", "https://www.nationalgeographic.com/science",
    "https://www.newscientist.com", "https://www.nature.com", "https://www.scientificamerican.com",
    "https://www.indiatoday.in/movies", "https://www.pinkvilla.com", "https://www.filmfare.com",
    "https://www.bollywoodhungama.com", "https://www.billboard.com", "https://www.rollingstone.com",
    "https://www.deadline.com", "https://www.hollywoodreporter.com", "https://www.variety.com",
    "https://www.thrillophilia.com/blog", "https://www.traveltriangle.com", "https://www.outlookindia.com/outlooktraveller",
    "https://www.cntraveler.com", "https://www.travelandleisure.com", "https://www.nationalgeographic.com/travel",
    "https://www.lonelyplanet.com", "https://www.perniaspopupshop.com", "https://www.femina.in", "https://www.vogue.in",
    "https://www.gq.com", "https://www.harpersbazaar.com", "https://www.elle.com", "https://www.vogue.com",
    "https://www.hebbarskitchen.com", "https://www.archanaskitchen.com", "https://www.sanjeevkapoor.com",
    "https://www.allrecipes.com", "https://www.seriouseats.com", "https://www.proptiger.com", "https://www.commonfloor.com",
    "https://www.housing.com", "https://www.magicbricks.com", "https://www.curbed.com", "https://www.autocarindia.com",
    "https://www.gaadiwaadi.com", "https://www.overdrive.in", "https://www.cardekho.com", "https://www.autocar.co.uk",
    "https://www.timesofindia.indiatimes.com/home/environment", "https://www.thehindu.com/sci-tech/energy-and-environment",
    "https://www.downtoearth.org.in", "https://www.scientificamerican.com/environment",
    "https://www.nationalgeographic.com/environment", "https://www.theguardian.com/environment",
    "https://www.ecowatch.com", "https://www.treehugger.com", "https://www.timesofindia.indiatimes.com/life-style",
    "https://www.vogue.in", "https://www.idiva.com", "https://www.femina.in", "https://www.self.com",
    "https://www.refinery29.com", "https://www.wellandgood.com", "https://www.mindbodygreen.com",
    "https://www.defenceaviationpost.com", "https://www.spsmai.com", "https://www.indiandefensenews.in",
    "https://www.bharat-rakshak.com", "https://www.globalsecurity.org", "https://www.defenseone.com",
    "https://www.militarytimes.com", "https://www.janes.com", "https://www.defensenews.com", "https://www.bbc.com/news",
    "https://www.cnn.com", "https://www.timesofindia.indiatimes.com", "https://www.thehindu.com",
    "https://www.hindustantimes.com", "https://www.indiatoday.in", "https://www.economictimes.indiatimes.com",
    "https://www.news18.com", "https://www.nytimes.com"
]

# Enhanced field classification with more categories
FIELD_KEYWORDS = {
    'Politics': ['government', 'election', 'party', 'minister', 'policy', 'political', 'law', 'bill', 'senate', 'congress'],
    'Sports': ['cricket', 'football', 'tennis', 'sports', 'game', 'player', 'team', 'tournament', 'match', 'olympics'],
    'Technology': ['tech', 'technology', 'software', 'digital', 'online', 'internet', 'ai', 'computer', 'smartphone', 'innovation'],
    'Environment': ['environment', 'climate', 'pollution', 'sustainability', 'green', 'energy', 'carbon', 'wildlife', 'conservation'],
    'Health': ['health', 'medical', 'disease', 'treatment', 'hospital', 'doctor', 'medicine', 'vaccine', 'pandemic', 'healthcare'],
    'Business': ['business', 'economy', 'market', 'company', 'finance', 'stock', 'investment', 'trade', 'industry', 'startup'],
    'Entertainment': ['movie', 'film', 'actor', 'music', 'celebrity', 'entertainment', 'hollywood', 'tv', 'streaming', 'awards'],
    'Science': ['science', 'research', 'experiment', 'physics', 'chemistry', 'biology', 'space', 'discovery', 'scientist', 'nasa'],
    'Education': ['education', 'school', 'college', 'university', 'student', 'teacher', 'learning', 'degree', 'academic', 'campus'],
    'World': ['global', 'international', 'world', 'nation', 'country', 'diplomacy', 'foreign', 'united nations', 'summit'],
    'Crime': ['crime', 'police', 'investigation', 'arrest', 'court', 'law', 'trial', 'justice', 'murder', 'theft'],
    'Lifestyle': ['fashion', 'travel', 'food', 'culture', 'trend', 'wellness', 'fitness', 'recipe', 'restaurant', 'hotel']
}

def fetch_newsapi_data():
    """Fetch data from NewsAPI with enhanced error handling"""
    try:
        api_url = f"https://newsapi.org/v2/top-headlines?country=us&apiKey={NEWS_API_KEY}&pageSize=50"
        response = requests.get(api_url, timeout=15)
        response.raise_for_status()
        data = response.json()
        articles = data.get('articles', [])
        news_text = ' '.join(
            f"{article.get('title', '')} {article.get('description', '')} {article.get('content', '')}"
            for article in articles if article.get('title')
        )
        logging.info(f"Fetched {len(articles)} articles from NewsAPI")
        return news_text
    except requests.RequestException as e:
        logging.error(f"Error fetching NewsAPI: {str(e)}")
        return ""

def scrape_website(url):
    """Enhanced web scraper with better content extraction"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'footer', 'iframe', 'noscript']):
            element.decompose()
            
        # Get text from meaningful tags
        content_tags = soup.find_all(['h1', 'h2', 'h3', 'p', 'article', 'section'])
        text = ' '.join(tag.get_text().strip() for tag in content_tags if tag.get_text().strip())
        
        if len(text) > 50000:
            text = text[:50000]
            
        logging.info(f"Scraped {len(text)} chars from {url}")
        return text
    except Exception as e:
        logging.error(f"Error scraping {url}: {str(e)}")
        return ""

def scrape_urls(urls):
    """Scrape multiple URLs with improved parallel processing"""
    logging.info(f"Scraping {len(urls)} URLs")
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(scrape_website, urls))
    valid_results = [r for r in results if r and len(r) > 100]
    logging.info(f"Scraped {len(valid_results)} URLs successfully")
    return valid_results

def clean_sentence(sentence):
    """Enhanced sentence cleaning with more patterns"""
    patterns = [
        r'©\s?\d{4}.*?(Reserved|notified otherwise)', 
        r'Copyright\s©.*?Reserved', 
        r'All Rights Reserved.*',
        r'All rights reserved.*', 
        r'Copyright .*', 
        r'\b\d{4}.*?rights reserved',
        r'Hearst Magazine Media, Inc.*',
        r'Commonfloor\.com.*',
        r'-\s*All Rights Reserved',
        r'ADVERTISEMENT',
        r'Sign up for.*',
        r'Read more.*',
        r'Continue reading.*',
        r'More:.*'
    ]
    for pattern in patterns:
        sentence = re.sub(pattern, '', sentence, flags=re.IGNORECASE)
    return re.sub(r'\s+', ' ', sentence).strip()

def is_valid_sentence(sentence):
    """More robust sentence validation"""
    cleaned = clean_sentence(sentence)
    if len(cleaned) < 20 or cleaned.lower().strip() in ["all rights", "copyright", ""]:
        return False
    if not any(c.isalpha() for c in cleaned) or len(cleaned.split()) < 3:
        return False
    if cleaned.startswith(('http://', 'https://')):
        return False
    return True

def split_long_sentence(sentence):
    """Improved sentence splitting logic"""
    if not nlp:
        return [sentence[:150]] if is_valid_sentence(sentence) else []
    if 100 <= len(sentence) <= 150:
        return [sentence] if is_valid_sentence(sentence) else []
    
    doc = nlp(sentence)
    chunks = []
    current_chunk = []
    current_length = 0
    
    for token in doc:
        word = token.text_with_ws
        word_len = len(word)
        if current_length + word_len > 150 and current_chunk:
            chunk_text = "".join(current_chunk).strip()
            if 100 <= len(chunk_text) <= 150 and is_valid_sentence(chunk_text):
                chunks.append(chunk_text)
            current_chunk = [word]
            current_length = word_len
        else:
            current_chunk.append(word)
            current_length += word_len
    
    if current_chunk:
        chunk_text = "".join(current_chunk).strip()
        if 100 <= len(chunk_text) <= 150 and is_valid_sentence(chunk_text):
            chunks.append(chunk_text)
        elif len(chunk_text) < 100 and is_valid_sentence(chunk_text):
            chunks.append(chunk_text + " Details emerging.")
    
    final_chunks = []
    for chunk in chunks:
        if len(chunk) > 150:
            final_chunks.extend(split_long_sentence(chunk))
        else:
            final_chunks.append(chunk)
    
    return final_chunks

def select_meaningful_sentences(sentences, min_n=6, max_n=14):
    """Improved sentence selection with better fallbacks"""
    cleaned_sentences = [clean_sentence(sent) for sent in sentences if is_valid_sentence(sent)]
    unique_sentences = list(dict.fromkeys(cleaned_sentences))
    if not unique_sentences:
        logging.warning("No valid sentences in cluster")
        return ["No meaningful content available."] * min_n

    concise_sentences = []
    for sent in unique_sentences:
        if len(sent) > 150:
            concise_sentences.extend(split_long_sentence(sent))
        elif 100 <= len(sent) <= 150:
            concise_sentences.append(sent)
        elif len(sent) < 100 and is_valid_sentence(sent):
            concise_sentences.append(sent + " More details to follow.")

    if len(concise_sentences) < min_n:
        for sent in unique_sentences:
            if len(sent) > 150:
                extra_chunks = split_long_sentence(sent)
                concise_sentences.extend([c for c in extra_chunks if c not in concise_sentences])
            if len(concise_sentences) >= max_n:
                break

    if not nlp:
        selected_sentences = concise_sentences[:max_n]
    else:
        docs = list(nlp.pipe(concise_sentences[:50]))  # Limit to top 50 for performance
        scored_sentences = []
        for doc in docs:
            score = len(doc.ents) * 5  # Entities are important
            score += len([token for token in doc if token.pos_ in ['NOUN', 'PROPN']]) * 2  # Nouns and proper nouns
            score -= len([token for token in doc if token.pos_ in ['DET', 'CCONJ']])  # Less important words
            scored_sentences.append((score, doc.text))
        
        sorted_sentences = sorted(scored_sentences, key=lambda x: x[0], reverse=True)
        n = min(max(min_n, len(sorted_sentences)), max_n)
        selected_sentences = [sent for _, sent in sorted_sentences[:n]]

    while len(selected_sentences) < min_n:
        selected_sentences.append(f"Update {len(selected_sentences) + 1}: Details pending soon.")

    logging.info(f"Selected {len(selected_sentences)} sentences")
    return selected_sentences[:max_n]

def cluster_sentences(sentences, n_clusters=10):
    """Enhanced clustering with better preprocessing"""
    if len(sentences) > 10000:
        sentences = sentences[:10000]
    try:
        vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=5000,
            ngram_range=(1, 2) ) # Include bigrams
        X = vectorizer.fit_transform(sentences)
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10  # Reduce warning
        )
        kmeans.fit(X)
        clusters = [[] for _ in range(n_clusters)]
        for i, label in enumerate(kmeans.labels_):
            clusters[label].append(sentences[i])
        return clusters
    except Exception as e:
        logging.error(f"Clustering failed: {str(e)}")
        return [sentences]

def generate_topic_name(sentences):
    """Improved topic name generation"""
    try:
        if not sentences:
            return "General News"
            
        vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=5,
            ngram_range=(1, 2)
        )
        X = vectorizer.fit_transform(sentences)
        features = vectorizer.get_feature_names_out()
        
        # Try to generate a more readable name
        if len(features) >= 3:
            return f"{features[0]}, {features[1]}, and {features[2]}"
        elif features:
            return ", ".join(features)
        else:
            return "Current Events"
    except Exception as e:
        logging.error(f"Topic name generation failed: {str(e)}")
        return "Breaking News"

def classify_topic(topic_name):
    """Enhanced topic classification with more fields"""
    topic_lower = topic_name.lower()
    for field, keywords in FIELD_KEYWORDS.items():
        if any(keyword in topic_lower for keyword in keywords):
            return field
    return 'General'

def extract_entities(text):
    """Enhanced entity extraction with more types"""
    if not nlp or not text:
        return {}
    
    doc = nlp(text[:100000])  # Limit text size for performance
    entities = {
        'PERSON': set(),
        'ORG': set(),
        'GPE': set(),
        'DATE': set(),
        'MONEY': set(),
        'EVENT': set(),
        'PRODUCT': set()
    }
    
    for ent in doc.ents:
        if ent.label_ in entities:
            entities[ent.label_].add(ent.text)
    
    return {k: list(v) for k, v in entities.items() if v}

def extract_locations(text):
    """Improved location extraction with caching"""
    if not nlp:
        return []
    
    doc = nlp(text[:50000])  # Limit text size for performance
    locations = []
    seen_locations = set()
    
    for ent in doc.ents:
        if ent.label_ == 'GPE' and ent.text not in seen_locations:
            seen_locations.add(ent.text)
            try:
                location = geolocator.geocode(ent.text, timeout=10)
                if location:
                    locations.append({
                        "name": ent.text,
                        "latitude": location.latitude,
                        "longitude": location.longitude
                    })
            except Exception as e:
                logging.warning(f"Geocoding failed for {ent.text}: {str(e)}")
    
    return locations[:20]  # Limit to top 20 locations

def extract_key_phrases(text, n=10):
    """Improved key phrase extraction"""
    if not nlp or not text:
        return []
    
    doc = nlp(text[:50000])  # Limit text size for performance
    phrases = []
    
    # Extract noun chunks
    for chunk in doc.noun_chunks:
        if len(chunk.text.split()) > 1:  # Only multi-word phrases
            phrases.append(chunk.text)
    
    # Extract named entities
    for ent in doc.ents:
        if ent.label_ in ['ORG', 'PRODUCT', 'EVENT', 'LAW']:
            phrases.append(ent.text)
    
    # Deduplicate and return top N
    return list(dict.fromkeys(phrases))[:n]

def generate_headline_points(sentences, num_points=15):
    """Generate more meaningful news headlines from content"""
    meaningful_sentences = select_meaningful_sentences(sentences, min_n=6, max_n=14)
    if not meaningful_sentences or meaningful_sentences[0].startswith("No meaningful"):
        return ["Breaking news update"] * num_points

    content = ' '.join(meaningful_sentences[:5])
    
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = (
            f"Generate {num_points} concise, meaningful news headlines (max 60 chars each) "
            f"from this text. Follow these rules:\n"
            f"1. Use complete sentences with subject-verb-object structure\n"
            f"2. Focus on the most important information\n"
            f"3. Avoid vague phrases like 'something happens'\n"
            f"4. Include key entities when possible\n"
            f"5. Make each headline distinct and specific\n\n"
            f"Text:\n{content}"
        )
        response = model.generate_content(prompt)
        headline_candidates = [
            line.strip() 
            for line in response.text.split('\n') 
            if line.strip() and 25 <= len(line.strip()) <= 60
            and not line.strip().startswith(('1.', '2.', '3.', '-', '*'))
        ]
    except Exception as e:
        logging.error(f"Gemini API failed: {str(e)}. Using fallback.")
        headline_candidates = []

    if not headline_candidates and nlp:
        docs = list(nlp.pipe(meaningful_sentences[:5]))
        headline_candidates = []
        for doc in docs:
            for sent in doc.sents:
                if len(sent.text) > 60:
                    continue
                if any(ent.label_ in ['PERSON', 'ORG', 'GPE'] for ent in sent.ents):
                    headline_candidates.append(sent.text)
                    if len(headline_candidates) >= num_points:
                        break

    # Ensure we have enough headlines
    seen_headlines = set()
    headline_points = []
    
    for headline in headline_candidates:
        headline = headline.strip()
        if (headline and headline not in seen_headlines 
            and 25 <= len(headline) <= 60
            and not headline.lower().startswith(('update', 'breaking'))):
            headline_points.append(headline)
            seen_headlines.add(headline)
        if len(headline_points) >= num_points:
            break

    # Fallback if we don't have enough good headlines
    field = "General"  # Default value for field
    while len(headline_points) < num_points:
        idx = len(headline_points)
        if meaningful_sentences and idx < len(meaningful_sentences):
            headline = meaningful_sentences[idx][:60]
            if len(headline) >= 25:
                headline_points.append(headline)
            else:
                headline_points.append(f"Major development in {field} sector")
        else:
            headline_points.append(f"Important update on topic {idx + 1}")

    return headline_points[:num_points]

def generate_topic_insight(sentences):
    """Improved insight generation"""
    if not sentences:
        return "Recent developments in this area."
    
    content = ' '.join(sentences[:5])
    
    # Try Gemini API first
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = (
            "Summarize this news content in 1-2 concise sentences. "
            "Focus on the most important developments and their potential impact:\n\n"
            f"{content}"
        )
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        logging.error(f"Insight generation failed: {str(e)}")
    
    # Fallback to NLP-based summary
    if nlp:
        doc = nlp(content[:10000])  # Limit size
        sentences = [sent.text for sent in doc.sents]
        if len(sentences) >= 2:
            return f"{sentences[0]} {sentences[1]}"
        elif sentences:
            return sentences[0]
    
    return "Recent developments impact this field."

def write_to_json(data, filename=None):
    """Write data to JSON with improved formatting"""
    try:
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"news_analysis_{timestamp}.json"
        
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        with open(filepath, 'w', encoding='utf-8') as jsonfile:
            json.dump(data, jsonfile, ensure_ascii=False, indent=2)
        
        # Also update the latest.json file
        latest_path = os.path.join(OUTPUT_DIR, 'latest.json')
        with open(latest_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logging.info(f"Data written to {filepath}")
    except Exception as e:
        logging.error(f"Failed to write JSON: {str(e)}")

def generate_graphs(parsed_data):
    """Enhanced graph generation with better formatting"""
    try:
        if not parsed_data:
            return {}
            
        # Bar chart: Number of Entity Types per Topic
        topics = [row['Topic Name'] for row in parsed_data]
        entity_counts = [len(row['Key Entities']) for row in parsed_data]
        
        plt.figure(figsize=(15, 8))
        bars = plt.bar(topics, entity_counts, color='skyblue')
        plt.title('Number of Entity Types per Topic', fontsize=14)
        plt.xlabel('Topics', fontsize=12)
        plt.ylabel('Number of Entity Types', fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        bar_filename = f"topic_entities_graph_{timestamp}.png"
        plt.savefig(os.path.join(OUTPUT_DIR, bar_filename))
        plt.close()

        # Pie chart: Distribution of News Topics
        fields = [row['Field'] for row in parsed_data]
        field_counts = defaultdict(int)
        for field in fields:
            field_counts[field] += 1
        
        # Sort by count and take top 8
        sorted_fields = sorted(field_counts.items(), key=lambda x: x[1], reverse=True)[:8]
        labels = [f[0] for f in sorted_fields]
        sizes = [f[1] for f in sorted_fields]
        
        plt.figure(figsize=(10, 10))
        colors = plt.cm.Paired(range(len(labels)))
        _, texts, autotexts = plt.pie(
            sizes, 
            labels=labels, 
            colors=colors,
            autopct='%1.1f%%',
            startangle=90,
            textprops={'fontsize': 10}
        )
        
        # Improve label readability
        for text in texts + autotexts:
            text.set_fontsize(10)
        
        plt.title('Distribution of News Topics', fontsize=14)
        plt.axis('equal')
        pie_filename = f"topic_distribution_pie_{timestamp}.png"
        plt.savefig(os.path.join(OUTPUT_DIR, pie_filename))
        plt.close()

        logging.info(f"Graphs saved: {bar_filename}, {pie_filename}")
        return field_counts
    except Exception as e:
        logging.error(f"Graph generation failed: {str(e)}")
        return {}

def process_news():
    """Main news processing function with enhanced error handling"""
    try:
        current_time = datetime.now()
        logging.info(f"Starting news update: {current_time.strftime('%Y-%m-%d %I:%M:%S %p')}")
        print(f"\n--- Starting news update: {current_time.strftime('%Y-%m-%d %I:%M:%S %p')} ---")

        # Step 1: Data Collection
        newsapi_data = fetch_newsapi_data()
        scraped_data = scrape_urls(urls)
        
        combined_data = scraped_data + ([newsapi_data] if newsapi_data else [])
        if not combined_data:
            logging.warning("No data collected")
            return

        # Step 2: Text Processing
        all_text = ' '.join(combined_data)
        sentences = sent_tokenize(all_text)
        if len(sentences) < 10:
            logging.warning("Not enough sentences")
            return

        # Step 3: Clustering
        n_clusters = min(20, len(sentences) // 10)
        logging.info(f"Clustering {len(sentences)} sentences into {n_clusters} clusters")
        clusters = cluster_sentences(sentences, n_clusters=n_clusters)
        
        # Step 4: Topic Processing
        parsed_data = []
        for i, cluster in enumerate(clusters, 1):
            meaningful_sentences = select_meaningful_sentences(cluster)
            if not meaningful_sentences or meaningful_sentences[0].startswith("No meaningful"):
                logging.info(f"Skipping topic {i}: No meaningful sentences")
                continue

            topic_name = generate_topic_name(meaningful_sentences)
            field = classify_topic(topic_name)

            content = ' '.join(meaningful_sentences)
            entities = extract_entities(content)
            locations = extract_locations(content)
            key_phrases = [phrase for phrase in extract_key_phrases(content, n=10) 
                          if phrase.lower() not in ["rights", "reserved"]]
            
            if not entities and not key_phrases:
                logging.info(f"Skipping topic {i}: No entities or key phrases")
                continue

            headline_points = generate_headline_points(cluster)
            insight = generate_topic_insight(meaningful_sentences)

            parsed_data.append({
                "Timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                "Topic Number": i,
                "Topic Name": topic_name,
                "Field": field,
                "Generated Headline Points": headline_points,
                "Insight": insight,
                "Key Entities": entities,
                "Locations": locations,
                "Key Phrases": key_phrases,
                "Sample Sentences": meaningful_sentences
            })

        # Step 5: Output
        if parsed_data:
            write_to_json(parsed_data)
            generate_graphs(parsed_data)
        
        logging.info("News update completed successfully")
        return True
    except Exception as e:
        logging.error(f"Error in process_news: {str(e)}")
        print(f"ERROR: {str(e)}")
        return False

def run_continuously():
    """Continuous runner with improved scheduling"""
    while True:
        try:
            success = process_news()
            if success:
                logging.info("Waiting 15 minutes for next update")
                time.sleep(900)  # 15 minutes
            else:
                logging.warning("Last update failed, retrying in 5 minutes")
                time.sleep(300)  # 5 minutes on failure
        except KeyboardInterrupt:
            logging.info("Received keyboard interrupt, shutting down")
            break
        except Exception as e:
            logging.error(f"Unexpected error in run_continuously: {str(e)}")
            time.sleep(60)  # Wait 1 minute before retrying

if __name__ == "__main__":
    logging.info("Real-time News Updater started")
    print("Real-time News Updater started")
    run_continuously()