# preprocess.py
import re
import nltk
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

nltk.download('stopwords')
stop_words = set(nltk.corpus.stopwords.words('english'))

def preprocess_text(text):
    if isinstance(text, str):
        text = re.sub(r'http\S+', '', text)  # Remove URLs
        text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabetic characters
        text = text.lower()  # Convert to lowercase
        text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stop words
    else:
        text = ""
    return text

def analyze_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    score = analyzer.polarity_scores(text)
    if score['compound'] >= 0.05:
        return "positive"  # Positive
    elif score['compound'] <= -0.05:
        return "negative"  # Negative
    else:
        return "neutral"  # Neutral

def preprocess_comments(input_filename='comments.csv', output_filename='preprocessed_comments.csv'):
    data = pd.read_csv(input_filename)
    data['comment'] = data['comment'].apply(preprocess_text)
    data['sentiment'] = data['comment'].apply(analyze_sentiment)
    data.to_csv(output_filename, index=False)
    return data

if __name__ == "__main__":
    preprocess_comments()
