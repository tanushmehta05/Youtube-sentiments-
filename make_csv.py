import csv
import re
from googleapiclient.discovery import build
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd

# Replace with your API key
api_key = 'AIzaSyAcJFnmtjH_cWSY2zWNaYvGLk_gXu3H-js'
youtube = build('youtube', 'v3', developerKey=api_key)

def get_video_comments(video_id, max_results=100):
    comments = []
    response = youtube.commentThreads().list(
        part='snippet',
        videoId=video_id,
        textFormat='plainText',
        maxResults=max_results
    ).execute()
    
    while response:
        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments.append(comment)
        
        if 'nextPageToken' in response:
            response = youtube.commentThreads().list(
                part='snippet',
                videoId=video_id,
                textFormat='plainText',
                pageToken=response['nextPageToken'],
                maxResults=max_results
            ).execute()
        else:
            break
    
    return comments

def preprocess_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabetic characters
    text = text.lower()  # Convert to lowercase
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stop words
    return text

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment_vader(text):
    scores = analyzer.polarity_scores(text)
    if scores['compound'] >= 0.05:
        return 'positive'
    elif scores['compound'] <= -0.05:
        return 'negative'
    else:
        return 'neutral'

# Example video ID
video_id = 'KO43_56HN8Q'
comments = get_video_comments(video_id)

# Label comments with sentiment
labeled_comments = []
for comment in comments:
    sentiment = analyze_sentiment_vader(comment)
    labeled_comments.append([comment, sentiment])

# Save labeled comments to CSV
with open('labeled_comments.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['comment', 'sentiment'])  # Add headers
    writer.writerows(labeled_comments)

# Load and display the CSV file
df = pd.read_csv('labeled_comments.csv')
print(df.head())
