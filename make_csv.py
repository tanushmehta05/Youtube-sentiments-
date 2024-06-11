import csv
import re
from googleapiclient.discovery import build
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd

# Replace with your API key
api_key = 'ENTER YOUR API KEY'
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
#Remove URLs,HTML tags,non-alphabetic characters and Convert to lowercase
def preprocess_text(text):
    text = re.sub(r'http\S+', '', text) 
    text = re.sub(r'<.*?>', '', text)  
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()  # 
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Removes stop words
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

video_id = 'KO43_56HN8Q'# enter just the video id in order to download that specific comments from the section and label em 
comments = get_video_comments(video_id)

# Label comments with sentiment i.e positive,negative or neutral
labeled_comments = []
for comment in comments:
    sentiment = analyze_sentiment_vader(comment)
    labeled_comments.append([comment, sentiment])

# Save labeled comments to CSV to labeled_comments.csv in your depositary you can change the name to whatever you want 
with open('labeled_comments.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['comment', 'sentiment'])
    writer.writerows(labeled_comments)

# Load and display the CSV file
df = pd.read_csv('labeled_comments.csv')
print(df.head())
