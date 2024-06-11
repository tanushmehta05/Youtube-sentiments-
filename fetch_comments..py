# fetch_comments.py
import csv
from googleapiclient.discovery import build
from config import api_key,video_id # Import the API key from config.py

def get_video_comments(video_id, max_results=100):
    youtube = build('youtube', 'v3', developerKey=api_key)
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

def save_comments_to_csv(comments, filename='comments.csv'):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['comment', 'sentiment'])  # Add headers
        for comment in comments:
            writer.writerow([comment, ''])  # Sentiment will be filled later

if __name__ == "__main__":
    comments = get_video_comments(video_id)
    save_comments_to_csv(comments)
