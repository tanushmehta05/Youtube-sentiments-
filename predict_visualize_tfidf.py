


import pandas as pd
import joblib
import preprocess
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

def predict_and_visualize(input_filename='comments.csv', model_filename='tfidf_model.pkl'):
    # Load the model and vectorizer
    model, vectorizer = joblib.load(model_filename)

    # Preprocess the comments
    preprocess.preprocess_comments(input_filename='comments.csv', output_filename='preprocessed_comments.csv')
    
    # Load preprocessed comments
    data = pd.read_csv('preprocessed_comments.csv')
    
    # Remove rows with NaN comments
    data = data.dropna(subset=['comment'])

    texts = data['comment'].tolist()
    labels = data['sentiment'].tolist()

    # Map sentiment labels to integers
    label_map = {"positive": 0, "neutral": 1, "negative": 2}
    labels = [label_map[label] for label in labels]

    # Transform the texts using the trained vectorizer
    features = vectorizer.transform(texts)

    # Make predictions
    predictions = model.predict(features)
    
    # Reverse map integer labels to original labels
    reverse_label_map = {0: "positive", 1: "neutral", 2: "negative"}
    predicted_labels = [reverse_label_map[pred] for pred in predictions]

    # Print evaluation results
    report = classification_report(labels, predictions, target_names=['positive', 'neutral', 'negative'])
    print("Evaluation results:\n", report)

    # Visualize results
    plt.figure(figsize=(10, 6))
    pd.Series(predicted_labels).value_counts().plot(kind='bar')
    plt.title('Sentiment Analysis Results')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.show()

if __name__ == "__main__":
    predict_and_visualize()
