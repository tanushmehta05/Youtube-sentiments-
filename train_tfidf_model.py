'''USING TFIDF VECTOR WAY INSTEAD OF BERT CLASSFIER'''

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import preprocess

def train_tfidf_model(input_filename='comments.csv', output_filename='tfidf_model.pkl'):
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

    # Split dataset into train and test sets
    train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

    # Initialize TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(max_features=5000)
    
    # Fit and transform the train texts and transform the test texts
    train_features = vectorizer.fit_transform(train_texts)
    test_features = vectorizer.transform(test_texts)

    # Initialize and train the Logistic Regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(train_features, train_labels)

    # Evaluate the model
    predictions = model.predict(test_features)
    report = classification_report(test_labels, predictions, target_names=['positive', 'neutral', 'negative'])
    print("Evaluation results:\n", report)

    # Save the trained model and vectorizer
    import joblib
    joblib.dump((model, vectorizer), output_filename)

if __name__ == "__main__":
    train_tfidf_model()
