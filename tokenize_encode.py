import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

def tokenize_and_encode(input_filename='preprocessed_comments.csv', max_length=128, test_size=0.2, random_state=42):
    # Load preprocessed comments
    data = pd.read_csv(input_filename)
    texts = data['comment'].tolist()
    
    # Ensure that texts are strings
    texts = [str(text) for text in texts]
    
    # Extract sentiment labels
    labels = data['sentiment'].tolist()
    
    # Map sentiment labels to integers
    label_map = {"positive": 0, "neutral": 1, "negative": 2}
    labels = [label_map[label] for label in labels]
    
    # Split dataset into train and test sets
    train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=test_size, random_state=random_state)
    
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Tokenize and encode the train and test texts
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length, return_tensors='pt')
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=max_length, return_tensors='pt')
    
    return train_encodings, test_encodings, train_labels, test_labels

if __name__ == "__main__":
    train_encodings, test_encodings, train_labels, test_labels = tokenize_and_encode()
    print("Tokenization and encoding completed successfully.")
