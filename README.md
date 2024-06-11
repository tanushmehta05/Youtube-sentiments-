# YouTube Sentiment Analysis with NLP
-------------------------------------

    Overview
        This project aims to perform sentiment analysis on YouTube comments using Natural Language Processing (NLP) techniques. The goal is to analyze the sentiment expressed in the comments and classify them as positive, negative, or neutral.
    
    Features
        -Tokenization and encoding of comments using BERT tokenizer.
        -Training a BERT-based model for sentiment classification.
        -Evaluating the model's performance using various metrics.
        -Deployment of the model for real-time sentiment analysis on new comments

    Requirements
        -Python 3.6 or higher
        -Libraries: pandas, nltk, torch, transformers

    Usage
        1>Preprocess the comments:
            -Place the YouTube comments dataset in the project directory.
            -Run the preprocess_comments.py script to clean and preprocess the comments.
        2>Tokenize and encode the comments:
            -Run the tokenize_encode.py script to tokenize and encode the preprocessed comments using the BERT tokenizer.
        3>Train the sentiment classification model:
            -Run the train_model.py script to train the BERT-based model for sentiment classification.
        4>Evaluate the model:
            -Assess the model's performance using various evaluation metrics by running the evaluate_model.py script.
        5>Deployment:
            -Deploy the trained model for real-time sentiment analysis on new YouTube comments.

Contributors
-Tanush Mehta

License
    This project is licensed under the MIT License.

Acknowledgments
    The project utilizes the Hugging Face Transformers library for BERT-based NLP tasks.
    Special thanks to Hugging Face for providing pre-trained BERT models