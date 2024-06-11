import torch
from transformers import BertForSequenceClassification, Trainer, TrainingArguments
from preprocess import preprocess_comments
from tokenize_encode import tokenize_and_encode  # Import tokenize_and_encode function

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(int(self.labels[idx]))  # Convert labels to integers
        return item

    def __len__(self):
        return len(self.labels)


def train_model():
    preprocess_comments()  # Preprocess the comments before tokenization and encoding

    train_encodings, test_encodings, train_labels, test_labels = tokenize_and_encode()

    train_dataset = Dataset(train_encodings, train_labels)
    test_dataset = Dataset(test_encodings, test_labels)

    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset
    )

    trainer.train()
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")

if __name__ == "__main__":
    train_model()
