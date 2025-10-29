import argparse
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from utils import load_data
import os

# Get the directory of the current script to build absolute paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
models_dir = os.path.join(project_dir, "models")

def main(model_name='distilbert-base-uncased'):
    """
    Fine-tunes a DistilBERT model on a custom dataset.
    """
    # Load tokenizer and model
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model = DistilBertForSequenceClassification.from_pretrained(model_name)

    # Load and preprocess data (this is a placeholder)
    # You will need to replace this with your actual dataset loading and preprocessing
    # For demonstration, we use a sample dataset from `datasets` library
    def preprocess_function(examples):
        return tokenizer(examples['text'], truncation=True, padding=True)

    # Replace with your dataset
    # train_dataset, val_dataset = load_data('path/to/your/train.csv', 'path/to/your/val.csv')
    dataset = load_dataset('imdb')
    tokenized_datasets = dataset.map(preprocess_function, batched=True)

    # Using a smaller subset of the data to reduce load
    train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(10)) # Reduced to 10
    eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(5)) # Reduced to 5


    # Define training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(models_dir, 'results'),
        num_train_epochs=1,
        per_device_train_batch_size=1,  # Reduced to 1
        per_device_eval_batch_size=1, # Reduced to 1
        warmup_steps=10, # Reduced from 100
        weight_decay=0.01,
        logging_dir=os.path.join(models_dir, 'logs'),
        logging_steps=1,
        eval_strategy="epoch"
    )

    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )

    # Train the model
    trainer.train()

    # Save the fine-tuned model
    model_save_path = os.path.join(models_dir, 'fine-tuned-distilbert')
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"Model fine-tuning complete and saved to '{model_save_path}'")

if __name__ == "__main__":
    main()
