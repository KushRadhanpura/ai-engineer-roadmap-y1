import argparse
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import os

# Get the directory of the current script to build absolute paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
default_model_path = os.path.join(project_dir, 'models', 'fine-tuned-distilbert')

def predict(text, model_path=default_model_path):
    """
    Loads the fine-tuned model and makes a prediction on the given text.
    """
    # Load the tokenizer and model
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)

    # Tokenize the input text
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)

    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = torch.argmax(logits, dim=1).item()

    return predicted_class_id

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict with a fine-tuned model.")
    parser.add_argument("--text", type=str, required=True, help="Text to classify.")
    parser.add_argument("--model_path", type=str, default=default_model_path, help="Path to the fine-tuned model.")
    args = parser.parse_args()

    prediction = predict(args.text, args.model_path)
    print(f"Text: '{args.text}'")
    print(f"Predicted class: {prediction}")
