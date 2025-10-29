# Fine-Tuning a Language Model

This project demonstrates how to fine-tune a pre-trained Transformer model (like DistilBERT) for a specific task. The goal is to create a deployed, fine-tuned language model that solves a specific problem.

## Project Structure

```
08-Fine_Tuning_LLM/
│
├── app/
│   ├── Dockerfile
│   └── main.py
│
├── data/
│   └── .gitkeep
│
├── models/
│   └── .gitkeep
│
├── notebooks/
│   └── .gitkeep
│
├── src/
│   ├── __init__.py
│   ├── fine_tune.py
│   ├── predict.py
│   └── utils.py
│
├── README.md
└── requirements.txt
```

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd 08-Fine_Tuning_LLM
    ```

2.  **Create a virtual environment and install dependencies:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

## Usage

### 1. Data Preparation

Place your training and testing data in the `data/` directory. The `utils.py` script should be updated to load and preprocess your specific dataset.

### 2. Fine-Tuning

Run the fine-tuning script:

```bash
python src/fine_tune.py
```

This will fine-tune the model and save the trained model to the `models/` directory.

### 3. Prediction

To make predictions using the fine-tuned model:

```bash
python src/predict.py --text "Your text for prediction"
```

### 4. API Deployment

To build and run the Docker container for the API:

```bash
cd app
docker build -t fine-tuned-llm .
docker run -p 8000:8000 fine-tuned-llm
```

The API will be available at `http://localhost:8000`.
