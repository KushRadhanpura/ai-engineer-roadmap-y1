# Dockerfile for NLP QA System
FROM python:3.11-slim

WORKDIR /app

COPY ./07-NLP_Transformer_QA/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY ./07-NLP_Transformer_QA/qa_system.py .

# The CMD is commented out as this image is intended to be a base for running the QA system,
# not to run a service.
# CMD ["python", "qa_system.py"]
