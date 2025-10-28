# 07-NLP_Transformer_QA

This project is a technical Question-Answering system that can find answers in a body of text. It's built using the Hugging Face `transformers` library, leveraging a pre-trained BERT model fine-tuned for question answering.

## How it works

The system takes a context (a body of text) and a question as input, and it returns the most likely answer from the context.

## How to run

1. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the script:
   ```
   python qa_system.py
   ```

## CI/CD Pipeline: MLOps Automation

This project is configured with a full CI/CD (Continuous Integration/Continuous Delivery) pipeline using GitHub Actions to automate testing and packaging.

### How It Works

The pipeline is defined in the `.github/workflows/nlp_ci_cd.yml` file and consists of two main **jobs**:

1.  **`build-and-test` (The "CI" Job):**
    *   **What it is:** This is the Continuous Integration job. It acts as a quality gate.
    *   **How it works:** Every time you push a new commit, this job automatically starts. It creates a fresh, clean environment, installs all the project dependencies (like `torch` and `pytest`), and then runs the test suite located in the `tests/` directory.
    *   **Why it helps:** This guarantees that no new code breaks the existing functionality. If a test fails, the entire pipeline stops, preventing a buggy version from being published.

2.  **`build-and-push-docker` (The "CD" Job):**
    *   **What it is:** This is the Continuous Delivery job. It prepares the application for deployment.
    *   **How it works:** This job only runs if the `build-and-test` job succeeds. It takes the application and packages it into a standardized **Docker image**—a lightweight, portable container with everything the app needs to run. It then pushes this image to the **GitHub Container Registry**.
    *   **Why it helps:** This creates a reliable, version-controlled "release" of your application. The resulting Docker image can be deployed anywhere (on a server, in the cloud, etc.) with a single command, eliminating "it works on my machine" problems.

### Pipeline Summary

A `git push` to the `main` branch now results in a new, tested Docker image being automatically built and published, ready for deployment. You can view the published images in the **"Packages"** section of the main repository page.