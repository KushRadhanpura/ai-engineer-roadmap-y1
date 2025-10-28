from transformers import pipeline

def create_qa_pipeline():
    """
    Initializes and returns a question-answering pipeline from Hugging Face.
    """
    qa_pipeline = pipeline(
        "question-answering",
        model="distilbert-base-cased-distilled-squad",
        tokenizer="distilbert-base-cased-distilled-squad"
    )
    return qa_pipeline

def answer_question(qa_pipeline, context, question):
    """
    Takes a context and a question and returns the answer.
    """
    result = qa_pipeline(question=question, context=context)
    return result['answer']

if __name__ == "__main__":
    # Example usage
    context = """
    The Hugging Face Transformers library provides a unified API for a wide range of pre-trained models 
    for Natural Language Understanding (NLU) and Natural Language Generation (NLG). 
    It's a powerful tool for NLP practitioners and researchers.
    """
    question = "What does the Hugging Face Transformers library provide?"

    qa_pipeline = create_qa_pipeline()
    answer = answer_question(qa_pipeline, context, question)

    print(f"Question: {question}")
    print(f"Answer: {answer}")

    question_2 = "What is it a powerful tool for?"
    answer_2 = answer_question(qa_pipeline, context, question_2)
    print(f"Question: {question_2}")
    print(f"Answer: {answer_2}")
