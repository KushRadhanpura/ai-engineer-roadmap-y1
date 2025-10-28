import sys
import os
import pytest
from unittest.mock import patch

# Add the parent directory to the path so we can import qa_system
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from qa_system import create_qa_pipeline, answer_question

@pytest.fixture(scope="module")
def qa_pipeline():
    """
    Fixture to initialize the QA pipeline once per module.
    """
    return create_qa_pipeline()

@pytest.fixture
def context():
    return "The quick brown fox jumps over the lazy dog."
@pytest.fixture
def question():
    return "What does the fox jump over?"

def test_answer_question(qa_pipeline, context, question):
    """
    Test the answer_question function with a simple context and question.
    """
    answer = answer_question(qa_pipeline, context, question)
    assert isinstance(answer, str)
    assert answer.lower() == "the lazy dog"
