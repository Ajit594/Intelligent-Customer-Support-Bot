import pytest
from backend import FAQItem, FAQSearchEngine

@pytest.fixture
def sample_faqs():
    return [
        FAQItem(id=1, question="What is your return policy?", answer="30 days return.", category="returns"),
        FAQItem(id=2, question="How can I track my order?", answer="Use the tracking link.", category="orders"),
    ]

def test_search_exact_match(sample_faqs):
    engine = FAQSearchEngine(sample_faqs)
    result, score = engine.search("What is your return policy?")
    assert result is not None
    assert result.id == 1
    assert score > 0.9

def test_search_relevant_match(sample_faqs):
    engine = FAQSearchEngine(sample_faqs)
    result, score = engine.search("return policy")
    assert result is not None
    assert result.id == 1
    assert score > 0.5

def test_search_no_match(sample_faqs):
    engine = FAQSearchEngine(sample_faqs)
    result, score = engine.search("random gibberish xyz")
    assert result is None
    assert score < engine.threshold

def test_empty_query(sample_faqs):
    engine = FAQSearchEngine(sample_faqs)
    result, score = engine.search("")
    assert result is None
    assert score == 0.0
