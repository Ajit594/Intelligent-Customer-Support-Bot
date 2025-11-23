import json
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# -------------------------------
# Data Model
# -------------------------------
@dataclass
class FAQItem:
    id: int
    question: str
    answer: str
    category: str


# -------------------------------
# Load FAQ Data
# -------------------------------
@st.cache_data
def load_faqs(path: str = "faqs.json") -> List[FAQItem]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    faqs = []
    for item in data:
        faqs.append(
            FAQItem(
                id=item["id"],
                question=item["question"],
                answer=item["answer"],
                category=item["category"],
            )
        )
    return faqs


# -------------------------------
# FAQ Search Engine (TF-IDF)
# -------------------------------
class FAQSearchEngine:
    def __init__(self, faqs: List[FAQItem], threshold: float = 0.35):
        self.faqs = faqs
        self.threshold = threshold
        self.questions = [faq.question for faq in faqs]

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_df=0.9
        )
        self.question_vectors = self.vectorizer.fit_transform(self.questions)

    def search(self, query: str) -> Tuple[Optional[FAQItem], float]:
        if not query.strip():
            return None, 0.0

        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.question_vectors)[0]
        idx = int(np.argmax(sims))
        best_score = float(sims[idx])

        if best_score < self.threshold:
            return None, best_score

        return self.faqs[idx], best_score
