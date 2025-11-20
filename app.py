import json
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import streamlit as st
from openai import OpenAI, RateLimitError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

# -------------------------------
# Load .env file
# -------------------------------
load_dotenv()


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


# -------------------------------
# OpenAI Client (via .env)
# -------------------------------
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) if api_key else None


# -------------------------------
# LLM: Refine FAQ Answer
# -------------------------------
def refine_with_llm(client: OpenAI, user_query: str, base_answer: str) -> str:
    prompt = f"""
    You are a helpful customer support assistant.

    User asked: "{user_query}"
    FAQ system returned this answer: "{base_answer}"

    Improve this answer in a polite, friendly, and clear manner.
    If the FAQ answer is too short or incomplete, expand it.
    Respond in simple English.
    """

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
        )
        return completion.choices[0].message.content.strip()

    except RateLimitError:
        return base_answer + "\n\n_(AI refinement temporarily unavailable: quota exceeded.)_"

    except Exception:
        return base_answer


# -------------------------------
# LLM: Fallback Answer
# -------------------------------
def llm_fallback_answer(client: OpenAI, user_query: str) -> str:
    prompt = f"""
    You are a customer support assistant. The user asked: "{user_query}"

    No matching FAQ exists. Provide a useful, short, friendly answer.
    If the question requires account-specific or order-specific info,
    politely ask the user to contact human support.
    """

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
        )
        return completion.choices[0].message.content.strip()

    except RateLimitError:
        return (
            "AI assistance is temporarily unavailable due to usage limits.\n"
            "Please contact our human support team at support@example.com or +91-9876543210."
        )

    except Exception:
        return (
            "I ran into a technical issue.\n"
            "Please contact our human support team at support@example.com or +91-9876543210."
        )


# -------------------------------
# Main Bot Logic
# -------------------------------
def generate_bot_reply(query, faq_engine, client, use_llm):
    query = query.strip()

    if not query:
        return "Please type a message 🙂"

    # Greetings
    if query.lower() in ["hi", "hello", "hey"]:
        return "Hi there! 👋 How can I help you today?"

    # Search FAQ
    best_faq, score = faq_engine.search(query)

    # If no FAQ match
    if best_faq is None:
        if client and use_llm:
            return llm_fallback_answer(client, query)
        else:
            return (
                "I couldn't find an exact answer.\n"
                "Please contact our human support team at support@example.com or +91-9876543210."
            )

    # FAQ match found
    base_answer = best_faq.answer

    # Without LLM
    if not use_llm or client is None:
        return (
            f"**Q:** {best_faq.question}\n"
            f"**A:** {base_answer}\n\n"
            f"_Match confidence: {score:.2f}_"
        )

    # With LLM refinement
    refined = refine_with_llm(client, query, base_answer)
    return (
        f"**Q:** {best_faq.question}\n"
        f"**A:** {refined}\n\n"
        f"_Match confidence: {score:.2f}_"
    )


# -------------------------------
# Streamlit UI
# -------------------------------
def main():
    st.set_page_config(page_title="Intelligent Customer Support Bot", page_icon="🤖")

    st.title("🤖 Intelligent Customer Support Bot")
    st.write("Ask anything about **orders, returns, payments, or shipping**.")

    # Sidebar
    st.sidebar.header("Settings")
    use_llm = st.sidebar.checkbox("Use OpenAI LLM refinement", value=True)
    st.sidebar.info("LLM requires a valid OPENAI_API_KEY in your `.env` file.")

    # Load FAQ engine
    faqs = load_faqs("faqs.json")
    faq_engine = FAQSearchEngine(faqs)

    # Chat history
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "bot", "content": "Hi! 👋 How can I help you today?"}
        ]

    # Show chat messages
    for msg in st.session_state.messages:
        with st.chat_message("assistant" if msg["role"] == "bot" else "user"):
            st.markdown(msg["content"])

    # User Input
    user_input = st.chat_input("Type your message...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        reply = generate_bot_reply(user_input, faq_engine, client, use_llm)
        st.session_state.messages.append({"role": "bot", "content": reply})
        st.rerun()


if __name__ == "__main__":
    main()
