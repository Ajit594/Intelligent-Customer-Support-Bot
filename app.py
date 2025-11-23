import os
import streamlit as st
from openai import OpenAI, RateLimitError
import requests
import time
import random
from dotenv import load_dotenv
from backend import load_faqs, FAQSearchEngine

# -------------------------------
# Load .env file
# -------------------------------
load_dotenv()


# -------------------------------
# OpenAI / OpenRouter Client configuration (via .env)
# -------------------------------
api_key = os.getenv("OPENAI_API_KEY")
model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
base_url = os.getenv("OPENAI_BASE_URL")
client = OpenAI(api_key=api_key, base_url=base_url) if api_key else None

# OpenRouter settings
openrouter_key = os.getenv("OPENROUTER_API_KEY")
openrouter_base = os.getenv("OPENROUTER_BASE_URL", "https://api.openrouter.ai/v1").rstrip("/")
openrouter_fallback_base = "https://openrouter.ai/api/v1"
openrouter_model = os.getenv("OPENROUTER_MODEL", "gpt-oss-20b")


def refine_with_openrouter(api_key: str, user_query: str, base_answer: str) -> str:
    prompt = f"You are a helpful customer support assistant.\n\nUser asked: \"{user_query}\"\nFAQ system returned this answer: \"{base_answer}\"\n\nImprove this answer in a polite, clear, and helpful manner. Keep the reply concise."

    url = f"{openrouter_base}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    max_retries = 3
    backoff0 = 1.0
    for attempt in range(max_retries):
        try:
            payload = {
                "model": openrouter_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.4,
            }
            resp = requests.post(url, json=payload, headers=headers, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                # Try to extract text from common fields
                text = None
                if isinstance(data, dict):
                    # OpenRouter often returns choices -> [ {message: {content}} ]
                    choices = data.get("choices") or []
                    if choices:
                        msg = choices[0].get("message") or {}
                        text = msg.get("content") or msg.get("text")
                if not text:
                    # fallback to stringifying
                    text = str(data)
                return text.strip()
            elif resp.status_code in (429, 503):
                # retryable
                if attempt == max_retries - 1:
                    break
                sleep = min(backoff0 * (2 ** attempt), 8.0) + random.uniform(0, 0.1)
                time.sleep(sleep)
                continue
            else:
                # non-retryable, return base answer
                break
        except requests.RequestException as e:
            # Try fallback host if DNS/connection error
            if attempt == 0 and openrouter_base != openrouter_fallback_base:
                try:
                    alt_url = f"{openrouter_fallback_base}/chat/completions"
                    resp = requests.post(alt_url, json=payload, headers=headers, timeout=15)
                    if resp.status_code == 200:
                        data = resp.json()
                        text = None
                        if isinstance(data, dict):
                            choices = data.get("choices") or []
                            if choices:
                                msg = choices[0].get("message") or {}
                                text = msg.get("content") or msg.get("text")
                        if not text:
                            text = str(data)
                        return text.strip()
                except Exception:
                    # fallback failed; proceed to backoff/retry logic
                    pass

            if attempt == max_retries - 1:
                break
            sleep = min(backoff0 * (2 ** attempt), 8.0) + random.uniform(0, 0.1)
            time.sleep(sleep)
            continue

    return base_answer + "\n\n_(AI refinement temporarily unavailable.)_"


def openrouter_fallback(api_key: str, user_query: str) -> str:
    prompt = f"You are a customer support assistant. The user asked: \"{user_query}\"\n\nNo matching FAQ was found. Provide a short, helpful customer service style answer. If the question is too specific, ask the user to contact human support."
    url = f"{openrouter_base}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    max_retries = 3
    backoff0 = 1.0
    for attempt in range(max_retries):
        try:
            payload = {"model": openrouter_model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.5}
            resp = requests.post(url, json=payload, headers=headers, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                choices = data.get("choices") or []
                if choices:
                    msg = choices[0].get("message") or {}
                    text = msg.get("content") or msg.get("text") or str(choices[0])
                    return text.strip()
                return str(data)
            elif resp.status_code in (429, 503):
                if attempt == max_retries - 1:
                    break
                sleep = min(backoff0 * (2 ** attempt), 8.0) + random.uniform(0, 0.1)
                time.sleep(sleep)
                continue
            else:
                break
        except requests.RequestException:
            # Try fallback host on first connection error
            if attempt == 0 and openrouter_base != openrouter_fallback_base:
                try:
                    alt_url = f"{openrouter_fallback_base}/chat/completions"
                    resp = requests.post(alt_url, json=payload, headers=headers, timeout=15)
                    if resp.status_code == 200:
                        data = resp.json()
                        choices = data.get("choices") or []
                        if choices:
                            msg = choices[0].get("message") or {}
                            text = msg.get("content") or msg.get("text") or str(choices[0])
                            return text.strip()
                except Exception:
                    pass

            if attempt == max_retries - 1:
                break
            sleep = min(backoff0 * (2 ** attempt), 8.0) + random.uniform(0, 0.1)
            time.sleep(sleep)
            continue

    return (
        "I’m not able to use the AI assistant right now because of usage limits or an error.\n"
        "Please contact our human support team at support@example.com or +91-9876543210."
    )


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
            model=model_name,
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
            model=model_name,
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
def generate_bot_reply(query, faq_engine, client, use_llm, provider="OpenAI", openrouter_ready=False):
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
        if use_llm:
            if provider == "OpenAI" and client:
                return llm_fallback_answer(client, query)
            if provider == "OpenRouter" and openrouter_ready:
                return openrouter_fallback(openrouter_key, query)
        return (
            "I couldn't find an exact answer.\n"
            "Please contact our human support team at support@example.com or +91-9876543210."
        )

    # FAQ match found
    base_answer = best_faq.answer

    # Without LLM or provider not ready
    if not use_llm:
        return (
            f"**Q:** {best_faq.question}\n"
            f"**A:** {base_answer}\n\n"
            f"_Match confidence: {score:.2f}_"
        )

    # With LLM refinement
    if provider == "OpenAI":
        if client is None:
            return (
                f"**Q:** {best_faq.question}\n"
                f"**A:** {base_answer}\n\n"
                f"_Match confidence: {score:.2f}_"
            )
        refined = refine_with_llm(client, query, base_answer)
    elif provider == "OpenRouter":
        if not openrouter_ready:
            return (
                f"**Q:** {best_faq.question}\n"
                f"**A:** {base_answer}\n\n"
                f"_Match confidence: {score:.2f}_"
            )
        refined = refine_with_openrouter(openrouter_key, query, base_answer)
    else:
        # default fallback to base answer
        refined = base_answer
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
    use_llm = st.sidebar.checkbox("Use LLM refinement", value=True)
    provider = st.sidebar.selectbox("LLM Provider", ["OpenAI", "OpenRouter"]) 
    st.sidebar.info("LLM requires a valid OPENAI_API_KEY or OPENROUTER_API_KEY in your `.env` file.")

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

        # Determine provider readiness
        openrouter_ready = False
        if provider == "OpenRouter":
            openrouter_ready = bool(openrouter_key)

        reply = generate_bot_reply(user_input, faq_engine, client, use_llm, provider=provider, openrouter_ready=openrouter_ready)
        st.session_state.messages.append({"role": "bot", "content": reply})
        st.rerun()


if __name__ == "__main__":
    main()
