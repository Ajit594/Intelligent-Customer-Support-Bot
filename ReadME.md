# Intelligent Customer Support Bot

AI-powered FAQ assistant using TF-IDF + Cosine Similarity with optional LLM refinement. Built with Python and Streamlit for a fast, modular, production-ready support chatbot.

## Features
- Fast FAQ retrieval using TF-IDF + Cosine similarity
- Optional LLM refinement (OpenAI, Gemini, Groq, Hugging Face)
- Streamlit chat UI for an interactive experience
- Robust fallback logic when no FAQ match or LLM fails
- Clean, modular code and secure .env support
- Ready for deployment (Streamlit Cloud, Render, etc.)

## Quick Start

1. Clone
```bash
git clone https://github.com/Ajit594/Intelligent-Customer-Support-Bot.git
cd intelligent-support-bot
```

2. (Optional) Create a venv
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

3. Install
```bash
pip install -r requirements.txt
```

4. Add environment variables
Create a `.env` file in the project root:
```
OPENAI_API_KEY=your_api_key_here
```
Do NOT commit `.env`. Add to `.gitignore`.

5. Run
```bash
streamlit run app.py
```
App opens at http://localhost:8501

## Usage Flow
1. User enters a query.  
2. Convert FAQs and query to TF-IDF vectors.  
3. Compute cosine similarity and pick best FAQ.  
4. If similarity ≥ threshold → return FAQ answer.  
5. If LLM enabled → optionally refine the answer.  
6. If no FAQ match → generate fallback via LLM (or safe local fallback if LLM fails).

## Example Queries
- How can I track my order?
- What is your return policy?
- Do you ship internationally?
- My payment failed, what should I do?

## LLM Providers
Default: OpenAI GPT-4o-mini. Swap easily to:
- Groq (LLaMA-3)
- Google Gemini
- Hugging Face Inference API
Ask for provider-specific integration examples.

## Project Structure
intelligent-support-bot/
- app.py
- faqs.json
- requirements.txt
- .env (local only)
- assets/ (images/screenshots)
- README.md

## Example faqs.json
```json
[
  {
    "id": 1,
    "question": "How can I track my order?",
    "answer": "We send a tracking link once your order ships...",
    "category": "orders"
  }
]
```

## Error Handling
- Graceful handling for: no FAQ match, LLM quota errors, invalid API keys, network issues, missing `.env`.
- Safe user-facing messages; the app won't crash.


## Contributing
PRs and issues welcome. Suggestions for new domains (banking, travel, healthcare, SaaS, EdTech) are encouraged.

## License
MIT License — see LICENSE file.
