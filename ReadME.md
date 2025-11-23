# Intelligent Customer Support Bot

A small, practical customer-support chatbot that combines a TF-IDF FAQ retriever with optional LLM refinement. Designed for local use and simple deployment via Streamlit.

## Features
- TF-IDF + cosine-similarity FAQ retrieval (fast and offline)
- Optional LLM refinement and fallback (OpenAI-compatible endpoints and OpenRouter)
- Streamlit chat UI for testing and demos
- Resilience: retry/backoff for LLM calls and OpenRouter host fallback

---

## Requirements
- Python 3.8+ (verify with `python --version`)
- Install packages from `requirements.txt`:

```powershell
pip install -r requirements.txt
```

## Quick start (Windows / PowerShell)
1. Clone and enter the repo:

```powershell
git clone https://github.com/Ajit594/Intelligent-Customer-Support-Bot.git
cd intelligent-support-bot
```

2. (Optional) Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Install dependencies (again):

```powershell
pip install -r requirements.txt
```

4. Create a `.env` file in the project root with your keys (do NOT commit this file):

```dotenv
# Example for OpenRouter
OPENROUTER_API_KEY=sk-or-...YOUR_KEY...
# Optional: force a particular base URL if your network needs it
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

# Or, for OpenAI-compatible endpoints
OPENAI_API_KEY=sk-...YOUR_KEY...
OPENAI_BASE_URL=https://openrouter.ai/api/v1
```

5. Run the app:

```powershell
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## How it works (high level)
- `backend.py` loads `faqs.json` and vectorizes FAQ questions with TF-IDF.
- On each user query, the app computes cosine similarity to pick the best FAQ.
- If a match passes the threshold, the FAQ answer is returned; if LLM refinement is enabled the app will call the selected provider to polish or expand the answer.
- If no FAQ matches, the app asks the LLM for a short fallback answer.

---

## Environment variables and configuration
- `OPENROUTER_API_KEY` — OpenRouter API key (recommended for free gpt-oss models)
- `OPENROUTER_BASE_URL` — base URL to use for OpenRouter. The app defaults to the primary host, but includes a fallback to `https://openrouter.ai/api/v1` for networks where `api.openrouter.ai` does not resolve.
- `OPENAI_API_KEY` / `OPENAI_BASE_URL` — set when using an OpenAI-compatible provider
- `OPENROUTER_MODEL` / `OPENAI_MODEL_NAME` — optional model name override (defaults are set in `app.py`)

Notes:
- Keep `.env` local and out of version control. If any key was previously committed, rotate it immediately.

---

## Testing OpenRouter connectivity
Use the included `test_openrouter.py` to verify your key and base URL.

Example (PowerShell):

```powershell
# set env var for the current session (optional)
$Env:OPENROUTER_API_KEY = 'sk-or-...'
python test_openrouter.py
```

The script honors `OPENROUTER_BASE_URL` and will try an alternate host if the primary one fails to resolve.

---

## Project layout
- `app.py` — main Streamlit app and provider adapter logic
- `backend.py` — FAQ dataclass and TF-IDF search engine
- `faqs.json` — the FAQ dataset
- `test_openrouter.py` — connectivity test for OpenRouter
- `requirements.txt` — dependency list

---

## Troubleshooting
- DNS/network errors for OpenRouter: set `OPENROUTER_BASE_URL=https://openrouter.ai/api/v1` in your `.env` to force the alternate host.
- Authentication errors: verify the env var names and the key value; rotate if compromised.
- Rate limits / transient failures: the app uses exponential backoff by default. If you see persistent failures, wait and retry or switch providers.

---

## Extending the knowledge base
- Edit `faqs.json` to add, remove, or update FAQ entries. Each item should contain `id`, `question`, `answer`, and optional `category`.
- Tune TF-IDF parameters in `backend.py` (ngram range, stop words, similarity threshold) to improve matching for your content.

---

## Contributing
- PRs and issues welcome. Do not include secrets in pull requests; use `.env.example` for placeholder values.

---

## License
MIT

---


