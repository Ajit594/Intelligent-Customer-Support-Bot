# test_openrouter.py
import os, requests, json
key = os.getenv("OPENROUTER_API_KEY")
print("OPENROUTER_API_KEY set:", bool(key))
if not key:
    raise SystemExit("Set OPENROUTER_API_KEY first")

# Use environment-configurable base URL; fall back to openrouter.ai
base = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
url = f"{base}/chat/completions"
payload = {
    "model": os.getenv("OPENROUTER_MODEL", "gpt-oss-20b"),
    "messages": [{"role": "user", "content": "Say hello in one sentence."}],
}

try:
    r = requests.post(url, headers={"Authorization": f"Bearer {key}"}, json=payload, timeout=20)
except requests.exceptions.RequestException as e:
    # If the configured base failed, try the alternative host
    alt = "https://openrouter.ai/api/v1"
    if base != alt:
        print(f"Primary base {base} failed with: {e}; trying fallback {alt}")
        try:
            r = requests.post(f"{alt}/chat/completions", headers={"Authorization": f"Bearer {key}"}, json=payload, timeout=20)
        except Exception as e2:
            print("Fallback also failed:", e2)
            raise
    else:
        print("Request failed:", e)
        raise

print("status:", r.status_code)
try:
    print(json.dumps(r.json(), indent=2)[:2000])
except Exception:
    print(r.text[:2000])