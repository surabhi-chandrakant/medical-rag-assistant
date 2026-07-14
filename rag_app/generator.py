"""
Generation step of the RAG pipeline, using the Groq API free tier
(OpenAI-compatible endpoint, no cost, no credit card required).

Get a free key at https://console.groq.com -> API Keys, then set:
    export GROQ_API_KEY="gsk_..."
or put it in a .env file (see .env.example).
"""
import os
import requests

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# llama-3.1-8b-instant: fast, generous free-tier limits, good default.
# llama-3.3-70b-versatile: higher quality, lower free-tier request cap.
DEFAULT_MODEL = "llama-3.1-8b-instant"

SYSTEM_PROMPT = (
    "You are a medical information assistant. Answer the user's question "
    "using ONLY the provided context passages, which come from NIH/MedlinePlus "
    "source documents. If the context does not contain the answer, say so "
    "plainly instead of guessing. Cite which passage number you used for each "
    "claim, like [1], [2]. Keep the answer clear and avoid unnecessary jargon. "
    "Always end your answer with a short reminder that this is general "
    "information, not a substitute for professional medical advice."
)


def build_context_block(passages):
    lines = []
    for i, p in enumerate(passages, start=1):
        lines.append(f"[{i}] (Topic: {p['focus']}) {p['answer']}")
    return "\n\n".join(lines)


def generate_answer(query, passages, model=DEFAULT_MODEL, api_key=None, temperature=0.2):
    api_key = api_key or os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Get a free key at https://console.groq.com "
            "and export it as an environment variable, or pass api_key= directly."
        )

    context_block = build_context_block(passages)
    user_prompt = (
        f"Context passages:\n\n{context_block}\n\n"
        f"Question: {query}\n\n"
        "Answer using only the context above."
    )

    resp = requests.post(
        GROQ_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_completion_tokens": 700,
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]
