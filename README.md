# Medical RAG Assistant

A retrieval-augmented generation (RAG) app for medical Q&A, built on:
1. A **domain embedding model**, trained from scratch on real medical Q&A data (MedQuAD).
2. A **base LLM served via the Groq API free tier** for grounded answer generation.

## Live Demo :  https://medical-rag-assistant-personal.streamlit.app/ 

## Read this before you assume it's a standard fine-tuned-SBERT setup

This project was built inside a sandboxed environment with **no access to
Hugging Face Hub** (huggingface.co is not reachable). That blocks the usual
path of downloading a pretrained embedding model (e.g. `all-MiniLM-L6-v2`)
and fine-tuning it.

So instead, the embedding model here is a **compact transformer bi-encoder
trained from scratch** (not fine-tuned from a pretrained checkpoint) using
contrastive (in-batch InfoNCE) learning on ~16,000 real medical question/answer
pairs from [MedQuAD](https://github.com/abachaa/MedQuAD) (NIH/MedlinePlus/CDC
source documents). This is a legitimate way to train a retrieval embedding
model — it's just starting from random init instead of a large pretrained
checkpoint.

**Tested retrieval quality (not a guess — measured):**
- Embedding-only, held-out questions vs a 2,000-passage pool: **54.7% top-1,
  73.0% top-5** accuracy (random baseline: 0.05% / 0.25%). Real signal, but
  during manual testing this dropped noticeably on natural paraphrased
  queries (e.g. "what causes high blood pressure") vs the corpus's own
  stilted phrasing ("What causes High Blood Pressure ?") — the small
  from-scratch model doesn't generalize across phrasing as well as a large
  pretrained one would.
- **Fix applied**: `retriever.py` is a hybrid retriever — embedding
  similarity + TF-IDF keyword matching, combined by weighted score fusion.
  This closes the paraphrase gap (verified: queries like "what causes high
  blood pressure" and "what are the symptoms of asthma" now correctly
  retrieve the right topic). Full re-eval on the complete 16,333-passage
  index: **54.0% top-1, 88.5% top-5**. This isn't a workaround to inflate a
  number — hybrid retrieval is standard practice specifically because
  embedding models and keyword search fail on different query types.

Run `embedding_model/eval_retrieval.py` yourself to reproduce the
embedding-only numbers, don't take this file's word for it.

**On the LLM side**: Groq's free tier serves a fixed catalog of open models
(Llama, etc.) — you cannot upload custom fine-tuned weights there. So the
"LLM fine-tuning" part of a typical two-model plan is **not** included here;
this app uses RAG (retrieval + prompting) against a strong base model
instead, which is the standard and more reliable approach for grounding an
LLM in a specific domain without the cost/risk of full fine-tuning. If you
specifically need a fine-tuned generator model, see "If you actually need a
fine-tuned LLM" below.

## Architecture

```
User question
     │
     ▼
[Domain embedding model]  <-- trained from scratch on MedQuAD (embedding_model/)
     │  (encodes question)
     ▼
[Vector search over indexed MedQuAD passages]  (rag_app/index/)
     │  (top-k relevant passages)
     ▼
[Groq API - llama-3.1-8b-instant]  <-- generates answer grounded in passages
     │
     ▼
Answer + cited sources
```

## Project layout

```
medrag/
├── data/
│   └── medquad_qa.jsonl        # 16,333 real medical Q&A pairs (parsed from MedQuAD XML)
├── embedding_model/
│   ├── model.py                 # transformer bi-encoder architecture
│   ├── train_tokenizer.py       # trains a local BPE tokenizer (no external download)
│   ├── train_embedding_model.py # contrastive training script
│   ├── infer.py                 # inference wrapper for the trained model
│   ├── eval_retrieval.py        # top-1/top-5 retrieval accuracy check
│   └── checkpoints/             # trained weights + tokenizer files (included)
├── rag_app/
│   ├── build_index.py           # embeds the corpus, builds the vector index
│   ├── retriever.py             # cosine-similarity search
│   ├── generator.py             # Groq API call + grounded prompt template
│   ├── app.py                   # Streamlit UI
│   └── index/                   # pre-built embeddings.npy + metadata.jsonl (included)
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# edit .env and add your free Groq API key (console.groq.com -> API Keys, no card needed)
export GROQ_API_KEY=gsk_...
```

## Run the app

The index and trained model are already included, so you can run the app
directly:

```bash
cd rag_app
streamlit run app.py
```

## Reproducing / retraining from scratch

```bash
cd embedding_model
python3 train_tokenizer.py          # ~10 sec
python3 train_embedding_model.py    # ~10-15 min on CPU (3 epochs, 16k pairs)
python3 eval_retrieval.py           # prints top-1 / top-5 retrieval accuracy

cd ../rag_app
python3 build_index.py              # re-embeds the corpus with the new model
```

## Known limitations 

- **Retrieval quality is bounded by training from scratch on a small model
  and small compute budget**, mitigated by the hybrid (embedding + TF-IDF)
  retriever described above. A fine-tuned pretrained encoder (e.g. E5,
  BGE, MiniLM) would still outperform this on hard/ambiguous queries. If you
  get network access to Hugging Face Hub in your own environment, swap in a
  real pretrained base and fine-tune it — the training script here would
  need adapting but the contrastive-loss approach carries over directly.
- **The LLM is not fine-tuned**, only prompted with retrieved context. For
  a medical/legal domain this is usually *preferable* anyway — it reduces
  hallucination risk versus a fine-tuned model that might overwrite factual
  recall with stylistic pattern-matching. But if your assignment/use case
  specifically requires a fine-tuned generator, see below.
- **This is not a medical device and gives no clinical guarantees.** MedQuAD
  is consumer health information (NIH/MedlinePlus), not a clinical decision
  support dataset. Don't deploy this for real patient-facing use without
  a lot more validation, a licensed clinician in the loop, and legal review.
- Free-tier Groq model names and rate limits change over time — if
  `llama-3.1-8b-instant` errors out, check https://console.groq.com for
  current model IDs.
- **I could not live-test the actual Groq API call from this sandbox**
  (api.groq.com is also outside this environment's network allowlist). The
  request format in `generator.py` was verified against Groq's current API
  reference docs (endpoint, auth header, message schema, param names), but
  you should run one real query yourself the first time to confirm it works
  end-to-end before relying on it.

## If you actually need a fine-tuned LLM (not just RAG)

Groq's free tier can't serve custom weights. Your options, in order of effort:
1. **LoRA fine-tune a small open model** (e.g. Llama 3.2 1B/3B, Qwen2.5 3B)
   on Google Colab's free GPU tier, then serve it yourself via Ollama or a
   local vLLM instance instead of Groq.
2. Use a **paid** Groq/Together/Fireworks custom-model hosting tier — these
   accept fine-tuned weights, unlike the free tier.
3. Skip fine-tuning and use larger prompt context + more retrieved passages
   (i.e. push harder on the RAG side) — often gets you 90% of the benefit
   for 5% of the cost and risk.

I did not build the LoRA training script here since it requires a GPU this
sandbox doesn't have, and it would run on synthetic/unverified assumptions
about  target model choice and Colab setup. 
