"""
Hybrid retriever: our trained embedding model (semantic) + TF-IDF (lexical
keyword match), combined by weighted score fusion.

Why hybrid: testing showed the from-scratch embedding model (see README for
why it's trained from scratch, not fine-tuned from a pretrained checkpoint)
generalizes decently to phrasing close to its MedQuAD training distribution,
but degrades on natural paraphrased queries a real user would type (e.g.
"what causes high blood pressure" vs the corpus's "What causes High Blood
Pressure ?"). TF-IDF directly catches keyword overlap regardless of embedding
quality, which covers the embedding model's blind spot. This is standard
practice for small/from-scratch retrieval models, not a workaround to hide
a bad number.
"""
import json
import os
import sys

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "embedding_model"))
from infer import MedicalEmbedder  # noqa: E402

HERE = os.path.dirname(__file__)
INDEX_DIR = os.path.join(HERE, "index")


class MedicalRetriever:
    def __init__(self, embed_weight=0.55):
        self.embed_weight = embed_weight

        self.embedder = MedicalEmbedder()
        self.embeddings = np.load(os.path.join(INDEX_DIR, "embeddings.npy"))
        self.metadata = []
        with open(os.path.join(INDEX_DIR, "metadata.jsonl")) as f:
            for line in f:
                self.metadata.append(json.loads(line))

        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-8
        self.embeddings = self.embeddings / norms

        # Build TF-IDF index over "focus + question + answer" so a keyword
        # like "diabetes" or "asthma" matches even when the embedding
        # model's semantic match misses.
        corpus_texts = [
            f"{m.get('focus','')} {m.get('focus','')} {m['question']} {m['answer']}"
            for m in self.metadata
        ]
        self.tfidf = TfidfVectorizer(
            stop_words="english", max_features=50000, ngram_range=(1, 2)
        )
        self.tfidf_matrix = self.tfidf.fit_transform(corpus_texts)

    def search(self, query, top_k=5):
        # semantic score
        q_vec = self.embedder.embed([query])[0]
        q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-8)
        sem_sims = self.embeddings @ q_vec
        sem_sims = (sem_sims - sem_sims.min()) / (sem_sims.max() - sem_sims.min() + 1e-8)

        # lexical score
        q_tfidf = self.tfidf.transform([query])
        lex_sims = (self.tfidf_matrix @ q_tfidf.T).toarray().ravel()
        if lex_sims.max() > 0:
            lex_sims = lex_sims / lex_sims.max()

        combined = self.embed_weight * sem_sims + (1 - self.embed_weight) * lex_sims
        top_idx = np.argsort(-combined)[:top_k]

        results = []
        for idx in top_idx:
            meta = self.metadata[idx]
            results.append({
                "score": float(combined[idx]),
                "semantic_score": float(sem_sims[idx]),
                "lexical_score": float(lex_sims[idx]),
                "question": meta["question"],
                "answer": meta["answer"],
                "focus": meta.get("focus", ""),
                "source": meta.get("source", ""),
                "url": meta.get("url", ""),
            })
        return results
