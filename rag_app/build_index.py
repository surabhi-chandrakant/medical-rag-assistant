"""
Build the retrieval index for the RAG app.
Embeds every (question, answer) passage from the medical corpus with our
fine-tuned embedding model and stores the vectors + metadata on disk.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "embedding_model"))
from infer import MedicalEmbedder  # noqa: E402

HERE = os.path.dirname(__file__)
DATA_PATH = os.path.join(HERE, "..", "data", "medquad_qa.jsonl")
INDEX_DIR = os.path.join(HERE, "index")


def main():
    os.makedirs(INDEX_DIR, exist_ok=True)

    records = []
    with open(DATA_PATH) as f:
        for line in f:
            records.append(json.loads(line))

    print(f"Indexing {len(records)} passages...")

    embedder = MedicalEmbedder()

    # We embed the ANSWER text (the actual knowledge passage). The model
    # was trained so that questions and their matching answers land close
    # together in vector space, so embedding answers lets us retrieve them
    # from a natural-language user question at query time.
    passages = [r["answer"] for r in records]

    batch_size = 128
    vecs = []
    for i in range(0, len(passages), batch_size):
        batch = passages[i : i + batch_size]
        vecs.append(embedder.embed(batch))
        if (i // batch_size) % 20 == 0:
            print(f"  {i}/{len(passages)}")
    vecs = np.concatenate(vecs, axis=0).astype("float32")

    np.save(os.path.join(INDEX_DIR, "embeddings.npy"), vecs)
    with open(os.path.join(INDEX_DIR, "metadata.jsonl"), "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print("Index saved:", vecs.shape, "->", INDEX_DIR)


if __name__ == "__main__":
    main()
