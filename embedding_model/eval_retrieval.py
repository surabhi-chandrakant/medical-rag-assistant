"""
Quick sanity check: for a held-out sample of questions, does the model
retrieve the CORRECT matching answer out of a large pool of candidates?
Reports top-1 and top-5 accuracy -- a standard retrieval quality metric.
"""
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from infer import MedicalEmbedder  # noqa: E402

HERE = os.path.dirname(__file__)
DATA_PATH = os.path.join(HERE, "..", "data", "medquad_qa.jsonl")


def main(n_eval=300, pool_size=2000, seed=42):
    random.seed(seed)
    records = []
    with open(DATA_PATH) as f:
        for line in f:
            records.append(json.loads(line))

    pool = random.sample(records, min(pool_size, len(records)))
    eval_set = random.sample(pool, min(n_eval, len(pool)))

    embedder = MedicalEmbedder()

    print(f"Embedding pool of {len(pool)} answers...")
    answers = [r["answer"] for r in pool]
    answer_vecs = embedder.embed(answers, batch_size=128)
    answer_vecs = answer_vecs / (np.linalg.norm(answer_vecs, axis=1, keepdims=True) + 1e-8)

    print(f"Embedding {len(eval_set)} eval questions...")
    questions = [r["question"] for r in eval_set]
    q_vecs = embedder.embed(questions, batch_size=128)
    q_vecs = q_vecs / (np.linalg.norm(q_vecs, axis=1, keepdims=True) + 1e-8)

    # map each eval question to the index of its true answer in the pool
    pool_answer_to_idx = {}
    for i, r in enumerate(pool):
        pool_answer_to_idx.setdefault(r["answer"], []).append(i)

    top1, top5 = 0, 0
    for i, r in enumerate(eval_set):
        sims = answer_vecs @ q_vecs[i]
        ranked = np.argsort(-sims)
        true_idxs = set(pool_answer_to_idx[r["answer"]])
        if ranked[0] in true_idxs:
            top1 += 1
        if any(idx in true_idxs for idx in ranked[:5]):
            top5 += 1

    print(f"\nTop-1 accuracy: {top1/len(eval_set):.3f}")
    print(f"Top-5 accuracy: {top5/len(eval_set):.3f}")
    print(f"(random baseline top-1 ~= {1/len(pool):.5f}, top-5 ~= {5/len(pool):.5f})")


if __name__ == "__main__":
    main()
