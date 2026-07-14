"""
Train a byte-level BPE tokenizer from scratch on the medical corpus.
This runs 100% locally -- no download from any model hub required.
"""
import json
import os
from tokenizers import ByteLevelBPETokenizer

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "medquad_qa.jsonl")
OUT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
VOCAB_SIZE = 8000

def main():
    corpus_path = os.path.join(OUT_DIR, "_corpus.txt")
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(DATA_PATH) as f, open(corpus_path, "w") as out:
        for line in f:
            r = json.loads(line)
            out.write(r["question"] + "\n")
            out.write(r["answer"] + "\n")

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=[corpus_path],
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        special_tokens=["<pad>", "<s>", "</s>", "<unk>", "<mask>"],
    )
    tokenizer.save_model(OUT_DIR)
    os.remove(corpus_path)
    print(f"Tokenizer trained and saved to {OUT_DIR} (vocab_size={VOCAB_SIZE})")

if __name__ == "__main__":
    main()
