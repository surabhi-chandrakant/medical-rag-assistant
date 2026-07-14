"""
Fine-tune / train the medical domain embedding model using contrastive
learning on (question, answer) pairs from MedQuAD. Matching Q-A pairs are
pulled together in embedding space; all other in-batch answers act as
negatives (standard in-batch InfoNCE, the same core idea used to train
SBERT/E5-style retrieval models).
"""
import json
import os
import random
import time

import torch
import torch.nn.functional as F
from tokenizers import ByteLevelBPETokenizer

from model import MedicalEmbeddingModel

HERE = os.path.dirname(__file__)
DATA_PATH = os.path.join(HERE, "..", "data", "medquad_qa.jsonl")
CKPT_DIR = os.path.join(HERE, "checkpoints")

MAX_LEN = 64
BATCH_SIZE = 64
EPOCHS = 3
LR = 3e-4
TEMPERATURE = 0.05
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(os.cpu_count() or 1)


def load_data():
    pairs = []
    with open(DATA_PATH) as f:
        for line in f:
            r = json.loads(line)
            pairs.append((r["question"], r["answer"]))
    return pairs


def encode_batch(tokenizer, texts, max_len):
    ids = []
    for t in texts:
        enc = tokenizer.encode(t)
        toks = enc.ids[: max_len - 2]
        toks = [1] + toks + [2]  # <s> ... </s>
        pad_len = max_len - len(toks)
        toks = toks + [0] * pad_len  # <pad> = 0
        ids.append(toks)
    return torch.tensor(ids, dtype=torch.long)


def info_nce_loss(q_emb, a_emb, temperature):
    logits = q_emb @ a_emb.T / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    loss_q = F.cross_entropy(logits, labels)
    loss_a = F.cross_entropy(logits.T, labels)
    return (loss_q + loss_a) / 2


def main():
    print("Device:", DEVICE)
    tokenizer = ByteLevelBPETokenizer(
        os.path.join(CKPT_DIR, "vocab.json"),
        os.path.join(CKPT_DIR, "merges.txt"),
    )
    vocab_size = tokenizer.get_vocab_size()
    print("Vocab size:", vocab_size)

    pairs = load_data()
    print("Training pairs:", len(pairs))

    model = MedicalEmbeddingModel(
        vocab_size=vocab_size, dim=160, n_heads=4, n_layers=3,
        ff_dim=384, max_len=MAX_LEN,
    ).to(DEVICE)
    optim = torch.optim.AdamW(model.parameters(), lr=LR)
    n_params = sum(p.numel() for p in model.parameters())
    print("Model params:", n_params)

    steps_per_epoch = len(pairs) // BATCH_SIZE
    total_steps = steps_per_epoch * EPOCHS
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optim, max_lr=LR, total_steps=max(total_steps, 1)
    )

    model.train()
    t0 = time.time()
    step = 0
    for epoch in range(EPOCHS):
        random.shuffle(pairs)
        epoch_loss = 0.0
        for i in range(0, len(pairs) - BATCH_SIZE + 1, BATCH_SIZE):
            batch = pairs[i : i + BATCH_SIZE]
            questions = [b[0] for b in batch]
            answers = [b[1] for b in batch]

            q_ids = encode_batch(tokenizer, questions, MAX_LEN).to(DEVICE)
            a_ids = encode_batch(tokenizer, answers, MAX_LEN).to(DEVICE)

            q_emb = model(q_ids)
            a_emb = model(a_ids)

            loss = info_nce_loss(q_emb, a_emb, TEMPERATURE)

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            scheduler.step()

            epoch_loss += loss.item()
            step += 1

        avg_loss = epoch_loss / steps_per_epoch
        elapsed = time.time() - t0
        print(f"Epoch {epoch+1}/{EPOCHS}  avg_loss={avg_loss:.4f}  elapsed={elapsed:.1f}s")

        torch.save(
            {
                "state_dict": model.state_dict(),
                "vocab_size": vocab_size,
                "max_len": MAX_LEN,
                "dim": 160,
                "n_heads": 4,
                "n_layers": 3,
                "ff_dim": 384,
                "epoch": epoch + 1,
            },
            os.path.join(CKPT_DIR, "medical_embedding_model.pt"),
        )
        print(f"  checkpoint saved (epoch {epoch+1})")

    torch.save(
        {
            "state_dict": model.state_dict(),
            "vocab_size": vocab_size,
            "max_len": MAX_LEN,
            "dim": 160,
            "n_heads": 4,
            "n_layers": 3,
            "ff_dim": 384,
        },
        os.path.join(CKPT_DIR, "medical_embedding_model.pt"),
    )
    print("Saved model to", os.path.join(CKPT_DIR, "medical_embedding_model.pt"))


if __name__ == "__main__":
    main()
