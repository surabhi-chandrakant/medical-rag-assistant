"""
Lightweight inference wrapper around the trained medical embedding model.
No external hub dependency -- loads local checkpoint + local tokenizer files.
"""
import os
import torch
from tokenizers import ByteLevelBPETokenizer

from model import MedicalEmbeddingModel

HERE = os.path.dirname(__file__)
CKPT_DIR = os.path.join(HERE, "checkpoints")


class MedicalEmbedder:
    def __init__(self, ckpt_path=None, device="cpu"):
        ckpt_path = ckpt_path or os.path.join(CKPT_DIR, "medical_embedding_model.pt")
        ckpt = torch.load(ckpt_path, map_location=device)

        self.tokenizer = ByteLevelBPETokenizer(
            os.path.join(CKPT_DIR, "vocab.json"),
            os.path.join(CKPT_DIR, "merges.txt"),
        )
        self.max_len = ckpt["max_len"]
        self.device = device

        self.model = MedicalEmbeddingModel(
            vocab_size=ckpt["vocab_size"],
            dim=ckpt.get("dim", 160),
            n_heads=ckpt.get("n_heads", 4),
            n_layers=ckpt.get("n_layers", 3),
            ff_dim=ckpt.get("ff_dim", 384),
            max_len=self.max_len,
        )
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(device)
        self.model.eval()

    def _encode_batch_ids(self, texts):
        ids = []
        for t in texts:
            enc = self.tokenizer.encode(t)
            toks = enc.ids[: self.max_len - 2]
            toks = [1] + toks + [2]
            pad_len = self.max_len - len(toks)
            toks = toks + [0] * pad_len
            ids.append(toks)
        return torch.tensor(ids, dtype=torch.long, device=self.device)

    @torch.no_grad()
    def embed(self, texts, batch_size=64):
        if isinstance(texts, str):
            texts = [texts]
        all_vecs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            ids = self._encode_batch_ids(batch)
            vecs = self.model(ids)
            all_vecs.append(vecs.cpu())
        return torch.cat(all_vecs, dim=0).numpy()
