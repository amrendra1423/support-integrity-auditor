"""Submission-grade classifier backend: fine-tuned transformer (HF).

Requires: torch, transformers, datasets (see requirements.txt). Run on a GPU
box / Colab. Default model is DeBERTa-v3-small, fully fine-tuned with a
class-weighted loss; --lora enables adapter training via peft instead.

Interface mirrors lite_model.LiteMLP: fit / predict_proba / save / load.
Inputs are the serialized text built by features.serialized_text(), which
embeds the structured metadata (assigned priority, channel, customer tier,
category, resolution hours) into the sequence.
"""
from __future__ import annotations

import json
import os

import numpy as np


class HfClassifier:
    def __init__(self, model_name: str = "microsoft/deberta-v3-small",
                 max_length: int = 128, epochs: int = 3, lr: float = 2e-5,
                 batch_size: int = 32, lora: bool = False, seed: int = 42):
        self.model_name = model_name
        self.max_length, self.epochs, self.lr = max_length, epochs, lr
        self.batch_size, self.lora, self.seed = batch_size, lora, seed
        self.tokenizer = None
        self.model = None

    # ------------------------------------------------------------------ fit
    def fit(self, texts, y, val_texts=None, val_y=None, verbose=True):
        import torch
        from torch.utils.data import DataLoader, Dataset
        from transformers import (AutoModelForSequenceClassification,
                                  AutoTokenizer, get_linear_schedule_with_warmup)

        torch.manual_seed(self.seed)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=2)
        if self.lora:
            from peft import LoraConfig, TaskType, get_peft_model
            cfg = LoraConfig(task_type=TaskType.SEQ_CLS, r=16, lora_alpha=32,
                             lora_dropout=0.05)
            self.model = get_peft_model(self.model, cfg)
        self.model.to(device)

        y = np.asarray(y, int)
        counts = np.bincount(y, minlength=2).astype(np.float64)
        class_weights = torch.tensor(counts.sum() / (2.0 * counts),
                                     dtype=torch.float32, device=device)
        if verbose:
            print(f"device={device} class_weights={class_weights.tolist()}")

        class DS(Dataset):
            def __init__(s, t, l): s.t, s.l = list(t), l
            def __len__(s): return len(s.t)
            def __getitem__(s, i): return s.t[i], int(s.l[i])

        def collate(batch):
            t, l = zip(*batch)
            enc = self.tokenizer(list(t), truncation=True, padding=True,
                                 max_length=self.max_length, return_tensors="pt")
            return enc, torch.tensor(l)

        loader = DataLoader(DS(texts, y), batch_size=self.batch_size,
                            shuffle=True, collate_fn=collate)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr,
                                weight_decay=0.01)
        steps = len(loader) * self.epochs
        sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
        lossf = torch.nn.CrossEntropyLoss(weight=class_weights)

        self.model.train()
        for ep in range(self.epochs):
            tot = 0.0
            for enc, labels in loader:
                enc = {k: v.to(device) for k, v in enc.items()}
                labels = labels.to(device)
                opt.zero_grad()
                out = self.model(**enc)
                loss = lossf(out.logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step(); sched.step()
                tot += float(loss)
            msg = f"epoch {ep} loss={tot / len(loader):.4f}"
            if val_texts is not None:
                acc = float(np.mean(self.predict(val_texts) == np.asarray(val_y)))
                msg += f" val_acc={acc:.4f}"
            if verbose:
                print(msg)
        return self

    # ------------------------------------------------------------- inference
    def predict_proba(self, texts):
        import torch
        device = next(self.model.parameters()).device
        self.model.eval()
        probs = []
        with torch.no_grad():
            for s in range(0, len(texts), 64):
                enc = self.tokenizer(list(texts[s:s + 64]), truncation=True,
                                     padding=True, max_length=self.max_length,
                                     return_tensors="pt").to(device)
                p = torch.softmax(self.model(**enc).logits, dim=-1)
                probs.append(p.cpu().numpy())
        return np.vstack(probs)

    def predict(self, texts):
        return self.predict_proba(texts).argmax(1)

    # ----------------------------------------------------------- persistence
    def save(self, outdir):
        os.makedirs(outdir, exist_ok=True)
        self.model.save_pretrained(outdir)
        self.tokenizer.save_pretrained(outdir)
        with open(os.path.join(outdir, "sia_hf_config.json"), "w") as f:
            json.dump({"model_name": self.model_name,
                       "max_length": self.max_length}, f)

    @classmethod
    def load(cls, outdir):
        from transformers import (AutoModelForSequenceClassification,
                                  AutoTokenizer)
        with open(os.path.join(outdir, "sia_hf_config.json")) as f:
            cfg = json.load(f)
        obj = cls(model_name=cfg["model_name"], max_length=cfg["max_length"])
        obj.tokenizer = AutoTokenizer.from_pretrained(outdir)
        obj.model = AutoModelForSequenceClassification.from_pretrained(outdir)
        return obj
