"""Stage 1 severity signals.

Three independent, label-free signals, each mapping a ticket to a continuous
severity score in [0, 3] (0=Low .. 3=Critical):

  1. RuleSignal            - weighted urgency lexicon + negation + escalation phrases
  2. EmbeddingSignal       - embedding-based clustering with severity anchor scoring
  3. ResolutionTimeSignal  - inverse-percentile regression of resolution time

None of the signals ever reads the assigned Priority_Level column.
"""
from __future__ import annotations

import json
import re

import numpy as np

from .config import RANDOM_SEED

# --------------------------------------------------------------------------
# 1. Rule-based NLP signal
# --------------------------------------------------------------------------
# Each entry: (compiled-pattern-source, weight). Severity contribution is the
# max weight matched, modulated by negation and escalation detection.

CRITICAL_PATTERNS = [
    (r"\bcompromis\w+", 3.0),
    (r"\bunauthori[sz]ed\b", 3.0),
    (r"\bfraud\w*", 3.0),
    (r"\bstolen\b", 3.0),
    (r"\bphishing\b", 3.0),
    (r"\bsuspicious\b", 2.9),
    (r"\bhack\w*", 3.0),
    (r"(didn'?t|did not|never)\s+(make|authori[sz]e|request)", 3.0),
    (r"wasn'?t\s+me\b", 3.0),
    (r"\block\s+my\s+account\b", 3.0),
    (r"login\s+alert", 2.8),
    (r"new\s+device\b.*\b(trusted|added)|(trusted|added)\b.*\bnew\s+device", 2.8),
    (r"(security|data)\s+breach", 3.0),
    (r"\bsomeone\s+(used|accessed|is\s+using)\b", 2.9),
    (r"(country|city|location|place)\s+i\s*('ve|\s+have)?\s+never\b", 2.8),
    (r"sign[- ]?in\s+from\b", 2.4),
    (r"\b(purchases?|payments?|withdrawals?)\b.*\bmy\s+card\b|\bmy\s+card\b.*\b(purchases?|payments?|withdrawals?)\b", 2.9),
    (r"sign[- ]?in\s+(history|activity)\b", 2.4),
]

HIGH_PATTERNS = [
    (r"\bcrash\w*", 2.0),
    (r"\bnot\s+loading\b", 2.0),
    (r"\b(500|internal\s+server)\b", 2.1),
    (r"\b(cannot|can'?t|unable\s+to)\s+(log|sign|access)\w*", 2.0),
    (r"\b(hasn'?t|not)\s+sync\w*|\bsync\w*\s+(failed|stopped)", 2.0),
    (r"\b(outage|down|downtime|unavailable)\b", 2.1),
    (r"\bspinning\s+wheel\b", 1.9),
    (r"\bfreez\w+|\bfrozen\b", 1.9),
    (r"\b(error|failing|fails|broken)\b", 1.6),
    (r"\bproduction\b.*\b(unable|failing|down|blocked)|"
     r"\b(unable|failing|down|blocked)\b.*\bproduction\b", 2.3),
    (r"\b(no|none of (our|my))\s+(customers?|users?|team|staff|employees?|anyone)\s+can\b", 2.2),
    (r"\blos(?:ing|e|t)\s+(orders?|revenue|sales|customers?|money)\b", 2.2),
    (r"\b(cannot|can'?t|unable\s+to)\s+(reach|use|open)\b.*\b(platform|site|app|system)\b", 2.1),
    (r"\blocked\s+out\b", 1.9),
]

MEDIUM_PATTERNS = [
    (r"\b(double|twice|duplicate)\s*charg\w*|\bcharged\s+(twice|double)\b", 1.2),
    (r"\brefund\w*", 1.1),
    (r"\bbill\w*\s+(higher|wrong|incorrect)|\bovercharg\w*", 1.2),
    (r"\bpayment\s+method\b.*\bfail\w*|\bfail\w*.*\bpayment\s+method\b", 1.2),
    (r"\brenewed\s+automatically\b|\bauto[- ]?renew\w*", 1.0),
    (r"\bnot\s+receiving\b.*\b(email|reset)|\b(email|reset)\b.*\bnot\s+arriv\w*", 1.1),
    (r"\b2fa\b|\btwo[- ]factor\b", 1.3),
    (r"\bcancel\w*", 0.9),
    (r"\bcharge\b|\bcharged\b", 1.0),
]

LOW_PATTERNS = [
    (r"\bhow\s+(do|does|can)\b", 0.15),
    (r"\bwhere\s+(is|are)\b", 0.15),
    (r"\bdo\s+you\s+offer\b", 0.1),
    (r"\b(would|i'?d)\s+like\s+to\s+request\b", 0.2),
    (r"\broadmap\b|\bfeature\s+request\b", 0.1),
    (r"\bpricing\b|\bdiscount\b|\bdemo\b|\bupgrade\b", 0.2),
    (r"\boperating\s+hours\b|\bheadquarters\b|\boffice\s+location\b", 0.05),
    (r"\bprofile\s+picture\b", 0.3),
    (r"\bchange\s+the\s+email\b|\bchange\s+email\b", 0.3),
    (r"\bdelete\s+my\s+account\b", 0.4),
    (r"\binvoice\b", 0.5),
    (r"\binstall\w*", 0.4),
]

# Escalation phrases bump severity; they signal user-perceived urgency.
ESCALATION_PATTERNS = [
    r"\bimmediately\b", r"\burgent\w*\b", r"\basap\b", r"\bcritical\b",
    r"\bemergency\b", r"\bright\s+now\b", r"\bescalat\w+",
]
ESCALATION_BUMP = 0.35
ESCALATION_CAP = 0.7

# Negation within a short window before an urgency term cancels it:
# "there is no fraud", "this is not urgent", "no unauthorized activity".
NEGATION_RE = re.compile(
    r"\b(no|not|none|never|isn'?t|wasn'?t|aren'?t|without|nothing)\s+"
    r"(\w+\s+){0,2}?(fraud\w*|unauthori[sz]ed|suspicious|urgent\w*|critical|"
    r"compromis\w+|hack\w*|stolen|phishing|breach|emergency)", re.I)

# Negation cues open a forward scope: urgency terms whose match starts within
# NEG_WINDOW characters after a cue are treated as negated ("no fraud or
# suspicious activity" negates both terms).
NEG_CUE_RE = re.compile(
    r"\b(no|none|never|isn'?t|wasn'?t|aren'?t|without|nothing|not)\b", re.I)
NEG_WINDOW = 60


class RuleSignal:
    """Weighted lexicon scorer with negation and escalation handling."""

    name = "rule"

    def __init__(self):
        self.groups = [
            ("critical", CRITICAL_PATTERNS),
            ("high", HIGH_PATTERNS),
            ("medium", MEDIUM_PATTERNS),
            ("low", LOW_PATTERNS),
        ]
        self._compiled = [
            (g, [(re.compile(p, re.I), w) for p, w in pats]) for g, pats in self.groups
        ]
        self._esc = [re.compile(p, re.I) for p in ESCALATION_PATTERNS]

    def explain(self, text: str) -> dict:
        """Score one text and return matched evidence (for dossiers)."""
        cue_ends = [m.end() for m in NEG_CUE_RE.finditer(text)]

        def is_negated(span):
            return any(0 < span[0] - ce <= NEG_WINDOW for ce in cue_ends)

        hits, score = [], 0.0
        for group, pats in self._compiled:
            for rx, w in pats:
                m = rx.search(text)
                if m:
                    if group in ("critical", "high") and is_negated(m.span()):
                        hits.append({"phrase": m.group(0), "group": group,
                                     "weight": 0.0, "negated": True})
                        continue
                    hits.append({"phrase": m.group(0), "group": group,
                                 "weight": w, "negated": False})
                    score = max(score, w)
        esc_hits = [rx.search(text).group(0) for rx in self._esc if rx.search(text)]
        # Escalation language amplifies an existing substantive signal but cannot,
        # by itself, raise severity above ~Medium (anti keyword-stuffing guard).
        if esc_hits:
            if score >= 1.0:
                score = min(3.0, score + min(ESCALATION_CAP,
                                             ESCALATION_BUMP * len(esc_hits)))
            else:
                score = min(1.2, score + 0.3)
        # Pure-question damping: interrogative-only tickets with no problem terms.
        if score <= 0.6 and re.search(r"\?", text) and not re.search(
                r"\b(error|fail\w*|crash\w*|charge\w*|refund)\b", text, re.I):
            score = min(score, 0.3)
        return {"score": float(np.clip(score, 0.0, 3.0)),
                "matches": hits, "escalation": esc_hits}

    def score(self, texts) -> np.ndarray:
        return np.array([self.explain(t)["score"] for t in texts], dtype=np.float32)

    def negation_flags(self, texts) -> np.ndarray:
        """True where the text explicitly negates an urgency/security term."""
        return np.array([bool(NEGATION_RE.search(t)) for t in texts])


# --------------------------------------------------------------------------
# 2. Embedding-based clustering signal
# --------------------------------------------------------------------------
# Backend "st"  : sentence-transformers all-MiniLM-L6-v2 (needs internet/GPU box)
# Backend "lsa" : self-trained TF-IDF + truncated SVD embeddings (offline)
# Both share the same downstream logic: KMeans clustering, then each cluster is
# scored against severity *anchor texts* by cosine similarity; every member
# ticket inherits its cluster's continuous severity.

ANCHOR_TEXTS = {
    3: ["account compromised hacked unauthorized transaction fraud stolen card "
        "suspicious activity phishing security breach lock account immediately "
        "login alert unknown device not me"],
    2: ["application crashes error 500 internal server error dashboard not "
        "loading spinning wheel cannot log in unable to access data not syncing "
        "service down outage production broken freezes"],
    1: ["double charge refund not received bill higher overcharged payment "
        "method failing subscription renewed automatically cancel not receiving "
        "password reset email 2fa locked out account access"],
    0: ["how do i install question where is headquarters office location "
        "operating hours pricing tiers discount demo request feature roadmap "
        "upgrade plan change email profile picture general inquiry "
        "send me the invoice transaction receipt copy"],
}

_TOKEN_RE = re.compile(r"[a-z0-9']{2,}")


def _tokenize(text: str):
    return _TOKEN_RE.findall(text.lower())


class LsaEncoder:
    """TF-IDF (word unigram+bigram) -> truncated SVD, pure numpy, offline."""

    def __init__(self, max_features: int = 3000, dim: int = 128, min_df: int = 3):
        self.max_features, self.dim, self.min_df = max_features, dim, min_df
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray | None = None
        self.components: np.ndarray | None = None  # (vocab, dim)

    def _doc_terms(self, text):
        toks = _tokenize(text)
        return toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]

    def fit(self, texts):
        from collections import Counter
        df_counter = Counter()
        docs = []
        for t in texts:
            terms = set(self._doc_terms(t))
            docs.append(terms)
            df_counter.update(terms)
        eligible = [(term, c) for term, c in df_counter.items() if c >= self.min_df]
        eligible.sort(key=lambda x: -x[1])
        self.vocab = {t: i for i, (t, _) in enumerate(eligible[: self.max_features])}
        n = len(texts)
        dfv = np.zeros(len(self.vocab), dtype=np.float64)
        for term, i in self.vocab.items():
            dfv[i] = df_counter[term]
        self.idf = np.log((1 + n) / (1 + dfv)) + 1.0
        X = self._tfidf(texts)
        # Truncated SVD via eigendecomposition of X^T X (vocab x vocab).
        cov = X.T @ X
        eigval, eigvec = np.linalg.eigh(cov)
        order = np.argsort(eigval)[::-1][: self.dim]
        self.components = eigvec[:, order].astype(np.float32)
        return self

    def _tfidf(self, texts) -> np.ndarray:
        X = np.zeros((len(texts), len(self.vocab)), dtype=np.float32)
        for r, t in enumerate(texts):
            from collections import Counter
            counts = Counter(self._doc_terms(t))
            for term, c in counts.items():
                j = self.vocab.get(term)
                if j is not None:
                    X[r, j] = c
        X *= self.idf.astype(np.float32)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return X / norms

    def encode(self, texts) -> np.ndarray:
        E = self._tfidf(texts) @ self.components
        norms = np.linalg.norm(E, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return E / norms

    def to_dict(self):
        return {"vocab": self.vocab, "idf": self.idf.tolist(),
                "components": self.components.tolist(),
                "max_features": self.max_features, "dim": self.dim,
                "min_df": self.min_df}

    @classmethod
    def from_dict(cls, d):
        enc = cls(d["max_features"], d["dim"], d["min_df"])
        enc.vocab = d["vocab"]
        enc.idf = np.array(d["idf"], dtype=np.float64)
        enc.components = np.array(d["components"], dtype=np.float32)
        return enc


class StEncoder:
    """sentence-transformers backend (lazy import; used on the GPU/online box)."""

    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def fit(self, texts):
        return self

    def encode(self, texts) -> np.ndarray:
        E = self.model.encode(list(texts), normalize_embeddings=True,
                              show_progress_bar=False)
        return np.asarray(E, dtype=np.float32)


def _kmeans(X: np.ndarray, k: int, iters: int = 50, seed: int = RANDOM_SEED):
    rng = np.random.default_rng(seed)
    # k-means++ init
    centroids = [X[rng.integers(len(X))]]
    for _ in range(k - 1):
        d2 = np.min(
            np.stack([np.sum((X - c) ** 2, axis=1) for c in centroids]), axis=0)
        p = d2 / d2.sum()
        centroids.append(X[rng.choice(len(X), p=p)])
    C = np.stack(centroids)
    for _ in range(iters):
        d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1) if len(X) * k < 4e6 \
            else np.stack([((X - c) ** 2).sum(1) for c in C], axis=1)
        assign = d.argmin(1)
        newC = np.stack([X[assign == j].mean(0) if np.any(assign == j) else C[j]
                         for j in range(k)])
        if np.allclose(newC, C, atol=1e-6):
            C = newC
            break
        C = newC
    return C, assign


class EmbeddingSignal:
    """Semantic urgency grouping: cluster embeddings, score clusters vs anchors."""

    name = "embedding"

    def __init__(self, encoder=None, k: int = 40, temperature: float = 12.0):
        self.encoder = encoder or LsaEncoder()
        self.k = k
        self.temperature = temperature
        self.centroids: np.ndarray | None = None
        self.cluster_severity: np.ndarray | None = None
        self.cluster_anchor: list | None = None

    def fit(self, texts):
        self.encoder.fit(texts)
        E = self.encoder.encode(texts)
        self.centroids, _ = _kmeans(E, self.k)
        norms = np.linalg.norm(self.centroids, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.centroids = self.centroids / norms
        levels = sorted(ANCHOR_TEXTS)
        A = self.encoder.encode([ANCHOR_TEXTS[l][0] for l in levels])  # (4, dim)
        sims = self.centroids @ A.T                                    # (k, 4)
        w = np.exp(self.temperature * sims)
        w /= w.sum(1, keepdims=True)
        self.cluster_severity = (w * np.array(levels)).sum(1).astype(np.float32)
        self.cluster_anchor = [int(np.argmax(s)) for s in sims]
        return self

    def _assign(self, texts):
        E = self.encoder.encode(texts)
        sims = E @ self.centroids.T
        return sims.argmax(1), sims.max(1)

    def score(self, texts) -> np.ndarray:
        cl, _ = self._assign(texts)
        return self.cluster_severity[cl]

    def explain(self, text: str) -> dict:
        cl, sim = self._assign([text])
        c = int(cl[0])
        return {"score": float(self.cluster_severity[c]), "cluster": c,
                "similarity": float(sim[0]),
                "anchor_level": int(self.cluster_anchor[c])}

    def to_dict(self):
        return {"k": self.k, "temperature": self.temperature,
                "centroids": self.centroids.tolist(),
                "cluster_severity": self.cluster_severity.tolist(),
                "cluster_anchor": self.cluster_anchor,
                "encoder": self.encoder.to_dict(),
                "encoder_type": "lsa"}

    @classmethod
    def from_dict(cls, d):
        sig = cls(LsaEncoder.from_dict(d["encoder"]), d["k"], d["temperature"])
        sig.centroids = np.array(d["centroids"], dtype=np.float32)
        sig.cluster_severity = np.array(d["cluster_severity"], dtype=np.float32)
        sig.cluster_anchor = d["cluster_anchor"]
        return sig


# --------------------------------------------------------------------------
# 3. Resolution-time signal
# --------------------------------------------------------------------------
class ResolutionTimeSignal:
    """Inverse-percentile mapping: faster operational handling => higher severity.

    Caveat (documented in README): resolution time is partially contaminated by
    the assigned priority itself (SLA effect), so this signal receives the
    lowest fusion weight and is treated as corroborating evidence only.
    """

    name = "resolution_time"

    def __init__(self):
        self.sorted_hours: np.ndarray | None = None
        self.median: float | None = None

    def fit(self, hours):
        h = np.asarray(hours, dtype=np.float64)
        h = h[~np.isnan(h)]
        self.sorted_hours = np.sort(h)
        self.median = float(np.median(h))
        return self

    def percentile(self, hours):
        h = np.asarray(hours, dtype=np.float64)
        h = np.where(np.isnan(h), self.median, h)
        ranks = np.searchsorted(self.sorted_hours, h, side="right")
        return ranks / len(self.sorted_hours)

    def score(self, hours) -> np.ndarray:
        # severity = 3 * (1 - percentile): fastest-resolved -> ~3, slowest -> ~0
        return (3.0 * (1.0 - self.percentile(hours))).astype(np.float32)

    def explain(self, hour) -> dict:
        missing = hour is None or (isinstance(hour, float) and np.isnan(hour))
        h = self.median if missing else float(hour)
        pct = float(self.percentile([h])[0])
        return {"score": float(3.0 * (1.0 - pct)), "hours": h,
                "percentile": pct, "imputed": bool(missing)}

    def to_dict(self):
        return {"sorted_hours": self.sorted_hours.tolist(), "median": self.median}

    @classmethod
    def from_dict(cls, d):
        sig = cls()
        sig.sorted_hours = np.array(d["sorted_hours"], dtype=np.float64)
        sig.median = d["median"]
        return sig


def save_signals(path, emb: EmbeddingSignal, rt: ResolutionTimeSignal):
    with open(path, "w") as f:
        json.dump({"embedding": emb.to_dict(), "resolution_time": rt.to_dict()}, f)


def load_signals(path):
    with open(path) as f:
        d = json.load(f)
    return (RuleSignal(), EmbeddingSignal.from_dict(d["embedding"]),
            ResolutionTimeSignal.from_dict(d["resolution_time"]))
