#!/usr/bin/env python3
"""Local text/identity overlap screening, derived from PostTrainBench decon SIMPLE.

Uses BPE n-grams, cluster expansion, code/diff normalization and identity
signatures. This variant scans every string field in training JSONL, refuses
unreadable/empty records, and handles a zero-IDF reference index.

This local CLI compares only the reference explicitly supplied by its caller.
Visible tasks are allowed development data: matching visible is not itself a
contamination violation. For private select/test screening, researchers use the
separate ./contamination check broker client, never private reference paths.

Any text/identity hit is a review signal, not an automatic violation. Heavy
paraphrase, translation and instance-seeded generation can evade this method.
NO_MATCHES is not proof of absence of contamination; review provenance too.

Usage:
    python contamination_check.py --reference tasks_visible/swe --input train.jsonl
    python contamination_check.py --reference tasks_visible/tb --input train.patch --input-format patch
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass
from dataclasses import field as dc_field
from pathlib import Path

# ---------- decon defaults (ported verbatim from PTB v1.1) ----------
QUESTION_NGRAM_SIZE = 5
ANSWER_NGRAM_SIZE = 3
PASSAGE_NGRAM_SIZE = 4
CONTAMINATION_THRESHOLD = 0.8
PERFECT_MATCH_DECAY_START = 20   # tokens
PERFECT_MATCH_DECAY_END = 50     # tokens
SHORT_ANSWER_TOKEN_THRESHOLD = 3
SHORT_ANSWER_WINDOW_LENGTH = 50
MIN_LONG_ANSWER_WINDOW = 100
MIN_PASSAGE_DISTANCE = 100
PASSAGE_MAX_CONSECUTIVE_MISSES = 2
SAMPLE_EVERY_M_TOKENS = 10
QUESTION_MAX_CONSECUTIVE_MISSES = 11
MIN_INFORMATIVE_NGRAMS = 20
MIN_INFORMATIVE_ANSWER_TOKENS = 4
TOKENIZER_NAME = "cl100k_base"

# Text normalization (decon common/text.rs)
DEFAULT_PUNCTUATION_CHARS = "!'\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~“”‘’—"
_PUNCT_TABLE = str.maketrans({c: ' ' for c in DEFAULT_PUNCTUATION_CHARS})
_MULTI_WS = re.compile(r'\s+')

# Reference preprocessing (decon defaults)
EVAL_DEDUP = True
EVAL_MIN_TOKEN_LENGTH = 20
EVAL_MIN_UNIQUE_WORD_COUNT = 4

# Field detection order: first match wins.
# CoevoBench addition: SWE problem_statement / tb instruction fields prepended
# so an issue text or task instruction is picked up as the "question".
QUESTION_FIELDS = (
    "problem_statement", "problem", "issue", "instruction", "query",
    "question", "input", "prompt", "text", "content",
)
ANSWER_FIELDS = ("answer", "output", "response", "target", "patch", "solution")
PASSAGE_FIELDS = ("passage", "context", "document", "hints_text")

# Base component weights (decon defaults)
COMPONENT_WEIGHTS = {
    frozenset(["question", "answer", "passage"]): {"question": 0.70, "answer": 0.20, "passage": 0.10},
    frozenset(["question", "answer"]):            {"question": 0.75, "answer": 0.25},
    frozenset(["question", "passage"]):           {"question": 0.85, "passage": 0.15},
    frozenset(["question"]):                      {"question": 1.00},
}

# cl100k BPE is the faithful decon tokenizer. If tiktoken is unavailable (e.g.
# the tool is run outside the pre-installed venv), fall back to a deterministic
# whitespace word tokenizer and warn: the n-gram sizes then apply to words
# rather than BPE tokens — a coarser but still useful decon variant.
try:
    import tiktoken
    _enc = tiktoken.get_encoding(TOKENIZER_NAME)
    SPACE_TOKEN = _enc.encode(" ")[0]
    _USING_TIKTOKEN = True
except Exception as _e:  # pragma: no cover - environment dependent
    print(f"WARNING: tiktoken unavailable ({_e}); falling back to whitespace "
          f"word tokenization (coarser n-gram matching).", file=sys.stderr)

    class _WhitespaceEncoder:
        """Minimal deterministic stand-in for tiktoken's encode()."""
        def __init__(self):
            self._vocab: dict = {}

        def encode(self, text: str) -> list:
            ids = []
            for w in text.split():
                i = self._vocab.get(w)
                if i is None:
                    i = len(self._vocab) + 1  # ids start at 1; 0 reserved as SPACE_TOKEN sentinel
                    self._vocab[w] = i
                ids.append(i)
            return ids

    _enc = _WhitespaceEncoder()
    SPACE_TOKEN = 0  # sentinel never produced by the whitespace encoder
    _USING_TIKTOKEN = False


# ---------------------------------------------------------------------------
# Helpers (ported verbatim from PTB v1.1)
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Lowercase, replace punctuation with spaces, collapse whitespace (matching decon)."""
    return _MULTI_WS.sub(' ', text.lower().translate(_PUNCT_TABLE)).strip()


def tokenize(text: str) -> list[int]:
    """Clean text, then BPE tokenize with leading space padding (matching decon)."""
    return _enc.encode(" " + clean_text(text))


def make_ngrams(tokens, n: int) -> list[tuple]:
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def interpolate_threshold(cumulative_tokens: int, base_threshold: float) -> float:
    """Linear interpolation matching decon's interpolate_threshold."""
    if cumulative_tokens <= PERFECT_MATCH_DECAY_START:
        return 1.0
    if cumulative_tokens >= PERFECT_MATCH_DECAY_END:
        return base_threshold
    t = (cumulative_tokens - PERFECT_MATCH_DECAY_START) / (PERFECT_MATCH_DECAY_END - PERFECT_MATCH_DECAY_START)
    return 1.0 - (1.0 - base_threshold) * t


def extract_fields(item: dict) -> dict[str, str]:
    fields = {}
    for key in QUESTION_FIELDS:
        if key in item and isinstance(item[key], str) and item[key].strip():
            fields["question"] = item[key]
            break
    for key in ANSWER_FIELDS:
        if key in item and isinstance(item[key], str) and item[key].strip():
            fields["answer"] = item[key]
            break
    for key in PASSAGE_FIELDS:
        if key in item and isinstance(item[key], str) and item[key].strip():
            fields["passage"] = item[key]
            break
    if not fields:
        parts = [v for v in item.values() if isinstance(v, str) and v.strip()]
        if parts:
            fields["question"] = " ".join(parts)
    return fields


def adaptive_weights(base_weights: dict[str, float], item_data: dict) -> dict[str, float]:
    """Adjust component weights based on question/answer confidence (decon scoring.rs)."""
    present = set(base_weights.keys())
    has_answer = "answer" in present
    has_passage = "passage" in present

    q_ngrams = len(item_data.get("question", {}).get("ngrams", set()))
    q_conf = (0.5 + (q_ngrams / MIN_INFORMATIVE_NGRAMS) * 0.5
              if q_ngrams < MIN_INFORMATIVE_NGRAMS else 1.0)

    if has_answer and has_passage:
        a_tokens = item_data.get("answer", {}).get("unique_token_count", 0)
        a_conf = (0.5 + (a_tokens / MIN_INFORMATIVE_ANSWER_TOKENS) * 0.5
                  if a_tokens < MIN_INFORMATIVE_ANSWER_TOKENS else 1.0)
        if q_conf == 1.0 and a_conf == 1.0:
            return dict(base_weights)
        q_w = base_weights["question"] * q_conf
        q_lost = base_weights["question"] * (1.0 - q_conf)
        a_w = base_weights["answer"] + q_lost * 0.5
        p_w = base_weights["passage"] + q_lost * 0.5
        a_lost = a_w * (1.0 - a_conf)
        a_w *= a_conf
        p_w += a_lost
        return {"question": q_w, "answer": a_w, "passage": p_w}

    if has_answer:
        if q_conf == 1.0:
            return dict(base_weights)
        q_w = base_weights["question"] * q_conf
        a_w = base_weights["answer"] + base_weights["question"] * (1.0 - q_conf)
        return {"question": q_w, "answer": a_w}

    if has_passage:
        if q_conf == 1.0:
            return dict(base_weights)
        q_w = base_weights["question"] * q_conf
        p_w = base_weights["passage"] + base_weights["question"] * (1.0 - q_conf)
        return {"question": q_w, "passage": p_w}

    return dict(base_weights)


# ---------------------------------------------------------------------------
# CoevoBench additions: patch normalization + identity signatures
# ---------------------------------------------------------------------------

_DIFF_MARKERS = re.compile(r'(?m)^(diff --git |@@ |--- |\+\+\+ |index [0-9a-f]{7})')
_DIFF_FILE_HDR = re.compile(r'(?m)^diff --git a/(\S+) b/(\S+)')


def looks_like_diff(text: str) -> bool:
    """Heuristic: does this text contain unified-diff structure?"""
    return bool(_DIFF_MARKERS.search(text))


def normalize_patch(text: str) -> str:
    """Reduce a unified diff to its added/context *content*.

    Drops diff/hunk headers, file markers, and removed lines; keeps the content
    of added lines (leading '+') and context lines. The decon core then n-grams
    this like ordinary text (clean_text collapses whitespace, so indentation and
    reflowing do not defeat the match).
    """
    out = []
    for line in text.splitlines():
        if line.startswith(("diff --git", "index ", "--- ", "+++ ", "@@",
                            "new file mode", "deleted file mode", "similarity index",
                            "rename from", "rename to", "old mode", "new mode",
                            "Binary files", "GIT binary patch")):
            continue
        if line.startswith("-"):
            continue  # removed line: not present in the produced artifact
        if line.startswith("+"):
            out.append(line[1:])
        else:
            # context line (leading space) or plain line
            out.append(line[1:] if line.startswith(" ") else line)
    return "\n".join(out)


def split_patch_documents(text: str) -> list[dict]:
    """Split a (possibly multi-file) unified diff into one document per file.

    Returns docs whose 'text' is the normalized added/context content. Falls
    back to a single document if no file headers are present.
    """
    hdrs = list(_DIFF_FILE_HDR.finditer(text))
    docs: list[dict] = []
    if not hdrs:
        norm = normalize_patch(text)
        if norm.strip():
            docs.append({"text": norm, "raw": text, "source": "patch:0"})
        return docs
    for i, m in enumerate(hdrs):
        start = m.start()
        end = hdrs[i + 1].start() if i + 1 < len(hdrs) else len(text)
        chunk = text[start:end]
        norm = normalize_patch(chunk)
        if norm.strip():
            docs.append({"text": norm, "raw": chunk, "source": f"patch:{m.group(2)}"})
    return docs


def build_signatures(item: dict, min_signature_len: int) -> set[str]:
    """Derive precise identity signatures (lowercased) from a reference item.

    Only high-precision signatures are emitted so that a substring hit in a
    training document is meaningful (the "fetched by known instance ID" vector).
    Bare repo names and bare issue numbers are deliberately NOT signatures.
    """
    sigs: set[str] = set()
    iid = None
    for key in ("instance_id", "id", "task_id", "task_name"):
        v = item.get(key)
        if isinstance(v, str) and v.strip():
            iid = v.strip()
            break
    if not iid:
        return sigs

    iid_l = iid.lower()
    distinctive = ("__" in iid) or any(c.isdigit() for c in iid) or len(iid) >= min_signature_len
    if distinctive:
        sigs.add(iid_l)

    # SWE-bench instance-id form: {org}__{repo}-{number}
    m = re.match(r'^([\w.\-]+)__([\w.\-]+)-(\d+)$', iid)
    if m:
        org, repo, num = m.groups()
        full = f"{org}/{repo}".lower()
        sigs.add(f"{full}#{num}")
        sigs.add(f"{full}/pull/{num}")
        sigs.add(f"{full}/issues/{num}")
        sigs.add(f"{full}/pull/{num}/")
    # explicit repo field + a number in the id
    repo_field = item.get("repo")
    if isinstance(repo_field, str) and repo_field.strip():
        rf = repo_field.strip().lower()
        num_m = re.search(r'(\d{3,})', iid)
        if num_m:
            num = num_m.group(1)
            sigs.add(f"{rf}#{num}")
            sigs.add(f"{rf}/pull/{num}")
            sigs.add(f"{rf}/issues/{num}")
    return sigs


# ---------------------------------------------------------------------------
# Cluster data (ported verbatim)
# ---------------------------------------------------------------------------

@dataclass
class ClusterDocMatch:
    """Per-reference-item data within one cluster."""
    matched_ngs: set = dc_field(default_factory=set)
    start_idx: int = 0   # leftmost n-gram position
    end_idx: int = 0      # rightmost n-gram position


@dataclass
class Cluster:
    """Result of one seed + expansion pass."""
    doc_matches: dict = dc_field(default_factory=dict)
    rightmost_idx: int = 0


# ---------------------------------------------------------------------------
# Reference index + detection (decon core ported verbatim; two early-exit
# thresholds parameterized on the query threshold so a lower "near" band can be
# reported; zero-IDF handling and input coverage are hardened in this port)
# ---------------------------------------------------------------------------

class ReferenceIndex:
    """N-gram index with decon-matching cluster detection and scoring."""

    def __init__(self, reference_items: list, min_signature_len: int = 8, *, include_short=False):
        self.items = reference_items
        self.n_items = len(reference_items)
        self.include_short = include_short
        self.item_fields: list = []

        self.q_ng_to_items: dict = {}

        self.q_idf: dict = {}
        self.a_idf: dict = {}
        self.p_idf: dict = {}

        # CoevoBench: identity signatures -> reference item index
        self.min_signature_len = min_signature_len
        self.sig_to_item: dict = {}

        self._build()
        self._compute_total_idfs()

    # ---- index building ----

    def _build(self):
        q_df: Counter = Counter()
        a_df: Counter = Counter()
        p_df: Counter = Counter()
        seen_dedup: set = set()

        for idx, item in enumerate(self.items):
            # CoevoBench: collect identity signatures regardless of text filtering
            for sig in build_signatures(item, self.min_signature_len):
                self.sig_to_item.setdefault(sig, idx)

            fields = extract_fields(item)

            if not fields:
                self.item_fields.append({})
                continue

            # --- reference preprocessing (matching decon) ---

            # Dedup by (eval_key, fingerprint) if fingerprint present
            if EVAL_DEDUP:
                fp = item.get("fingerprint", "")
                if fp:
                    dedup_key = (item.get("eval_key", ""), fp)
                    if dedup_key in seen_dedup:
                        self.item_fields.append({})
                        continue
                    seen_dedup.add(dedup_key)

            # Build combined cleaned text for filtering
            combined_parts = []
            if "passage" in fields:
                combined_parts.append(clean_text(fields["passage"]))
            if "question" in fields:
                combined_parts.append(clean_text(fields["question"]))
            if "answer" in fields:
                combined_parts.append(clean_text(fields["answer"]))
            combined_text = " ".join(p for p in combined_parts if p)

            # Min unique words check
            if not self.include_short and EVAL_MIN_UNIQUE_WORD_COUNT > 0:
                if len(set(combined_text.split())) < EVAL_MIN_UNIQUE_WORD_COUNT:
                    self.item_fields.append({})
                    continue

            # Min token length check (non-space tokens)
            if not self.include_short and EVAL_MIN_TOKEN_LENGTH > 0:
                combined_tokens = _enc.encode(" " + combined_text)
                non_space = sum(1 for t in combined_tokens if t != SPACE_TOKEN)
                if non_space < EVAL_MIN_TOKEN_LENGTH:
                    self.item_fields.append({})
                    continue

            # --- index fields ---
            item_data: dict = {}

            if "question" in fields:
                tokens = tokenize(fields["question"])
                ng_set = set(make_ngrams(tokens, QUESTION_NGRAM_SIZE))
                item_data["question"] = {"tokens": tokens, "ngrams": ng_set, "token_count": len(tokens)}
                for ng in ng_set:
                    self.q_ng_to_items.setdefault(ng, set()).add(idx)
                    q_df[ng] += 1

            if "answer" in fields:
                tokens = tokenize(fields["answer"])
                unique_token_count = len(set(tokens))
                is_short = len(tokens) <= SHORT_ANSWER_TOKEN_THRESHOLD
                if is_short:
                    item_data["answer"] = {
                        "tokens": tokens, "ngrams": set(),
                        "token_count": len(tokens), "unique_token_count": unique_token_count,
                        "is_short": True,
                    }
                else:
                    ng_set = set(make_ngrams(tokens, ANSWER_NGRAM_SIZE))
                    item_data["answer"] = {
                        "tokens": tokens, "ngrams": ng_set,
                        "token_count": len(tokens), "unique_token_count": unique_token_count,
                        "is_short": False,
                    }
                    for ng in ng_set:
                        a_df[ng] += 1

            if "passage" in fields:
                tokens = tokenize(fields["passage"])
                ng_set = set(make_ngrams(tokens, PASSAGE_NGRAM_SIZE))
                item_data["passage"] = {"tokens": tokens, "ngrams": ng_set, "token_count": len(tokens)}
                for ng in ng_set:
                    p_df[ng] += 1

            self.item_fields.append(item_data)

        N = self.n_items
        for ng, df in q_df.items():
            self.q_idf[ng] = math.log(N / df) if df < N else 0.0
        for ng, df in a_df.items():
            self.a_idf[ng] = math.log(N / df) if df < N else 0.0
        for ng, df in p_df.items():
            self.p_idf[ng] = math.log(N / df) if df < N else 0.0

    def _compute_total_idfs(self):
        """Pre-compute per-item total IDF sums (decon does this at index build time)."""
        for item_data in self.item_fields:
            if "question" in item_data:
                q = item_data["question"]
                q["total_idf"] = sum(self.q_idf.get(ng, 0.0) for ng in q["ngrams"])
            if "answer" in item_data and not item_data["answer"]["is_short"]:
                a = item_data["answer"]
                a["total_idf"] = sum(self.a_idf.get(ng, 0.0) for ng in a["ngrams"])
            if "passage" in item_data:
                p = item_data["passage"]
                p["total_idf"] = sum(self.p_idf.get(ng, 0.0) for ng in p["ngrams"])

    # ---- identity-signature scan (CoevoBench) ----

    def scan_signatures(self, raw_text: str) -> list:
        """Return [(signature, ref_index), ...] for signatures present verbatim."""
        hay = raw_text.lower()
        hits = []
        for sig, idx in self.sig_to_item.items():
            if sig in hay:
                hits.append((sig, idx))
        return hits

    # ---- cluster detection (decon identify_contamination_clusters) ----

    def _find_clusters(self, doc_tokens: list) -> list:
        total_ngrams = len(doc_tokens) - QUESTION_NGRAM_SIZE + 1
        if total_ngrams <= 0:
            return []

        clusters: list = []
        i = 0
        while i < total_ngrams:
            ng = tuple(doc_tokens[i:i + QUESTION_NGRAM_SIZE])
            if ng not in self.q_ng_to_items:
                i += SAMPLE_EVERY_M_TOKENS
                continue

            initial_items = set(self.q_ng_to_items[ng])
            cluster = self._expand_cluster(i, doc_tokens, initial_items, ng)
            if cluster is not None:
                clusters.append(cluster)
                i = max(i, cluster.rightmost_idx + 1)
            else:
                i += SAMPLE_EVERY_M_TOKENS
        return clusters

    # ---- bidirectional cluster expansion (decon expand_simple_contamination_cluster) ----

    def _expand_cluster(self, hit_idx, doc_tokens, initial_items, initial_ng):
        if len(doc_tokens) < QUESTION_NGRAM_SIZE:
            return None
        total_ngrams = len(doc_tokens) - QUESTION_NGRAM_SIZE + 1

        doc_matches: dict = {}
        doc_misses: dict = {}
        for idx in initial_items:
            doc_matches[idx] = ClusterDocMatch(matched_ngs={initial_ng}, start_idx=hit_idx, end_idx=hit_idx)
            doc_misses[idx] = 0

        # --- left expansion ---
        active = set(initial_items)
        current = hit_idx
        while current > 0 and active:
            current -= 1
            ng = tuple(doc_tokens[current:current + QUESTION_NGRAM_SIZE])
            matched_items = self.q_ng_to_items.get(ng)
            if matched_items is not None:
                intersection = active & matched_items
                if intersection:
                    for idx in intersection:
                        is_new = ng not in doc_matches[idx].matched_ngs
                        doc_matches[idx].matched_ngs.add(ng)
                        if is_new:
                            doc_misses[idx] = 0
                        doc_matches[idx].start_idx = current

                    for idx in active - matched_items:
                        doc_misses[idx] += 1
                        if doc_misses[idx] >= QUESTION_MAX_CONSECUTIVE_MISSES:
                            active.discard(idx)
                    continue

            # no match or no intersection
            to_remove = []
            for idx in active:
                doc_misses[idx] += 1
                if doc_misses[idx] >= QUESTION_MAX_CONSECUTIVE_MISSES:
                    to_remove.append(idx)
            for idx in to_remove:
                active.discard(idx)

        # --- right expansion (reset misses, reset active) ---
        active = set(initial_items)
        for idx in initial_items:
            doc_misses[idx] = 0

        current = hit_idx
        while current + 1 < total_ngrams and active:
            current += 1
            ng = tuple(doc_tokens[current:current + QUESTION_NGRAM_SIZE])
            matched_items = self.q_ng_to_items.get(ng)
            if matched_items is not None:
                intersection = active & matched_items
                if intersection:
                    for idx in intersection:
                        is_new = ng not in doc_matches[idx].matched_ngs
                        doc_matches[idx].matched_ngs.add(ng)
                        if is_new:
                            doc_misses[idx] = 0
                        doc_matches[idx].end_idx = current

                    for idx in active - matched_items:
                        doc_misses[idx] += 1
                        if doc_misses[idx] >= QUESTION_MAX_CONSECUTIVE_MISSES:
                            active.discard(idx)
                    continue

            to_remove = []
            for idx in active:
                doc_misses[idx] += 1
                if doc_misses[idx] >= QUESTION_MAX_CONSECUTIVE_MISSES:
                    to_remove.append(idx)
            for idx in to_remove:
                active.discard(idx)

        return Cluster(doc_matches=doc_matches, rightmost_idx=current)

    # ---- answer boundary detection (decon answer_boundary.rs) ----

    def _find_answer_idf(self, doc_tokens, question_end_token, item_idx) -> float:
        """Find answer in training tokens after the question cluster.
        Returns the answer IDF overlap score."""
        item_data = self.item_fields[item_idx]
        if "answer" not in item_data:
            return 0.0
        a_data = item_data["answer"]
        a_len = a_data["token_count"]

        if a_data["is_short"]:
            # exact sequence match in window after question
            search_start = question_end_token + 1
            search_end = min(search_start + SHORT_ANSWER_WINDOW_LENGTH, len(doc_tokens))
            answer_tokens = a_data["tokens"]
            n = len(answer_tokens)
            if n == 0:
                return 0.0
            for i in range(search_start, search_end - n + 1):
                if doc_tokens[i:i + n] == answer_tokens:
                    return 1.0  # exact match -> full IDF overlap
            return 0.0
        else:
            # n-gram IDF overlap in window after question
            window_size = max(a_len * 2, MIN_LONG_ANSWER_WINDOW)
            search_start = question_end_token + 1
            search_end = min(search_start + window_size, len(doc_tokens))
            if search_end - search_start < ANSWER_NGRAM_SIZE:
                return 0.0

            ref_ngs = a_data["ngrams"]
            matched_ngs: set = set()
            for i in range(search_start, search_end - ANSWER_NGRAM_SIZE + 1):
                ng = tuple(doc_tokens[i:i + ANSWER_NGRAM_SIZE])
                if ng in ref_ngs:
                    matched_ngs.add(ng)

            if not matched_ngs:
                return 0.0
            matched_idf = sum(self.a_idf.get(ng, 0.0) for ng in matched_ngs)
            total_idf = a_data.get("total_idf", 0.0)
            return matched_idf / total_idf if total_idf > 0 else 0.0

    # ---- passage boundary detection (decon passage_boundary.rs) ----

    def _find_passage_idf(self, doc_tokens, question_start_token, item_idx) -> float:
        """Find passage in training tokens before the question cluster.
        Returns the passage IDF overlap score."""
        item_data = self.item_fields[item_idx]
        if "passage" not in item_data:
            return 0.0
        p_data = item_data["passage"]
        p_len = p_data["token_count"]

        window_size = max(p_len * 2, MIN_PASSAGE_DISTANCE)
        search_end = question_start_token  # exclusive
        search_start = max(0, search_end - window_size)
        if search_end - search_start < PASSAGE_NGRAM_SIZE:
            return 0.0

        ref_ngs = p_data["ngrams"]
        matched_ngs: set = set()
        first_match = False
        consecutive_misses = 0

        # traverse backwards from just before the question
        for i in range(search_end - PASSAGE_NGRAM_SIZE, search_start - 1, -1):
            ng = tuple(doc_tokens[i:i + PASSAGE_NGRAM_SIZE])
            if ng in ref_ngs:
                matched_ngs.add(ng)
                first_match = True
                consecutive_misses = 0
            else:
                if first_match:
                    consecutive_misses += 1
                    if consecutive_misses > PASSAGE_MAX_CONSECUTIVE_MISSES:
                        break

        if not matched_ngs:
            return 0.0
        matched_idf = sum(self.p_idf.get(ng, 0.0) for ng in matched_ngs)
        total_idf = p_data.get("total_idf", 0.0)
        return matched_idf / total_idf if total_idf > 0 else 0.0

    # ---- main query entry point ----

    def query(self, text: str, threshold: float) -> list:
        """Detect contamination using sampled n-gram detection + cluster expansion.

        `threshold` is the base contamination bar. The two decon early-exit
        thresholds are derived from it locally, so passing a lower value (a
        "near" band) reports weaker matches too. Zero-IDF references fall back
        to unweighted n-gram overlap instead of silently missing exact copies.
        """
        min_q_idf = max(0.0, (threshold - 0.3) / 0.7)

        doc_tokens = tokenize(text)
        if len(doc_tokens) < QUESTION_NGRAM_SIZE:
            return []

        clusters = self._find_clusters(doc_tokens)

        # Collect best match per reference item across all clusters
        best_matches: dict = {}

        for cluster in clusters:
            for item_idx, cmatch in cluster.doc_matches.items():
                item_data = self.item_fields[item_idx]
                present = set(item_data.keys())
                if not present:
                    continue

                # --- question IDF overlap ---
                q_total_idf = item_data.get("question", {}).get("total_idf", 0.0)
                if q_total_idf > 0:
                    q_matched_idf = sum(self.q_idf.get(ng, 0.0) for ng in cmatch.matched_ngs)
                    q_score = q_matched_idf / q_total_idf
                else:
                    # A one-item/all-identical reference has zero IDF everywhere.
                    # Exact overlap must still be detectable in that case.
                    q_ngrams = item_data.get("question", {}).get("ngrams", set())
                    q_score = len(cmatch.matched_ngs & q_ngrams) / len(q_ngrams) if q_ngrams else 0.0

                # early exit: question IDF below minimum (decon optimisation)
                if q_score < min_q_idf:
                    continue

                # second early exit: length-adjusted minimum threshold
                q_token_len = item_data.get("question", {}).get("token_count", 0)
                required_q_threshold = interpolate_threshold(q_token_len, min_q_idf)
                if q_score < required_q_threshold:
                    continue

                # --- cluster token boundaries ---
                doc_start = cmatch.start_idx
                doc_end_ng = cmatch.end_idx  # rightmost n-gram START position
                question_last_token = doc_end_ng + QUESTION_NGRAM_SIZE - 1
                cluster_token_len = question_last_token - doc_start + 1
                eval_q_len = item_data.get("question", {}).get("token_count", 0)
                excess = max(0, cluster_token_len - eval_q_len)
                question_end_adjusted = question_last_token - excess

                # --- answer IDF overlap ---
                field_scores: dict = {"question": q_score}

                if "answer" in item_data:
                    field_scores["answer"] = self._find_answer_idf(
                        doc_tokens, question_end_adjusted, item_idx,
                    )

                if "passage" in item_data:
                    field_scores["passage"] = self._find_passage_idf(
                        doc_tokens, doc_start, item_idx,
                    )

                # --- adaptive weights ---
                base_w = COMPONENT_WEIGHTS.get(frozenset(present))
                if base_w is None:
                    base_w = {f: 1.0 / len(present) for f in present}
                weights = adaptive_weights(base_w, item_data)

                # --- combined score (capped at 1.0) ---
                combined = min(
                    sum(weights.get(f, 0) * field_scores.get(f, 0) for f in present),
                    1.0,
                )

                # --- cumulative-length threshold ---
                cum_len = sum(item_data[f]["token_count"] for f in present)
                adj_threshold = interpolate_threshold(cum_len, threshold)

                if combined < adj_threshold:
                    continue

                # keep best match per reference item
                if item_idx in best_matches and best_matches[item_idx]["combined_score"] >= round(combined, 4):
                    continue

                best_field = max(field_scores, key=field_scores.get)
                best_matches[item_idx] = {
                    "ref_index": item_idx,
                    "ref_id": self._ref_id(item_idx),
                    "combined_score": round(combined, 4),
                    "best_field": best_field,
                    "best_field_score": round(field_scores[best_field], 4),
                    "field_scores": {k: round(v, 4) for k, v in field_scores.items()},
                    "threshold": round(adj_threshold, 4),
                    "cumulative_tokens": cum_len,
                }

        matches = sorted(best_matches.values(), key=lambda m: m["combined_score"], reverse=True)
        return matches

    def _ref_id(self, item_idx: int):
        item = self.items[item_idx]
        for key in ("instance_id", "id", "task_id", "task_name"):
            v = item.get(key)
            if isinstance(v, str) and v.strip():
                return v
        return None


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def _read_task_dir(path: Path) -> list:
    """Walk a directory of tasks. Two layouts supported:

      <dir>/<instance_id>/problem.md      (SWE visible tasks; gold stripped)
      <dir>/<instance_id>/instruction.md  (tb visible tasks; no solution/tests)

    Also accepts a flat directory of *.md / *.txt files (stem = instance id).
    """
    items: list = []
    subdirs = sorted([p for p in path.iterdir() if p.is_dir()])
    for sub in subdirs:
        text = None
        for fname in ("problem.md", "instruction.md", "problem_statement.md",
                      "problem.txt", "instruction.txt", "task.md"):
            fp = sub / fname
            if fp.is_file():
                text = fp.read_text(encoding="utf-8", errors="replace")
                break
        # tb: also fold in task.toml identity/instruction if present
        toml_fp = sub / "task.toml"
        toml_text = toml_fp.read_text(encoding="utf-8", errors="replace") if toml_fp.is_file() else ""
        if text is None and toml_text:
            text = toml_text
        elif text is not None and toml_text:
            text = text + "\n" + toml_text
        if text and text.strip():
            items.append({"instance_id": sub.name, "problem_statement": text})
    if items:
        return items
    # flat file layout
    for fp in sorted(path.iterdir()):
        if fp.is_file() and fp.suffix in (".md", ".txt"):
            text = fp.read_text(encoding="utf-8", errors="replace")
            if text.strip():
                items.append({"instance_id": fp.stem, "problem_statement": text})
    return items


def read_reference(path: Path) -> list:
    if path.is_dir():
        return _read_task_dir(path)
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if text.startswith("["):
        return json.loads(text)
    items = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            items.append(json.loads(line))
    return items


def document_text(obj, text_field=None):
    """Scan all string values, including nested messages, outputs and identity metadata.

    text_field remains a compatibility argument; it no longer excludes other fields.
    Empty/unsupported records fail closed instead of silently reducing coverage.
    """
    def strings(value):
        if isinstance(value, str):
            yield value
        elif isinstance(value, dict):
            for child in value.values():
                yield from strings(child)
        elif isinstance(value, list):
            for child in value:
                yield from strings(child)
    if not isinstance(obj, (dict, list, str)):
        raise ValueError("Training record must contain text")
    text = "\n".join(strings(obj)).strip()
    if not text:
        raise ValueError("Training record has no nonempty text")
    return text


def iter_jsonl_documents(stream):
    count = 0
    for line_no, line in enumerate(stream, 1):
        if not line.strip():
            continue
        try:
            text = document_text(json.loads(line))
        except (ValueError, TypeError, RecursionError):
            raise ValueError(f"Unreadable training record at line {line_no}") from None
        count += 1
        yield {"text": text, "raw": text, "source": f"line:{line_no}", "line": line_no}
    if not count:
        raise ValueError("No training records scanned")


def read_input_documents(input_file, input_format: str, text_field: str) -> list:
    # Patch format: parse the whole input as one or more unified diffs.
    if input_format == "patch":
        if input_file is None:
            raw = sys.stdin.read()
        else:
            raw = Path(input_file).read_text(encoding="utf-8", errors="replace")
        if not raw.strip():
            raise ValueError("No training records scanned")
        docs = split_patch_documents(raw)
        return docs if docs else [{"text": raw, "raw": raw, "source": "patch:0"}]

    if input_file is None:
        stream = sys.stdin
    else:
        stream = open(input_file)

    documents = []
    try:
        if input_format == "jsonl":
            return list(iter_jsonl_documents(stream))
        for line_no, line in enumerate(stream, 1):
            line = line.strip()
            if not line:
                continue
            if input_format == "text":
                documents.append({"text": line, "raw": line, "source": f"line:{line_no}"})

    finally:
        if input_file is not None:
            stream.close()
    if not documents:
        raise ValueError("No training records scanned")
    return documents


def score_document(index: ReferenceIndex, doc: dict, near_threshold: float, patch_aware: bool):
    """Return (matches, variant) for a document, taking the higher-scoring of
    the raw text and (if patch-like) its normalized-content variant."""
    raw = doc.get("text", "")
    matches = index.query(raw, near_threshold)
    variant = "text"
    if patch_aware and looks_like_diff(raw):
        norm = normalize_patch(raw)
        if norm.strip():
            pm = index.query(norm, near_threshold)
            pm_max = pm[0]["combined_score"] if pm else 0.0
            m_max = matches[0]["combined_score"] if matches else 0.0
            if pm_max > m_max:
                matches, variant = pm, "patch"
    return matches, variant


def print_summary(total_docs, contaminated_docs, near_docs, signature_docs,
                  total_matches, tolerance, verdict, file=sys.stderr):
    w = 54
    hr = "─" * w
    print("", file=file)
    print(f"┌{hr}┐", file=file)
    print(f"│{'Contamination Check Results':^{w}}│", file=file)
    print(f"├{hr}┤", file=file)
    print(f"│  Documents scanned {total_docs:>32}  │", file=file)
    print(f"│  Contaminated (>= 0.80) {contaminated_docs:>27}  │", file=file)
    print(f"│  Near-match band {near_docs:>34}  │", file=file)
    print(f"│  Identity-signature hits {signature_docs:>26}  │", file=file)
    print(f"│  Total item matches {total_matches:>31}  │", file=file)
    print("│  Policy: overlap requires provenance review          │", file=file)
    print(f"├{hr}┤", file=file)
    print(f"│  VERDICT: {verdict:<{w - 11}}│", file=file)
    print(f"└{hr}┘", file=file)
    print("", file=file)


def main():
    parser = argparse.ArgumentParser(
        description="Contamination detection for CoevoBench (decon SIMPLE core + patch/signature layer).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input", "-i", type=str, default=None,
                        help="Input file (JSONL / text / patch). Reads from stdin if omitted.")
    parser.add_argument("--input-format", choices=["jsonl", "text", "patch"], default="jsonl",
                        help="jsonl (default) | text (one doc per line) | patch (whole input is unified diff).")
    parser.add_argument("--text-field", default="text",
                        help="Compatibility option; all string fields are scanned.")
    parser.add_argument("--reference", "-r", type=str, required=True,
                        help="Reference: JSON array / JSONL file, OR a directory of task dirs "
                             "(problem.md / instruction.md).")
    parser.add_argument("--near-threshold", type=float, default=0.5,
                        help="Lower bound of the near-match distribution band (default 0.5; "
                             "contaminated bar is fixed at 0.80).")
    parser.add_argument("--tolerance", type=int, default=3,
                        help="Legacy display option; overlap always requires review.")
    parser.add_argument("--min-signature-len", type=int, default=8,
                        help="Min length for a bare instance id to count as an identity signature.")
    parser.add_argument("--no-patch-aware", action="store_true",
                        help="Disable diff-normalization fallback for diff-shaped documents.")
    parser.add_argument("--strict", action="store_true",
                        help="Compatibility flag; all overlap signals already exit with code 3.")
    parser.add_argument("--show-ref", action="store_true",
                        help="Include reference item text in output.")

    args = parser.parse_args()

    ref_path = Path(args.reference)
    if not ref_path.exists():
        print(f"Error: reference not found: {ref_path}", file=sys.stderr)
        sys.exit(2)

    print(f"Loading reference data from {ref_path}...", file=sys.stderr)
    ref_items = read_reference(ref_path)
    if not ref_items:
        print(f"Error: no reference items parsed from {ref_path}", file=sys.stderr)
        sys.exit(2)
    print(f"Building index over {len(ref_items)} reference items "
          f"(q_ngram={QUESTION_NGRAM_SIZE}, a_ngram={ANSWER_NGRAM_SIZE}, p_ngram={PASSAGE_NGRAM_SIZE}, "
          f"tokenizer={TOKENIZER_NAME if _USING_TIKTOKEN else 'whitespace-fallback'})...", file=sys.stderr)
    index = ReferenceIndex(ref_items, min_signature_len=args.min_signature_len)
    print(f"Index built: {len(index.q_ng_to_items)} unique question {QUESTION_NGRAM_SIZE}-grams, "
          f"{len(index.sig_to_item)} identity signatures.", file=sys.stderr)

    try:
        documents = read_input_documents(args.input, args.input_format, args.text_field)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(f"Checking {len(documents)} documents...", file=sys.stderr)

    patch_aware = not args.no_patch_aware
    total_matches = 0
    contaminated_docs = 0
    near_docs = 0
    signature_docs = 0

    for doc in documents:
        matches, variant = score_document(index, doc, args.near_threshold, patch_aware)
        sig_hits = index.scan_signatures(doc.get("raw", doc.get("text", "")))

        max_score = matches[0]["combined_score"] if matches else 0.0
        is_contaminated = max_score >= CONTAMINATION_THRESHOLD
        is_near = (not is_contaminated) and max_score >= args.near_threshold

        if not matches and not sig_hits:
            continue

        if sig_hits:
            signature_docs += 1
        if is_contaminated:
            contaminated_docs += 1
        elif is_near:
            near_docs += 1

        # Only n-gram matches that clear the contaminated bar are "matches";
        # near-band matches are reported under field but not counted as matches.
        counted = [m for m in matches if m["combined_score"] >= CONTAMINATION_THRESHOLD]
        total_matches += len(counted)

        result = {
            "source": doc["source"],
            "variant": variant,
            "max_score": max_score,
            "band": ("contaminated" if is_contaminated else "near" if is_near else "signature-only"),
            "num_matches": len(counted),
            "signature_hits": [{"signature": s, "ref_index": i, "ref_id": index._ref_id(i)}
                               for s, i in sig_hits],
            "matches": [],
        }
        for m in matches:
            entry = {
                "ref_index": m["ref_index"],
                "ref_id": m["ref_id"],
                "combined_score": m["combined_score"],
                "best_field": m["best_field"],
                "best_field_score": m["best_field_score"],
                "field_scores": m["field_scores"],
                "threshold": m["threshold"],
                "cumulative_tokens": m["cumulative_tokens"],
            }
            if args.show_ref:
                entry["ref_item"] = ref_items[m["ref_index"]]
            result["matches"].append(entry)

        print(json.dumps(result, ensure_ascii=False))

    # Similarity is a screening signal, never an automatic DQ or tolerated pass.
    if signature_docs or contaminated_docs or near_docs:
        verdict = "REVIEW_REQUIRED (overlap signal; inspect provenance)"
        exit_code = 3
    else:
        verdict = "NO_MATCHES (screening only)"
        exit_code = 0

    print_summary(len(documents), contaminated_docs, near_docs, signature_docs,
                  total_matches, args.tolerance, verdict)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
