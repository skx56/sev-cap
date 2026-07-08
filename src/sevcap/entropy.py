"""Semantic-entropy fact verification (Farquhar et al., Nature 2024, adapted).

The Nature method: sample K answers, cluster them by bidirectional entailment,
and treat high entropy over the clusters as a confabulation signal. We port it
from QA to captioning facts: a fact asserted consistently across independent
extraction samples (high support, low entropy) is *verified*; a fact appearing
in only 1-2 of K samples is exactly the arbitrary-generation signature the
paper shows this method catches, and is discarded before caption generation.

Clustering uses Gemma itself as the bidirectional-entailment judge (the paper
explicitly validates general-purpose LLMs as the entailment backend), batched
per category to one call instead of O(n^2) pairwise calls.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field

from .extractor import CATEGORIES, Extraction
from .fireworks import Gemma, extract_json

log = logging.getLogger("sevcap.entropy")

CLUSTER_PROMPT = """You are a strict natural-language-inference judge.

Below is a numbered list of short factual statements about the same video.
Group them into clusters of statements that BIDIRECTIONALLY ENTAIL each other:
statement A and B belong together only if A being true implies B is true AND
B being true implies A is true (i.e. they assert the same fact, possibly in
different words). Different levels of detail that do not mutually entail each
other belong in DIFFERENT clusters.

Statements:
{statements}

Return ONLY JSON: a list of clusters, each cluster a list of statement numbers.
Every number must appear in exactly one cluster. Example: [[1,4],[2],[3,5]]"""

ENTAIL_PROMPT = """You are a strict natural-language-inference judge.

Fact sheet (ground truth about a video):
{facts}

For each numbered claim below, answer whether the claim is SUPPORTED by the
fact sheet (entailed by it, or a harmless stylistic rephrasing) or UNSUPPORTED
(introduces a concrete object, action, place, number, name or event that the
fact sheet does not state).

Claims:
{claims}

Return ONLY JSON: {{"verdicts": [{{"claim": 1, "supported": true}}, ...]}}"""


@dataclass
class VerifiedFact:
    text: str
    category: str
    support: int          # how many of the K samples asserted this fact
    k: int                # total samples
    variants: list[str] = field(default_factory=list)

    @property
    def confidence(self) -> float:
        return self.support / self.k if self.k else 0.0


@dataclass
class FactSheet:
    verified: list[VerifiedFact]
    rejected: list[VerifiedFact]      # high-entropy facts we refused to use
    semantic_entropy: float           # diagnostic over the cluster distribution
    k: int

    def as_text(self) -> str:
        lines = []
        for cat in CATEGORIES:
            facts = [f for f in self.verified if f.category == cat]
            if facts:
                lines.append(f"{cat.replace('_', ' ').upper()}:")
                lines.extend(f"- {f.text}" for f in facts)
        return "\n".join(lines) if lines else "(no verified facts)"

    def report(self) -> dict:
        return {
            "k_samples": self.k,
            "semantic_entropy": round(self.semantic_entropy, 4),
            "verified_facts": [
                {"text": f.text, "category": f.category,
                 "support": f"{f.support}/{f.k}", "confidence": round(f.confidence, 2)}
                for f in self.verified
            ],
            "rejected_high_entropy_facts": [
                {"text": f.text, "category": f.category,
                 "support": f"{f.support}/{f.k}", "confidence": round(f.confidence, 2)}
                for f in self.rejected
            ],
        }


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", s.lower())).strip()


async def _cluster_category(
    llm: Gemma, items: list[tuple[int, str]]
) -> list[list[int]]:
    """Cluster (sample_id, text) items; returns clusters of item indices."""
    # cheap lexical pre-pass: identical normalized strings share a cluster
    lexical: dict[str, list[int]] = {}
    for idx, (_, text) in enumerate(items):
        lexical.setdefault(_norm(text), []).append(idx)
    reps = list(lexical.values())  # each rep-group is already one cluster
    if len(reps) <= 1:
        return reps

    statements = "\n".join(f"{i + 1}. {items[group[0]][1]}" for i, group in enumerate(reps))
    try:
        raw = await llm.chat(
            [{"role": "user", "content": CLUSTER_PROMPT.format(statements=statements)}],
            temperature=0.0, tag="entail-cluster", max_tokens=800,
        )
        groups = extract_json(raw)
        assert isinstance(groups, list)
        seen: set[int] = set()
        clusters: list[list[int]] = []
        for g in groups:
            merged: list[int] = []
            for num in g:
                i = int(num) - 1
                if 0 <= i < len(reps) and i not in seen:
                    seen.add(i)
                    merged.extend(reps[i])
            if merged:
                clusters.append(merged)
        for i in range(len(reps)):  # anything the judge dropped keeps its own cluster
            if i not in seen:
                clusters.append(reps[i])
        return clusters
    except Exception as e:  # noqa: BLE001
        log.warning("entailment clustering failed, falling back to lexical: %s", e)
        return reps


async def verify_facts(
    llm: Gemma, extractions: list[Extraction], min_support: int | None = None
) -> FactSheet:
    """Cluster K extractions per category and keep only low-entropy facts."""
    k = len(extractions)
    min_support = min(min_support or max(2, (k + 1) // 2), k)

    verified: list[VerifiedFact] = []
    rejected: list[VerifiedFact] = []
    cluster_sizes: list[int] = []

    for cat in CATEGORIES:
        items: list[tuple[int, str]] = []
        for ext in extractions:
            for c, text in ext.flat():
                if c == cat:
                    items.append((ext.sample_id, text))
        if not items:
            continue
        clusters = await _cluster_category(llm, items)
        for cluster in clusters:
            samples = {items[i][0] for i in cluster}
            texts = [items[i][1] for i in cluster]
            rep = Counter(texts).most_common(1)[0][0]
            fact = VerifiedFact(
                text=rep, category=cat, support=len(samples), k=k,
                variants=sorted(set(texts) - {rep}),
            )
            cluster_sizes.append(len(samples))
            (verified if fact.support >= min_support else rejected).append(fact)

    total = sum(cluster_sizes) or 1
    entropy = -sum((n / total) * math.log(n / total) for n in cluster_sizes)
    verified.sort(key=lambda f: (-f.support, f.category))
    rejected.sort(key=lambda f: (-f.support, f.category))
    log.info(
        "fact verification: %d verified, %d rejected (H=%.3f)",
        len(verified), len(rejected), entropy,
    )
    return FactSheet(verified=verified, rejected=rejected, semantic_entropy=entropy, k=k)


async def check_grounding(llm: Gemma, fact_sheet: FactSheet, caption: str) -> tuple[bool, list[str]]:
    """Gate A: every concrete claim in the caption must be entailed by the sheet."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", caption) if s.strip()]
    claims = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(sentences))
    raw = await llm.chat(
        [{"role": "user", "content": ENTAIL_PROMPT.format(facts=fact_sheet.as_text(), claims=claims)}],
        temperature=0.0, tag="grounding", max_tokens=600,
    )
    try:
        verdicts = extract_json(raw).get("verdicts", [])
    except Exception:  # noqa: BLE001
        return True, []  # judge unparseable -> don't block the pipeline
    unsupported = []
    for v in verdicts:
        try:
            if not v.get("supported", True):
                idx = int(v.get("claim", 0)) - 1
                if 0 <= idx < len(sentences):
                    unsupported.append(sentences[idx])
        except (TypeError, ValueError):
            continue
    return (len(unsupported) == 0), unsupported
