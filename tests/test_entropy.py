import json

import pytest

from sevcap.entropy import verify_facts, check_grounding, FactSheet, VerifiedFact
from sevcap.extractor import Extraction


def ext(sample_id, **facts):
    return Extraction(sample_id=sample_id, facts=facts)


@pytest.mark.asyncio
async def test_high_support_fact_verified_low_support_rejected(fake_llm):
    # 5 samples: "a dog runs" in all 5 (worded differently), "a cat" in 1 only.
    extractions = [
        ext(0, actions=["a dog runs in the park"]),
        ext(1, actions=["a dog is running through a park"]),
        ext(2, actions=["a dog runs in the park"]),
        ext(3, actions=["a dog runs across the park"], objects=["a cat on a fence"]),
        ext(4, actions=["a dog running in a park"]),
    ]
    # actions category: 4 unique normalized strings -> one cluster [1,2,3,4]
    fake_llm.queue(json.dumps([[1, 2, 3, 4]]))
    # objects category: single lexical group, no LLM call needed
    sheet = await verify_facts(fake_llm, extractions, min_support=3)

    verified_texts = [f.text for f in sheet.verified]
    rejected_texts = [f.text for f in sheet.rejected]
    assert any("dog" in t for t in verified_texts)
    assert any("cat" in t for t in rejected_texts)
    dog = next(f for f in sheet.verified if "dog" in f.text)
    assert dog.support == 5
    assert sheet.semantic_entropy > 0


@pytest.mark.asyncio
async def test_cluster_judge_failure_falls_back_to_lexical(fake_llm):
    extractions = [
        ext(0, objects=["a red car"]),
        ext(1, objects=["a red car"]),
        ext(2, objects=["a crimson automobile"]),
    ]
    fake_llm.queue("not json at all !!!")
    sheet = await verify_facts(fake_llm, extractions, min_support=2)
    # lexical fallback: "a red car" (support 2) verified, paraphrase rejected
    assert [f.text for f in sheet.verified] == ["a red car"]
    assert [f.text for f in sheet.rejected] == ["a crimson automobile"]


@pytest.mark.asyncio
async def test_grounding_gate_flags_unsupported_claim(fake_llm):
    sheet = FactSheet(
        verified=[VerifiedFact(text="a dog runs in a park", category="actions", support=5, k=5)],
        rejected=[], semantic_entropy=0.0, k=5,
    )
    fake_llm.queue(json.dumps({"verdicts": [
        {"claim": 1, "supported": True},
        {"claim": 2, "supported": False},
    ]}))
    ok, unsupported = await check_grounding(
        fake_llm, sheet, "A dog runs in a park. It wears a blue Nike collar."
    )
    assert ok is False
    assert unsupported == ["It wears a blue Nike collar."]
