"""The four caption styles: definitions, rules and hand-tuned exemplars.

Exemplars share three scenarios across all styles so the blind-lineup test
measures pure style separability, not content differences. Each style has an
explicit anti-pattern list — the failure modes the LLM-Judge will punish
("sarcastic" that is merely declarative-with-attitude, "humorous" that is a
description with an exclamation mark).
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Shared exemplar fact sheets (what Stage 1 would have verified).
_FACTS_DOG = (
    "OBJECTS:\n- a brown dog\n- a yellow tennis ball\n- a shallow pond\n"
    "SETTING:\n- a public park with grass\nEVENTS:\n- the dog chases the thrown ball\n"
    "- the dog falls into the pond\n- the dog climbs out and shakes off water"
)
_FACTS_PANCAKE = (
    "OBJECTS:\n- a frying pan\n- a pancake\n- a stovetop\nSETTING:\n- a home kitchen\n"
    "EVENTS:\n- a person flips the pancake in the air\n- the pancake misses the pan\n"
    "- the pancake lands on the floor"
)
_FACTS_CYCLIST = (
    "OBJECTS:\n- a cyclist wearing a helmet\n- cars in traffic\n- a traffic light\n"
    "SETTING:\n- a busy city street\nEVENTS:\n- the cyclist weaves between car lanes\n"
    "- the traffic light turns red\n- the cyclist stops at the light"
)


@dataclass
class StyleConfig:
    key: str
    label: str
    description: str
    rules: list[str]
    anti_patterns: list[str]
    temperature: float
    exemplars: list[tuple[str, str]] = field(default_factory=list)  # (facts, caption)


STYLES: dict[str, StyleConfig] = {
    "formal": StyleConfig(
        key="formal",
        label="Formal",
        description=(
            "Precise, neutral, professional register. Reads like a news agency "
            "or archival description: complete sentences, no contractions, no "
            "opinion, no humor."
        ),
        rules=[
            "Complete grammatical sentences; no contractions or slang.",
            "Objective and neutral: report only what happens.",
            "Prefer precise verbs (e.g. 'proceeds', 'attempts') over casual ones.",
            "1-2 sentences, information-dense.",
        ],
        anti_patterns=[
            "Any joke, irony, or editorializing.",
            "Exclamation marks.",
        ],
        temperature=0.4,
        exemplars=[
            (_FACTS_DOG,
             "A brown dog pursues a tennis ball across a public park before falling "
             "into a shallow pond; it subsequently climbs out and shakes off the water."),
            (_FACTS_PANCAKE,
             "An individual attempts to flip a pancake on a domestic stovetop; the "
             "pancake misses the pan and lands on the kitchen floor."),
            (_FACTS_CYCLIST,
             "A helmeted cyclist manoeuvres between lanes of heavy urban traffic and "
             "comes to a stop when the traffic signal turns red."),
        ],
    ),
    "sarcastic": StyleConfig(
        key="sarcastic",
        label="Sarcastic",
        description=(
            "Dry, deadpan irony with a clear target. The words praise while the "
            "meaning mocks, or the delivery is conspicuously underwhelmed by "
            "obvious failure. Think eye-roll in text form."
        ),
        rules=[
            "Pick ONE target in the scene (the plan, the outcome, the confidence) and needle it.",
            "Use ironic praise or deadpan understatement — say the opposite of what is meant.",
            "Keep it dry: no LOL energy, no exclamation marks, no puns.",
            "The sarcasm must be unmissable when read blind.",
        ],
        anti_patterns=[
            "Plain description with a snide adjective bolted on.",
            "Enthusiastic or silly tone (that is humor, not sarcasm).",
        ],
        temperature=0.9,
        exemplars=[
            (_FACTS_DOG,
             "Truly elite ball-retrieval technique — right up to the part where the "
             "pond was involved. Nailed it."),
            (_FACTS_PANCAKE,
             "A flawless pancake flip, assuming the goal all along was to feed the floor."),
            (_FACTS_CYCLIST,
             "Bold of him to treat lane markings as decorative, only to be humbled by "
             "a small red light like the rest of us."),
        ],
    ),
    "humorous_tech": StyleConfig(
        key="humorous_tech",
        label="Humorous (Tech)",
        description=(
            "Comedy built on software/engineering culture: bugs, deploys, git, "
            "CI, servers, versioning. The scene is described through a tech "
            "metaphor a developer would grin at."
        ),
        rules=[
            "Anchor the joke in ONE recognizable tech concept (deploy, retry, exception, merge conflict...).",
            "The metaphor must map onto what actually happens in the clip.",
            "Playful, not dry — this should read as a joke, not irony.",
            "Jargon is the punchline, so use it accurately.",
        ],
        anti_patterns=[
            "Random tech words with no mapping to the scene.",
            "Deadpan mockery (that is sarcasm, not tech humor).",
        ],
        temperature=0.95,
        exemplars=[
            (_FACTS_DOG,
             "Fetch request successful, but the dog hit an unhandled pond exception "
             "mid-run. One full-body shake later, service was restored."),
            (_FACTS_PANCAKE,
             "The pancake flip worked perfectly in the pan environment but failed in "
             "production. Rolling back to the floor was not the intended deploy."),
            (_FACTS_CYCLIST,
             "Cyclist tries to bypass the traffic queue with some aggressive load "
             "balancing, then gets rate-limited by a red light."),
        ],
    ),
    "humorous_non_tech": StyleConfig(
        key="humorous_non_tech",
        label="Humorous (Non-Tech)",
        description=(
            "Everyday observational or absurdist comedy with zero technical "
            "vocabulary. The humor comes from timing, personification, or the "
            "gap between ambition and reality — sitcom-narrator energy."
        ),
        rules=[
            "Absolutely no tech vocabulary or internet-culture jargon.",
            "Personify objects or find the tiny human tragedy/comedy in the moment.",
            "Warm and playful, laughing with the subject.",
            "One clear comedic beat; land it and stop.",
        ],
        anti_patterns=[
            "Any word a software engineer would claim (glitch, update, loading...).",
            "Dry irony or mean-spirited mockery.",
        ],
        temperature=0.95,
        exemplars=[
            (_FACTS_DOG,
             "He wanted the ball. The pond, apparently, wanted him. Everyone in the "
             "park got something out of the deal."),
            (_FACTS_PANCAKE,
             "The pancake took one look at the ceiling, reconsidered its options, and "
             "decided the floor was more its scene."),
            (_FACTS_CYCLIST,
             "City cycling: one part sport, two parts negotiation, and in the end a "
             "tiny red light wins the argument anyway."),
        ],
    ),
}

STYLE_ORDER = list(STYLES.keys())
