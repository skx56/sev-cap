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
            "opinion, no humor. Lead with specific visible details."
        ),
        rules=[
            "Complete grammatical sentences; no contractions or slang.",
            "Objective and neutral: report only what is visible.",
            "Name concrete attributes (colors, materials, subject type) when known.",
            "1-2 sentences, information-dense.",
        ],
        anti_patterns=[
            "Any joke, irony, or editorializing.",
            "Exclamation marks.",
            "Vague phrases like 'a scene' or 'various objects' with no specifics.",
        ],
        temperature=0.35,
        exemplars=[
            (_FACTS_DOG,
             "A brown dog pursues a yellow tennis ball across a grassy park before "
             "falling into a shallow pond, then climbs out and shakes off water."),
            (_FACTS_PANCAKE,
             "A person flips a pancake on a home stovetop; the pancake misses the pan "
             "and lands on the kitchen floor."),
            (_FACTS_CYCLIST,
             "A helmeted cyclist weaves between car lanes on a busy city street and "
             "stops when the traffic light turns red."),
        ],
    ),
    "sarcastic": StyleConfig(
        key="sarcastic",
        label="Sarcastic",
        description=(
            "Dry, deadpan irony that is still factually about the video. Praise "
            "that mocks, or understatement that makes the obvious look ridiculous. "
            "Short and sharp — still accurate to the scene."
        ),
        rules=[
            "Stay accurate to what is on screen; sarcasm is tone, not invention.",
            "Pick ONE target (the plan, the confidence, the setting) and needle it.",
            "Use ironic praise or deadpan understatement.",
            "Keep it dry: no LOL energy, no exclamation marks.",
            "Prefer one punchy sentence.",
        ],
        anti_patterns=[
            "Plain description with a snide adjective bolted on.",
            "Enthusiastic or silly tone (that is humor, not sarcasm).",
            "Hallucinated plot points not in the description.",
        ],
        temperature=0.85,
        exemplars=[
            (_FACTS_DOG,
             "Truly elite fetch technique — right up to the part where the pond "
             "joined the project."),
            (_FACTS_PANCAKE,
             "A flawless pancake flip, assuming the goal was always to feed the floor."),
            (_FACTS_CYCLIST,
             "Bold of him to treat lane markings as optional, until a tiny red light "
             "reminded him otherwise."),
        ],
    ),
    "humorous_tech": StyleConfig(
        key="humorous_tech",
        label="Humorous (Tech)",
        description=(
            "Tech/programming humor that still names real video details. One clear "
            "metaphor (deploy, bug, retry, stack, leaf nodes, agent, production) "
            "mapped onto something visible."
        ),
        rules=[
            "Name at least one visible object, character, or action from the clip.",
            "Anchor the joke in ONE recognizable tech concept.",
            "The metaphor must map onto what actually happens in the clip.",
            "Playful, not dry — this should read as a joke, not pure irony.",
            "One sentence is usually enough.",
        ],
        anti_patterns=[
            "Random tech words with no mapping to the scene.",
            "Deadpan mockery with no joke (that is sarcasm).",
            "Generic office jokes unrelated to what is on screen.",
        ],
        temperature=0.7,
        exemplars=[
            (_FACTS_DOG,
             "Fetch request launched on the tennis ball, hit an unhandled pond "
             "exception, then ran shake() to flush the buffer."),
            (_FACTS_PANCAKE,
             "Pancake flip passed in staging, but production missed the pan target "
             "and rolled back to the kitchen floor."),
            (_FACTS_CYCLIST,
             "Lane-hopping tried to skip the traffic queue, then got rate-limited by "
             "a red light."),
        ],
    ),
    "humorous_non_tech": StyleConfig(
        key="humorous_non_tech",
        label="Humorous (Non-Tech)",
        description=(
            "Everyday observational comedy with zero technical vocabulary. Warm, "
            "playful, still about the actual scene — sitcom-narrator energy."
        ),
        rules=[
            "Absolutely no tech vocabulary or programming jargon.",
            "Personify subjects or find the tiny human comedy in the moment.",
            "Warm and playful, laughing with the subject.",
            "One clear comedic beat; land it and stop.",
        ],
        anti_patterns=[
            "Any word a software engineer would claim (glitch, deploy, stack...).",
            "Dry irony or mean-spirited mockery.",
            "Generic jokes that ignore what is actually on screen.",
        ],
        temperature=0.9,
        exemplars=[
            (_FACTS_DOG,
             "He wanted the ball. The pond, apparently, wanted him."),
            (_FACTS_PANCAKE,
             "The pancake reconsidered the ceiling and decided the floor was more "
             "its scene."),
            (_FACTS_CYCLIST,
             "City cycling: one part sport, two parts negotiation, and a tiny red "
             "light still wins."),
        ],
    ),
}

STYLE_ORDER = list(STYLES.keys())
