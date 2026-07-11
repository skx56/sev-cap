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
        temperature=0.2,
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
            "Needle ONE already-stated fact (the plan, confidence, or setting).",
            "Use ironic praise or deadpan understatement.",
            "Keep it dry: no LOL energy, no exclamation marks.",
            "Prefer one punchy sentence with no new nouns.",
        ],
        anti_patterns=[
            "Inventing motion outcomes (standstill, crash, blur) not in the facts.",
            "Plain description with a snide adjective bolted on.",
            "Enthusiastic or silly tone (that is humor, not sarcasm).",
        ],
        temperature=0.55,
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
            "metaphor mapped onto something visible — never invent extra plot."
        ),
        rules=[
            "Name at least one visible object or action from the facts.",
            "Use exactly ONE tech metaphor (deploy, leaf nodes, retry, bug, agent...).",
            "The metaphor must map onto what actually happens.",
            "Keep it to one short sentence.",
            "Do not invent secondary objects or failure modes.",
        ],
        anti_patterns=[
            "Inventing objects not in the facts (lettuce, cloud migrations, segfaults).",
            "Long multi-clause bug narratives.",
            "Deadpan mockery with no joke (that is sarcasm).",
        ],
        temperature=0.55,
        exemplars=[
            (_FACTS_DOG,
             "Fetch request hit an unhandled pond exception, then shake() flushed "
             "the buffer."),
            (_FACTS_PANCAKE,
             "Pancake flip passed staging, then missed the pan target in production."),
            (_FACTS_CYCLIST,
             "Lane-hopping tried to skip the queue, then got rate-limited by a red light."),
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
            "Absolutely no tech vocabulary.",
            "Personify subjects already in the facts, or understate the moment.",
            "Warm and playful, laughing with the subject.",
            "One clear comedic beat; land it and stop.",
            "No invented interactions (staring contests, secret plots).",
        ],
        anti_patterns=[
            "Any software jargon (glitch, deploy, stack...).",
            "Invented events not in the facts.",
            "Dry irony or mean-spirited mockery.",
        ],
        temperature=0.6,
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
