"""Language instructions for every skill: 10 paraphrases each, with arm/object slots (plan Phase 2.3).

Slots: {arm} -> "A"/"B" (also "left"/"right" synonyms), {obj} -> object name in words, {zone} -> place name.
The parser in voice/ has to map any of these back to the canonical skill call, so keep the vocabulary
here the vocabulary the planner and verifier also use.
"""

from __future__ import annotations

import random

OBJ_WORDS = {"plate": "the plate", "mug": "the mug", "bottle": "the bottle",
             "fork_1": "a fork", "fork_2": "a fork", "spoon_1": "a spoon", "spoon_2": "a spoon"}
ZONE_WORDS = {"plate": "the plate spot", "fork": "the left of the plate", "spoon": "the right of the plate", "mug": "the mug spot"}
ARM_WORDS = {"a": ["arm A", "the left arm", "arm a"], "b": ["arm B", "the right arm", "arm b"]}

PARAPHRASES: dict[str, list[str]] = {
    "open_drawer": [
        "open the top drawer with {arm}",
        "{arm}, pull the drawer open",
        "use {arm} to open the drawer",
        "slide the drawer out with {arm}",
        "{arm}: open the cutlery drawer",
        "pull open the top drawer using {arm}",
        "open the drawer, {arm}",
        "{arm} opens the drawer",
        "get the drawer open with {arm}",
        "with {arm}, open the top drawer",
    ],
    "pick_place": [
        "pick up {obj} with {arm} and place it on {zone}",
        "{arm}, put {obj} on {zone}",
        "move {obj} to {zone} using {arm}",
        "use {arm} to set {obj} down on {zone}",
        "take {obj} with {arm} and put it at {zone}",
        "{arm}: {obj} goes to {zone}",
        "grab {obj} using {arm}, then place it on {zone}",
        "with {arm}, lay {obj} on {zone}",
        "set {obj} on {zone}, {arm}",
        "{arm} picks {obj} and places it on {zone}",
    ],
    "handoff": [
        "hand {obj} from {arm} to {arm2}",
        "{arm}, pass {obj} to {arm2}",
        "transfer {obj} from {arm} to {arm2} via the table",
        "give {obj} to {arm2} using {arm}",
        "{arm} hands {obj} over to {arm2}",
        "pass {obj} across from {arm} to {arm2}",
        "let {arm2} take {obj} from {arm}",
        "move {obj} to {arm2}'s side with {arm}",
        "hand over {obj}: {arm} to {arm2}",
        "{arm} gives {obj} to {arm2}",
    ],
    "hold_mug": [
        "hold the mug steady with {arm}",
        "{arm}, pick up the mug and hold it for pouring",
        "use {arm} to hold the mug",
        "lift the mug with {arm} and keep it still",
        "{arm}: hold the mug up",
        "keep the mug steady using {arm}",
        "grab the mug with {arm} and hold it",
        "hold the mug in place, {arm}",
        "{arm} holds the mug for the pour",
        "with {arm}, hold the mug ready",
    ],
    "pour": [
        "pour water into the mug with {arm}",
        "{arm}, pour from the bottle into the mug",
        "use {arm} to pour a drink",
        "tip the bottle into the mug using {arm}",
        "{arm}: pour water",
        "fill the mug from the bottle with {arm}",
        "pour a little water into the mug, {arm}",
        "with {arm}, pour the bottle into the mug",
        "{arm} pours water into the mug",
        "pour some water using {arm}",
    ],
}


def instruction(skill: str, rng: random.Random, arm: str, obj: str | None = None, zone: str | None = None,
                arm2: str | None = None) -> str:
    """One random paraphrase of ``skill`` with its slots filled."""
    tmpl = rng.choice(PARAPHRASES[skill])
    return tmpl.format(arm=rng.choice(ARM_WORDS[arm]), arm2=rng.choice(ARM_WORDS[arm2]) if arm2 else "",
                       obj=OBJ_WORDS.get(obj, obj), zone=ZONE_WORDS.get(zone, zone))


def canonical(skill: str, arm: str, obj: str | None = None, zone: str | None = None, arm2: str | None = None) -> str:
    """The first paraphrase, used as the canonical instruction for eval."""
    return PARAPHRASES[skill][0].format(arm=ARM_WORDS[arm][0], arm2=ARM_WORDS[arm2][0] if arm2 else "",
                                        obj=OBJ_WORDS.get(obj, obj), zone=ZONE_WORDS.get(zone, zone))
