"""Follow-ups and corrections resolved against what the robot just did.

"again" / "once more"       -> repeat the last executed step
"a bit more" / "more water" -> pour again, a little (re-holding the mug with the other arm first if it was set down)
"no, the other side"        -> the last placed piece of cutlery goes to the mirrored zone (fork zone <-> spoon zone), by the
                               arm on that side, with a handoff when the piece is on the wrong side of the table
"the other arm"             -> repeat the last step with the arms swapped (a handoff keeps its zone)

The resolver only produces a plan; the verifier still checks it and the runtime executes it like any other plan.
Follow-ups are only accepted when a previous command exists (``Runtime.last_steps``), so a fresh "again" is refused.
"""

from __future__ import annotations

import re

from souschef_env import constants as C

MIRROR_ZONE = {"fork": "spoon", "spoon": "fork"}
ARM_OF_ZONE = {"fork": "a", "plate": "a", "spoon": "b", "mug": "b"}

AGAIN = re.compile(r"\b(again|once more|one more time|repeat( that)?|do that again)\b")
MORE = re.compile(r"\b(a bit more|a little more|some more|more water|top (it|me) up|keep pouring|more)\b")
OTHER_SIDE = re.compile(r"\b(other side|wrong side|opposite side|other way round|the other place)\b")
OTHER_ARM = re.compile(r"\b(other arm|the other hand|with arm ([ab]) instead)\b")


def detect(text: str) -> str | None:
    """Kind of follow-up in ``text`` (None when it is an ordinary command)."""
    t = text.lower().strip()
    if OTHER_SIDE.search(t):
        return "other_side"
    if OTHER_ARM.search(t):
        return "other_arm"
    if AGAIN.search(t) and len(t.split()) <= 6:   # before MORE: "one more time" is a repeat, not a top-up
        return "again"
    if MORE.search(t):
        return "more"
    return None


def resolve(kind: str, last_steps: list[dict], scene: dict | None = None) -> dict | None:
    """Plan for follow-up ``kind`` given the executed steps of the previous command (most recent last)."""
    if not last_steps:
        return None
    last = last_steps[-1]
    other = {"a": "b", "b": "a"}
    if kind == "again":
        return {"steps": [dict(last)], "mode": "normal"}
    if kind == "more":
        pours = [s for s in last_steps if s["skill"] == "pour"]
        if not pours:
            return None
        pour_arm = pours[-1]["arm"]
        hold_arm = other[pour_arm]
        mug_held = (scene or {}).get("objects", {}).get("mug", {}).get("held_by") == hold_arm
        steps = [] if mug_held else [{"skill": "hold_mug", "arm": hold_arm}]
        steps.append({"skill": "pour", "arm": pour_arm, "amount": "little"})
        steps.append({"skill": "place_mug", "arm": hold_arm})
        return {"steps": steps, "mode": "normal"}
    if kind == "other_side":
        placed = [s for s in last_steps if s["skill"] in ("pick_place", "handoff") and s.get("obj") in C.CUTLERY and s.get("zone") in MIRROR_ZONE]
        if not placed:
            return None
        s = placed[-1]
        obj, zone = s["obj"], MIRROR_ZONE[s["zone"]]
        from_arm = ARM_OF_ZONE[s["zone"]]          # the piece sits on the side it was placed: that arm can reach it
        to_arm = ARM_OF_ZONE[zone]
        if from_arm == to_arm:
            return {"steps": [{"skill": "pick_place", "arm": to_arm, "obj": obj, "zone": zone}], "mode": "normal"}
        return {"steps": [{"skill": "handoff", "arm": from_arm, "obj": obj, "to_arm": to_arm, "zone": zone}], "mode": "normal"}
    if kind == "other_arm":
        s = dict(last)
        if s["skill"] == "handoff":
            s["arm"], s["to_arm"] = s.get("to_arm", other[s["arm"]]), s["arm"]
        else:
            s["arm"] = other[s["arm"]]
        return {"steps": [s], "mode": "normal"}
    return None
