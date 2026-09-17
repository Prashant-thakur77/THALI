"""ASR-tolerant command parser (ported from duet's src/lib/language, MIT), extended for Hindi/Hinglish.

Runs directly on Speechmatics transcript text, so it tolerates what live ASR produces: no punctuation,
disfluencies, "arm a" rendered as "arm eight"/"arm hey", filler openers, clauses joined by "and"/"then".
Output is a list of intents that the planner's rule backend and the VLM prompt both consume, plus
diagnostics for clauses that meant nothing (surfaced, never silently dropped).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

FILLERS = ["um", "uh", "erm", "er", "ah", "like", "you know", "i mean", "please", "okay", "ok", "so", "now", "just",
           "can you", "could you", "would you", "i want you to", "i'd like you to", "let's", "lets", "go ahead and",
           "robot", "thali", "sous chef", "souschef", "hey"]
OBJECTS = {"plate": "plate", "plates": "plate", "dish": "plate", "saucer": "plate", "thali": "plate",
           "mug": "mug", "mugs": "mug", "cup": "mug", "cups": "mug", "glass": "mug", "glaas": "mug",
           "fork": "fork", "forks": "fork", "kanta": "fork", "kaanta": "fork",
           "spoon": "spoon", "spoons": "spoon", "chamach": "spoon", "chammach": "spoon",
           "bottle": "bottle", "bottles": "bottle", "jug": "bottle", "pitcher": "bottle", "water": "bottle", "paani": "bottle", "pani": "bottle"}
ARM_TOKENS = {"a": "a", "eight": "a", "hey": "a", "ay": "a", "aye": "a", "first": "a", "one": "a", "left": "a", "ae": "a",
              "b": "b", "be": "b", "bee": "b", "bea": "b", "second": "b", "two": "b", "too": "b", "right": "b", "bi": "b"}
# Hinglish verbs / nouns that show up in Speechmatics' hi output (romanised) and in code-switched English sessions
HINDI = {"kholo": "open", "khol": "open", "kholna": "open", "uthao": "pick up", "utha": "pick up", "rakho": "place", "rakh": "place",
         "rakh do": "place", "daalo": "pour", "dalo": "pour", "dal": "pour", "pakdo": "hold", "pakad": "hold", "ruko": "stop", "rukho": "stop",
         "band": "stop", "table pe": "on the table", "table par": "on the table", "dabba": "drawer", "daraj": "drawer", "daraaz": "drawer",
         "se": "with", "aur": "and", "phir": "then", "ko": "", "mein": "into", "me": "into", "dheere": "gently", "thoda": "a little", "thora": "a little"}
# Devanagari output of a Speechmatics ``language="hi"`` session (plus the way it renders English loan words)
DEVANAGARI = {"टॉप": "top", "ऊपर": "top", "ड्राइवर": "drawer", "ड्रॉअर": "drawer", "ड्रावर": "drawer", "दराज": "drawer", "दराज़": "drawer",
              "खोलो": "open", "खोल": "open", "खोलिए": "open", "प्लेट": "plate", "थाली": "plate", "उठाओ": "pick up", "उठा": "pick up", "उठाइए": "pick up",
              "टेबल": "table", "मेज़": "table", "मेज": "table", "पर": "on", "पे": "on", "रख दो": "place", "रखो": "place", "रख": "place", "रखिए": "place",
              "पानी": "water", "डालो": "pour", "डाल": "pour", "डालिए": "pour", "भरो": "fill", "मग": "mug", "कप": "mug", "गिलास": "mug", "बोतल": "bottle",
              "चम्मच": "spoon", "चमच": "spoon", "कांटा": "fork", "काँटा": "fork", "कांटे": "fork", "रुको": "stop", "रोको": "stop", "बंद": "stop", "रुक": "stop",
              "धीरे": "gently", "थोड़ा": "a little", "थोड़ी": "a little", "हाथ": "arm", "बाएं": "left", "बायें": "left", "दाएं": "right", "दायें": "right",
              "आर्म": "arm", "ए": "a", "बी": "b", "से": "with", "को": "", "और": "and", "फिर": "then", "में": "into", "दूसरे": "other", "दूसरा": "other",
              "आराम से": "gently", "जारी": "continue", "सब": "everything", "पूरी": "whole", "लगाओ": "set", "सजाओ": "set"}
STOP_WORDS = ("stop", "halt", "wait", "freeze", "no no", "hold on", "ruko")
RESUME_WORDS = ("continue", "resume", "go on", "carry on", "aage badho")
OTHER_ARM = ("other arm", "the other one", "switch arms", "swap arms", "doosra haath", "dusra haath")
COMMAND_VERB = re.compile(r"\b(?:open|close|shut|clear|tidy|pick|grab|take|lift|fetch|grasp|place|put|drop|pour|fill|hand|pass|give|transfer|hold|keep|stop|halt|reset|set|lay)\b")


@dataclass
class Intent:
    kind: str                     # open_drawer | pick_place | handoff | hold_mug | pour | set_table | stop | resume | other_arm
    obj: str | None = None
    arm: str | None = None
    to_arm: str | None = None
    amount: str | None = None
    gentle: bool = False
    phrase: str = ""

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v not in (None, False, "")}


@dataclass
class ParseResult:
    intents: list[Intent]
    diagnostics: list[str] = field(default_factory=list)
    normalized: str = ""
    barge_in: str | None = None   # stop | resume | other_arm

    def as_dict(self) -> dict:
        return {"intents": [i.as_dict() for i in self.intents], "diagnostics": self.diagnostics, "normalized": self.normalized, "barge_in": self.barge_in}


def normalize(raw: str) -> str:
    s = raw.lower()
    s = s.replace("’", "'").replace("।", ".")
    for d, en in sorted(DEVANAGARI.items(), key=lambda kv: -len(kv[0])):
        s = s.replace(d, f" {en} ")
    for h, en in sorted(HINDI.items(), key=lambda kv: -len(kv[0])):
        s = re.sub(rf"\b{re.escape(h)}\b", en, s)
    s = re.sub(r"[.,!?;:]+", " ; ", s)
    for f in sorted(FILLERS, key=len, reverse=True):
        s = re.sub(rf"\b{re.escape(f)}\b", " ", s)
    # "arm eight" / "arm hey" / "arm be" -> "arm a" / "arm b"
    s = re.sub(r"\barm\s+(\w+)", lambda m: f"arm {ARM_TOKENS.get(m.group(1), m.group(1))}", s)
    return re.sub(r"\s+", " ", s).strip()


def split_clauses(text: str) -> list[str]:
    coarse = [c.strip() for c in re.split(r";|\b(?:then|after that|next|and then)\b", text) if c.strip()]
    out: list[str] = []
    for chunk in coarse:
        cuts = [m.start() for m in COMMAND_VERB.finditer(chunk) if m.start() > 0]
        # do not cut inside "pick up X and place it": a verb right after "and" continues the same command only
        # when its object is a pronoun ("it"), otherwise it is a new clause
        pieces, start = [], 0
        for cut in cuts:
            piece = chunk[start:cut].strip()
            rest = chunk[cut:]
            if piece.endswith("and") and re.match(r"(?:place|put|drop)\s+it\b", rest):
                continue
            # a verb with nothing nameable after it (verb-final Hindi/Hinglish, "...uthao") ends its clause
            if not re.search(r"\b(?:drawer|table|it|arm|" + "|".join(OBJECTS) + r")\b", rest.split(";")[0]):
                continue
            if piece:
                pieces.append(piece.rstrip(" and"))
            start = cut
        pieces.append(chunk[start:].strip())
        out.extend(p for p in pieces if p)
    return out


def detect_barge_in(text: str) -> str | None:
    """Matched against *partial* transcripts while a skill runs (plan 6.2)."""
    t = normalize(text)
    if any(w in t for w in OTHER_ARM):
        return "other_arm"
    if any(re.search(rf"\b{w}\b", t) for w in STOP_WORDS):
        return "stop"
    if any(w in t for w in RESUME_WORDS):
        return "resume"
    return None


def _arm(clause: str) -> str | None:
    m = re.search(r"\barm ([ab])\b", clause)
    if m:
        return m.group(1)
    if "left arm" in clause or "left hand" in clause:
        return "a"
    if "right arm" in clause or "right hand" in clause:
        return "b"
    return None


def _obj(clause: str) -> str | None:
    for w, o in OBJECTS.items():
        if re.search(rf"\b{w}\b", clause):
            return o
    return None


def parse(raw: str) -> ParseResult:
    norm = normalize(raw)
    res = ParseResult(intents=[], normalized=norm, barge_in=detect_barge_in(raw))
    if res.barge_in in ("stop", "resume", "other_arm") and len(norm.split()) <= 4:
        res.intents.append(Intent(res.barge_in, phrase=norm))
        return res
    gentle = bool(re.search(r"\b(gentl\w*|slow\w*|careful\w*|softly)\b", norm))
    for clause in split_clauses(norm):
        c = clause
        arm, obj = _arm(c), _obj(c)
        if re.search(r"\b(set|lay)\b.*\btable\b", c) or "everything" in c or "whole table" in c or "dinner" in c:
            res.intents.append(Intent("set_table", gentle=gentle, phrase=c))
        elif re.search(r"\b(clear|tidy|clean)\b.*\btable\b", c) or re.search(r"\b(put|pack)\b.*\b(away|back)\b", c):
            res.intents.append(Intent("clear_table", gentle=gentle, phrase=c))
        elif "drawer" in c and re.search(r"\b(close|shut|push)\b", c):
            res.intents.append(Intent("close_drawer", arm=arm, gentle=gentle, phrase=c))
        elif "drawer" in c and re.search(r"\b(open|pull|slide)\b", c):
            res.intents.append(Intent("open_drawer", arm=arm, gentle=gentle, phrase=c))
        elif re.search(r"\b(pour|fill)\b", c) or (obj == "bottle" and re.search(r"\binto\b", c)):
            amount = "little" if re.search(r"\b(little|bit|half|some)\b", c) else ("full" if re.search(r"\b(full|fill|brim|top)\b", c) else "normal")
            res.intents.append(Intent("pour", arm=arm, amount=amount, gentle=gentle, phrase=c))
        elif re.search(r"\b(hand|pass|give|transfer)\b", c):
            to = None
            m = re.search(r"\bto (?:the )?arm ([ab])\b", c)
            if m:
                to = m.group(1)
            res.intents.append(Intent("handoff", obj=obj, arm=arm, to_arm=to, gentle=gentle, phrase=c))
        elif re.search(r"\b(hold|keep)\b", c) and obj == "mug":
            res.intents.append(Intent("hold_mug", arm=arm, gentle=gentle, phrase=c))
        elif obj is not None and re.search(r"\b(pick|grab|take|lift|fetch|place|put|drop|move|lay|set|bring)\b", c):
            res.intents.append(Intent("pick_place", obj=obj, arm=arm, gentle=gentle, phrase=c))
        elif obj is not None and obj != "bottle":
            res.intents.append(Intent("pick_place", obj=obj, arm=arm, gentle=gentle, phrase=c))
        elif re.search(r"\b(place|put|drop)\b", c) and ("it" in c.split() or "table" in c) and obj is None:
            continue  # "place it on the table" after a pick: the pick already implies the place
        else:
            res.diagnostics.append(f"could not interpret: {clause!r}")
    return res


def intents_to_command(intents: list[Intent]) -> str:
    """Canonical English the rule planner understands, so the VLM and the rule fallback see the same request."""
    parts = []
    for i in intents:
        arm = f" with arm {i.arm.upper()}" if i.arm else ""
        if i.kind == "set_table":
            parts.append("set the table")
        elif i.kind == "open_drawer":
            parts.append(f"open the top drawer{arm}")
        elif i.kind == "clear_table":
            parts.append("clear the table")
        elif i.kind == "close_drawer":
            parts.append(f"close the drawer{arm}")
        elif i.kind == "pick_place":
            parts.append(f"put the {i.obj} on the table{arm}")
        elif i.kind == "handoff":
            parts.append(f"hand the {i.obj} from arm {(i.arm or 'a').upper()} to arm {(i.to_arm or 'b').upper()}")
        elif i.kind == "hold_mug":
            parts.append(f"hold the mug{arm}")
        elif i.kind == "pour":
            parts.append(("pour a little water" if i.amount == "little" else "fill the mug with water" if i.amount == "full" else "pour water") + f" into the mug{arm}")
        if i.gentle and "gently" not in parts[-1:]:
            parts[-1] += " gently"
    return ", ".join(parts)
