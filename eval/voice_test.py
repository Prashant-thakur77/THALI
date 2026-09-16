"""Voice test on the four recorded samples (plan: "a short voice test on slurred/accented/Hindi samples with accuracy").

For each WAV in voice/test_samples/: Speechmatics realtime transcription (language from the file name:
hindi.wav -> "hi"), WER against transcripts.txt, the parsed intents, whether they match the intents parsed from
the ground truth, and the speech-end -> parse-ready latency.  Writes results/voice_test.json.

    python -m eval.voice_test
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path

from voice.listen import transcribe_file
from voice.parser import normalize, parse

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "voice" / "test_samples"


def _words(s: str) -> list[str]:
    return re.findall(r"[a-z0-9ऀ-ॿ']+", s.lower())


def wer(ref: str, hyp: str) -> float:
    r, h = _words(ref), _words(hyp)
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (r[i - 1] != h[j - 1]))
    return d[len(r)][len(h)] / max(1, len(r))


def intent_signature(text: str) -> list[tuple]:
    return [(i.kind, i.obj, i.arm) for i in parse(text).intents]


def main() -> None:
    truth = {}
    for line in (SAMPLES / "transcripts.txt").read_text().splitlines():
        if "\t" in line:
            name, text = line.split("\t", 1)
            truth[name.strip()] = text.strip()
    rows = []
    for name, ref in truth.items():
        lang = "hi" if "hindi" in name else "en"
        time.sleep(5)  # the free tier counts a just-closed session against concurrency for a few seconds
        t0 = time.time()
        utts, lst = asyncio.run(transcribe_file(SAMPLES / name, lang, focus="first"))
        hyp = " ".join(u.text for u in utts if u.is_operator)
        ref_sig, hyp_sig = intent_signature(ref), intent_signature(hyp)
        lat = [u.t_ready_wall - (u.t_first_partial_wall or u.t_ready_wall) for u in utts]
        rows.append({"file": name, "language": lang, "reference": ref, "hypothesis": hyp, "wer": round(wer(ref, hyp), 3),
                     "normalized_wer": round(wer(normalize(ref), normalize(hyp)), 3),  # both sides through the parser's normaliser (Devanagari/Hinglish -> keywords)
                     "utterances": len(utts), "speakers": sorted({u.speaker for u in utts if u.speaker}),
                     "operator": lst.operator, "ignored_other_speaker": [x["text"][:60] for x in lst.ignored],
                     "ref_intents": ref_sig, "hyp_intents": hyp_sig, "intents_match": ref_sig == hyp_sig,
                     "skills_match": [k for k, _, _ in ref_sig] == [k for k, _, _ in hyp_sig],
                     "wall_s": round(time.time() - t0, 2), "partial_to_ready_s": [round(x, 3) for x in lat]})
        print(f"{name:12s} wer {rows[-1]['wer']:.2f} norm-wer {rows[-1]['normalized_wer']:.2f} intents {'MATCH' if rows[-1]['intents_match'] else 'diff '} | {hyp[:80]}")
    out = {"samples": len(rows), "mean_wer": round(sum(r["wer"] for r in rows) / len(rows), 3),
           "mean_normalized_wer": round(sum(r["normalized_wer"] for r in rows) / len(rows), 3),
           "intent_match_rate": sum(r["intents_match"] for r in rows) / len(rows),
           "skill_match_rate": sum(r["skills_match"] for r in rows) / len(rows),
           "features": ["realtime", "partials", "end_of_utterance_0.6s", "speaker_diarization", "custom_vocab", "language_hi"], "rows": rows}
    (ROOT / "results" / "voice_test.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"mean WER {out['mean_wer']:.3f}, intent match {out['intent_match_rate']:.2f}, skill match {out['skill_match_rate']:.2f}")


if __name__ == "__main__":
    main()
