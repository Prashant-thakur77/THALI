"""Render README.md from docs/README.template.md + results/*.json (plan §6; CLAUDE.md "The rule").

Every number in the README comes from a results file through a ``{{ expr }}`` placeholder evaluated here; a
missing file renders as "pending" rather than a made-up value.  Run ``python -m docs.render_readme``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"


def load(name: str) -> dict | None:
    p = R / name
    return json.loads(p.read_text()) if p.exists() else None


def skill_row(tag: str) -> str:
    """Per-skill policy-only success from results/skill_eval_<tag>.json, or 'pending'."""
    d = load(f"skill_eval_{tag}.json")
    if not d:
        return "pending"
    parts = []
    for k, v in d["skills"].items():
        cm = f" (median {v['median_zone_error_cm']} cm from zone)" if v.get("median_zone_error_cm") is not None else ""
        parts.append(f"{k} **{v['successes']}/{v['total']}**{cm}")
    return " · ".join(parts)


def smolvla_row() -> str:
    """Best and latest SmolVLA checkpoints (results/skill_eval_smolvla_<step>.json), each labelled with its training step."""
    files = sorted(R.glob("skill_eval_smolvla_*.json"), key=lambda p: int("".join(ch for ch in p.stem.split("_")[-1] if ch.isdigit()) or 0))
    if not files:
        return "pending"

    def step_of(p):
        tag = p.stem.split("_")[-1]
        n = int("".join(ch for ch in tag if ch.isdigit()) or 0)
        return n * 1000 if tag.endswith("k") else n

    def total(p):
        return sum(v.get("successes", 0) for v in json.loads(p.read_text())["skills"].values())

    best = max(files, key=total)
    latest = files[-1]
    out = f"best, step {step_of(best)}: " + skill_row(best.stem.replace("skill_eval_", ""))
    if latest != best:
        out += f" — latest, step {step_of(latest)}: " + skill_row(latest.stem.replace("skill_eval_", "")) + " (the last 6 000 steps ran on the RTX 3050 at batch 4 with a fresh optimizer after Kaggle's weekly GPU quota ran out, and lost ground; the step-14 000 checkpoint is kept on the Hub under `step_14000/`)"
    return out


def skill_rows_50k() -> str:
    """Every per-skill result trained at 50k steps (results/skill_eval_act_*50k*.json), merged; 'pending' if none."""
    best: dict[str, str] = {}
    for p in sorted(R.glob("skill_eval_act_*50k*.json"), key=lambda p: p.stat().st_mtime):   # newest measurement per skill wins
        d = json.loads(p.read_text())
        for k, v in d["skills"].items():
            if v.get("no_policy") or not v.get("total"):
                continue
            cm = f" (median {v['median_zone_error_cm']} cm from zone)" if v.get("median_zone_error_cm") is not None else ""
            best[k] = f"{k} **{v['successes']}/{v['total']}**{cm}"
    return " · ".join(best.values()) if best else "pending"


def anomaly_row() -> str:
    """Headline = the difference-image variant; the full-frame and crop variants are the ablation that motivated it."""
    d = load("anomaly_diffreal.json") or load("anomaly_diff.json")
    parts = []
    if d:
        lat = ", ".join(f"{k} {v['p50']} ms" for k, v in d["latency_ms"].items() if "p50" in v)
        op = d["at_10pct_fpr"]
        per = ", ".join(f"{k.replace('_', ' ')} {v['detected']}/{v['n']}" for k, v in d["per_type"].items())
        src = "nominal set from real expert runs (post-skill states)" if d.get("variant") == "diffreal" else "synthetic nominal set"
        parts.append(f"abs(frame − reset reference) crop, resnet18, {src}: image AUROC **{d['image_auroc']}** · {d['detected']}/{d['test_bad_total']} disturbances flagged with "
                     f"{d['false_positives']}/{d['test_good']} false alarms ({per}) · at 10% false alarms {op['detected']}/{d['test_bad_total']} · IR p50 {lat}")
    live = load("recovery_anomaly.json")
    if live:
        parts.append(f"**live, in the loop** (plate knocked mid-task, 10%-FPR threshold): flagged at the next check in {live['anomaly_flagged_after_knock']}/{live['total']} runs, "
                     f"{live['anomaly_false_alarms_before_knock']}/{live['total']} false alarms on clean steps, table nominal again after the redo")
    for name, label in (("anomaly.json", "full frame, wide_resnet50"), ("anomaly_crop.json", "table crop, resnet18"), ("anomaly_diff.json", "diff on the synthetic nominal set")):
        a = load(name)
        if a:
            parts.append(f"ablation {label}: AUROC {a['image_auroc']}, {a['false_positives']}/{a['test_good']} false alarms")
    return " — ".join(parts) if parts else "pending"


def followups_row() -> str:
    d = load("followups.json")
    if not d:
        return "pending"
    return f"**{d['correct']}/{d['cases']}** corrections resolved to the expected plan; {d['verifier_approved']}/{d['resolved']} resolved plans verifier-approved (the rest are refused with a reason, e.g. the other arm cannot reach)"


def clear_row() -> str:
    d = load("clear_table.json")
    if not d:
        return "pending"
    per = ", ".join(f"{k.replace('_', ' ')} {pct(v)}" for k, v in d["per_subgoal_rate"].items())
    return f"**{d['successes']}/{d['seeds']}** held-out seeds fully cleared ({per})"


def pour_amount_row() -> str:
    d = load("pour_amount.json")
    if not d:
        return "pending"
    return " · ".join(f"{k} (target {v['target']}): mean {v['mean_poured']} spheres, within ±2 in **{v['within_2']}/{v['n']}**, reached {v['reached_target']}/{v['n']}"
                      for k, v in d["summary"].items())


def concurrency_row() -> str:
    d = load("concurrency.json")
    if not d:
        return "pending"
    parts = []
    for name, c in d["commands"].items():
        s_ = c["summary"]
        parts.append(f"{name}: sequential {s_['sequential']['successes']}/{s_['sequential']['total']} in {s_['sequential']['mean_sim_steps']:.0f} sim steps → "
                     f"concurrent **{s_['concurrent']['successes']}/{s_['concurrent']['total']} in {s_['concurrent']['mean_sim_steps']:.0f}** ({100 * s_['sim_step_reduction']:.0f}% fewer)")
    return " · ".join(parts)


def pct(x: float | None) -> str:
    return "pending" if x is None else f"{100 * x:.0f}%"


def frac(d: dict | None, k1: str = "successes", k2: str = "seeds") -> str:
    return "pending" if not d else f"{d[k1]}/{d[k2]}"


def seeds_row(policy: str, mode: str, split: str = "test") -> str:
    d = load(f"seeds_{policy}_{mode}_{split}.json")
    if not d:
        return "pending SmolVLA run (docs/KAGGLE_TODO.md)" if policy == "smolvla" else "pending"
    won = d.get("skill_successes_won_by", {})
    return f"**{d['successes']}/{d['seeds']}** ({pct(d['success_rate'])}) — skills won by " + ", ".join(f"{k} {v}" for k, v in won.items())


def bench_table() -> str:
    b = load("bench.json")
    if not b:
        return "pending"
    lines = [f"_{b['caption']}_", "", "| precision / device | mean p50 ms over skills | skills |", "|---|---|---|"]
    for k, v in b["summary"].items():
        lines.append(f"| {k} | {v['mean_p50_ms']} | {v['skills']} |")
    lines.append("")
    lines.append("Per-skill rows in [results/bench.md](results/bench.md). Devices skipped: " + ", ".join(b["devices_skipped"]) + ".")
    return "\n".join(lines)


def preserve_table() -> str:
    p = load("preserve.json")
    if not p:
        return "pending"
    lines = [f"_{p['caption']}_", "", f"| precision ({p['device']}) | full-task successes / {p['seeds']} | Δ vs PyTorch |", "|---|---|---|"]
    for k, v in p["table"].items():
        lines.append(f"| {k} | {v['successes']} | {'' if v['delta_vs_torch'] is None else v['delta_vs_torch']:+d}".replace("+0", "0") + " |" if isinstance(v['delta_vs_torch'], int) else f"| {k} | {v['successes']} | — |")
    return "\n".join(lines)


def heat_table(policy: str) -> str:
    h = load(f"heatmap_{policy}.json")
    if not h:
        return "pending"
    cols = h["columns"]
    lines = ["| " + " | ".join(["seed"] + cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for s in range(h["seeds"]):
        lines.append(f"| {s} | " + " | ".join("✓" if h["matrix"][c][s] else "✗" for c in cols) + " |")
    lines.append("| **rate** | " + " | ".join(pct(h["per_axis_success_rate"][c]) for c in cols) + " |")
    return "\n".join(lines)


def swap_table() -> str:
    s = load("instruction_swap.json")
    if not s:
        return "pending"
    out = [f"**{s['correct']}/{s['total']}** variants encoded correctly ({s['from_vlm']} plans from the VLM, the rest from the rule fallback)."]
    for g in ("arm", "object"):
        m = s["matrix"][g]
        names = list(m)
        out += ["", f"{g} swaps — rows: requested, columns: what the plan encoded", "", "| | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
        for n in names:
            out.append(f"| {n} | " + " | ".join(str(m[n][c]) for c in names) + " |")
    out += ["", "order swaps: " + ", ".join(f"{x['variant']} {'✓' if x['correct'] else '✗'}" for x in s["order"])]
    return "\n".join(out)


CTX = {
    "load": load, "pct": pct, "frac": frac, "seeds_row": seeds_row, "bench_table": bench_table, "preserve_table": preserve_table,
    "heat_table": heat_table, "swap_table": swap_table, "skill_row": skill_row, "skill_rows_50k": skill_rows_50k, "smolvla_row": smolvla_row, "anomaly_row": anomaly_row, "concurrency_row": concurrency_row, "pour_amount_row": pour_amount_row, "clear_row": clear_row, "followups_row": followups_row, "json": json,
}


def render(template: str) -> str:
    def sub(m: re.Match) -> str:
        try:
            v = eval(m.group(1), CTX, CTX)  # comprehensions need the names as globals too
        except Exception as e:  # a missing key is a rendering bug, not a number to hide
            return f"⚠️ {e!r}"
        return "pending" if v is None else str(v)
    return re.sub(r"\{\{(.+?)\}\}", sub, template)


def main() -> None:
    for tpl, out in (("README.template.md", ROOT / "README.md"), ("EVIDENCE.template.md", ROOT / "docs" / "EVIDENCE.md"),
                     ("CHALLENGE_CHECKLIST.template.md", ROOT / "docs" / "CHALLENGE_CHECKLIST.md"), ("MODEL_CARD.template.md", ROOT / "docs" / "MODEL_CARD.md"), ("PITCH.template.md", ROOT / "docs" / "PITCH.md")):
        out.write_text(render((ROOT / "docs" / tpl).read_text()))
        print(f"{out.relative_to(ROOT)} rendered from results/")


if __name__ == "__main__":
    main()
