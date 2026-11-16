"""M5/M1 outcome scoring for the ChemGraph workload (harness host, ChemGraph venv).

Compares each session's structured output with ChemGraph's public ground truth using
ChemGraph's own deterministic judge (chemgraph.eval.structured_output_judge).

    ~/agentfn-harness-venv/bin/python score_cg.py --run results/m5/cg_shared_k4_s1

The `--output-file` state format of `chemgraph run` was not available when this was
written; the extractor searches the saved JSON for the ResponseFormatter fields. Check
the first run's `unparsed` count; a non-zero count means the extractor needs adjusting.
"""

import argparse
import json
from pathlib import Path

from chemgraph.eval.structured_output_judge import judge_structured_output

FIELDS = {"smiles", "scalar_answer", "dipole", "vibrational_answer", "ir_spectrum", "atoms_data"}


def find_structured(obj):
    """Depth-first search for a dict that looks like a ResponseFormatter payload."""
    if isinstance(obj, dict):
        if len(FIELDS & set(obj)) >= 2:
            return obj
        for v in obj.values():
            hit = find_structured(v)
            if hit is not None:
                return hit
    elif isinstance(obj, list):
        for v in reversed(obj):  # latest message first
            hit = find_structured(v)
            if hit is not None:
                return hit
    elif isinstance(obj, str) and "{" in obj:
        try:
            return find_structured(json.loads(obj[obj.index("{"): obj.rindex("}") + 1]))
        except (ValueError, json.JSONDecodeError):
            return None
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    a = p.parse_args()
    run = Path(a.run)
    gt = {str(d["id"]): d for d in json.load(open(run / "groundtruth.json"))}
    ses = [json.loads(l) for l in open(run / "sessions.jsonl") if '"session"' in l]
    out = open(run / "cg_scores.jsonl", "w")
    unparsed = 0
    for s in ses:
        f = run / "traj" / f"{s['session']}.json"
        exp = gt[str(s["query_id"])]["answer"]["structured_output"]
        actual = None
        if f.exists():
            try:
                actual = find_structured(json.load(open(f)))
            except json.JSONDecodeError:
                actual = find_structured(f.read_text())
        if actual is None:
            unparsed += 1
            rec = {"session": s["session"], "parsed": False, "correct": False}
        else:
            j = judge_structured_output(exp, actual)
            rec = {"session": s["session"], "parsed": True, "correct": bool(j.get("score", j.get("correct", 0))),
                   "judge": j}
        out.write(json.dumps(rec, default=str) + "\n")
    print(f"scored {len(ses)} sessions, unparsed {unparsed}")


if __name__ == "__main__":
    main()
