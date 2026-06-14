"""Build generic image preference-pair manifests from scored image manifests.

Preference-tuned image systems need a stable artifact between scoring and model
updates: prompt, chosen image, rejected image, and the score gap.  This module
creates that artifact from any manifest carrying generic `quality_score`,
`aesthetic`, `score`, or reward-model fields.  It does not know about a specific
reward model or prompt grammar.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
import tempfile

import numpy as np

from .image_score import DEFAULT_EXTERNAL_FIELDS, _optional_float, _stats, iter_manifest_rows


PAIR_MODES = ("top-bottom", "top-k", "all", "adjacent", "hard")
DEFAULT_GROUP_FIELDS = ("preference_group", "prompt_id", "prompt", "caption")


def _score_from_row(row, rec, score_field=""):
    fields = [score_field] if score_field else list(DEFAULT_EXTERNAL_FIELDS)
    for field in fields:
        if not field:
            continue
        val = _optional_float(row.get(field))
        if val is not None:
            return float(val), field
    if rec.aesthetic is not None:
        return float(rec.aesthetic), "aesthetic"
    return None, ""


def _field_value(row, rec, field):
    field = str(field)
    if field in ("caption", "text", "prompt"):
        return str(row.get(field) or rec.caption)
    if field in ("image", "path", "file", "filepath"):
        return str(row.get(field) or rec.path)
    if field == "split":
        return str(row.get("split") or rec.split)
    if field == "source":
        return str(row.get("source") or rec.source or "")
    return str(row.get(field) or "")


def _group_key(row, rec, group_by=""):
    if group_by:
        fields = [part.strip() for part in str(group_by).split(",") if part.strip()]
    else:
        fields = list(DEFAULT_GROUP_FIELDS)
    vals = []
    for field in fields:
        val = _field_value(row, rec, field).strip()
        if val:
            vals.append((field, val))
            if not group_by:
                break
    if not vals:
        vals.append(("caption", rec.caption))
    return " | ".join(f"{field}={value}" for field, value in vals)


def _manifest_image(path, root=""):
    if not root:
        return path
    abs_root = os.path.abspath(root)
    abs_path = os.path.abspath(path)
    try:
        rel = os.path.relpath(abs_path, abs_root)
    except ValueError:
        return path
    if rel.startswith("..") or os.path.isabs(rel):
        return path
    return rel


def _pair_row(group_key, chosen, rejected, score_field, root=""):
    chosen_row, rejected_row = chosen["row"], rejected["row"]
    chosen_rec, rejected_rec = chosen["record"], rejected["record"]
    prompt = (
        chosen_row.get("prompt")
        or chosen_row.get("caption")
        or rejected_row.get("prompt")
        or rejected_row.get("caption")
        or chosen_rec.caption
    )
    score_gap = float(chosen["score"] - rejected["score"])
    return {
        "prompt": str(prompt),
        "group_key": group_key,
        "chosen_image": _manifest_image(chosen_rec.path, root=root),
        "rejected_image": _manifest_image(rejected_rec.path, root=root),
        "chosen_caption": chosen_rec.caption,
        "rejected_caption": rejected_rec.caption,
        "chosen_score": float(chosen["score"]),
        "rejected_score": float(rejected["score"]),
        "score_gap": score_gap,
        "score_field": score_field or chosen["score_field"] or rejected["score_field"],
        "chosen_row": int(chosen["index"]),
        "rejected_row": int(rejected["index"]),
        "chosen_source": chosen_rec.source,
        "rejected_source": rejected_rec.source,
        "preference_source": "image_preferences_v1",
    }


def _candidate_pairs(rows, mode="top-bottom", min_gap=0.0):
    ordered = sorted(rows, key=lambda item: (-float(item["score"]), int(item["index"])))
    if len(ordered) < 2:
        return []
    min_gap = float(min_gap)
    pairs = []
    if mode == "top-bottom":
        pairs = [(ordered[0], ordered[-1])]
    elif mode == "top-k":
        pairs = [(ordered[0], row) for row in ordered[1:]]
    elif mode == "adjacent":
        pairs = list(zip(ordered[:-1], ordered[1:]))
    elif mode in ("all", "hard"):
        pairs = [
            (ordered[i], ordered[j])
            for i in range(len(ordered))
            for j in range(i + 1, len(ordered))
        ]
        if mode == "hard":
            pairs.sort(key=lambda pair: (
                abs(float(pair[0]["score"] - pair[1]["score"])),
                int(pair[0]["index"]),
                int(pair[1]["index"]),
            ))
    else:
        raise ValueError(f"unknown pair mode {mode!r}")
    return [
        (hi, lo)
        for hi, lo in pairs
        if float(hi["score"] - lo["score"]) >= min_gap
    ]


def build_preference_pairs(manifest, root="", split="", score_field="", group_by="",
                           mode="top-bottom", min_score_gap=0.0,
                           max_pairs_per_group=1, max_records=0, max_pairs=0,
                           out="", report_out=""):
    if mode not in PAIR_MODES:
        raise ValueError(f"unknown preference pair mode {mode!r}")
    groups = {}
    score_fields = Counter()
    missing_scores = 0
    records_seen = 0
    for item in iter_manifest_rows(manifest, root=root, split=split, max_records=max_records):
        records_seen += 1
        score, used_field = _score_from_row(item.row, item.record, score_field=score_field)
        if score is None:
            missing_scores += 1
            continue
        key = _group_key(item.row, item.record, group_by=group_by)
        score_fields[used_field] += 1
        groups.setdefault(key, []).append({
            "row": item.row,
            "record": item.record,
            "index": item.index,
            "score": float(score),
            "score_field": used_field,
        })

    out_rows = []
    gap_vals = []
    group_sizes = []
    skipped_singleton = 0
    skipped_no_gap = 0
    for key, rows in sorted(groups.items()):
        group_sizes.append(len(rows))
        if len(rows) < 2:
            skipped_singleton += 1
            continue
        pairs = _candidate_pairs(rows, mode=mode, min_gap=min_score_gap)
        if not pairs:
            skipped_no_gap += 1
            continue
        if max_pairs_per_group and max_pairs_per_group > 0:
            pairs = pairs[:int(max_pairs_per_group)]
        for chosen, rejected in pairs:
            out_rows.append(_pair_row(key, chosen, rejected, score_field, root=root))
            gap_vals.append(float(chosen["score"] - rejected["score"]))
            if max_pairs and len(out_rows) >= int(max_pairs):
                break
        if max_pairs and len(out_rows) >= int(max_pairs):
            break

    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            for row in out_rows:
                f.write(json.dumps(row) + "\n")

    report = {
        "experiment": "image_preference_pairs",
        "manifest": manifest,
        "root": root,
        "split": split,
        "score_field": score_field,
        "group_by": group_by or ",".join(DEFAULT_GROUP_FIELDS),
        "mode": mode,
        "min_score_gap": float(min_score_gap),
        "max_pairs_per_group": int(max_pairs_per_group),
        "max_records": int(max_records),
        "max_pairs": int(max_pairs),
        "out": out,
        "records_seen": int(records_seen),
        "records_with_score": int(records_seen - missing_scores),
        "records_missing_score": int(missing_scores),
        "score_fields": dict(sorted(score_fields.items())),
        "groups": int(len(groups)),
        "group_size_stats": _stats(group_sizes),
        "groups_skipped_singleton": int(skipped_singleton),
        "groups_skipped_min_gap": int(skipped_no_gap),
        "pairs_written": int(len(out_rows)),
        "score_gap_stats": _stats(gap_vals),
    }
    if report_out:
        os.makedirs(os.path.dirname(report_out) or ".", exist_ok=True)
        with open(report_out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1)
    return out_rows, report


def _read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def selftest():
    with tempfile.TemporaryDirectory() as td:
        manifest = os.path.join(td, "generated.jsonl")
        rows = [
            {"image": "p0_a.ppm", "caption": "red car", "prompt_id": "p0",
             "quality_score": 0.2},
            {"image": "p0_b.ppm", "caption": "red car", "prompt_id": "p0",
             "quality_score": 0.9},
            {"image": "p0_c.ppm", "caption": "red car", "prompt_id": "p0",
             "quality_score": 0.5},
            {"image": "p1_a.ppm", "caption": "blue boat", "prompt_id": "p1",
             "quality_score": 0.4},
            {"image": "p1_b.ppm", "caption": "blue boat", "prompt_id": "p1",
             "quality_score": 0.45},
            {"image": "missing.ppm", "caption": "single"},
        ]
        with open(manifest, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        out = os.path.join(td, "pairs.jsonl")
        pairs, report = build_preference_pairs(
            manifest, root=td, out=out, mode="top-bottom",
            min_score_gap=0.1, max_pairs_per_group=1)
        assert report["records_seen"] == 6
        assert report["records_missing_score"] == 1
        assert report["pairs_written"] == 1
        assert len(pairs) == 1
        assert pairs[0]["chosen_image"] == "p0_b.ppm"
        assert pairs[0]["rejected_image"] == "p0_a.ppm"
        assert pairs[0]["score_gap"] > 0.6
        reread = _read_jsonl(out)
        assert reread == pairs

        hard_pairs, hard_report = build_preference_pairs(
            manifest, root=td, mode="hard", min_score_gap=0.01,
            max_pairs_per_group=2)
        assert hard_report["pairs_written"] == 3
        assert hard_pairs[0]["group_key"].endswith("p0") or "prompt_id=p1" in hard_pairs[0]["group_key"]
        assert all(row["chosen_score"] > row["rejected_score"] for row in hard_pairs)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="", help="input scored image manifest")
    ap.add_argument("--root", default="", help="base directory for relative manifest images")
    ap.add_argument("--split", default="", help="optional split to read")
    ap.add_argument("--score-field", default="",
                    help="explicit score field; default searches quality/reward fields")
    ap.add_argument("--group-by", default="",
                    help=("comma-separated fields for prompt grouping; default tries "
                          "preference_group,prompt_id,prompt,caption"))
    ap.add_argument("--mode", default="top-bottom", choices=PAIR_MODES,
                    help="pair construction strategy inside each group")
    ap.add_argument("--min-score-gap", type=float, default=0.0,
                    help="minimum chosen-rejected score gap")
    ap.add_argument("--max-pairs-per-group", type=int, default=1,
                    help="cap pairs per group; 0 means unlimited")
    ap.add_argument("--max-records", type=int, default=0, help="cap input rows; 0 means all")
    ap.add_argument("--max-pairs", type=int, default=0, help="cap output pairs; 0 means all")
    ap.add_argument("--out", default="", help="preference-pair JSONL")
    ap.add_argument("--report-out", default="", help="JSON report")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        selftest()
        print("image_preferences selftest ok")
        return 0
    if not args.manifest:
        ap.error("--manifest is required unless --selftest is set")
    if not args.out and not args.report_out:
        ap.error("provide --out and/or --report-out")
    if args.max_pairs_per_group < 0:
        ap.error("--max-pairs-per-group must be non-negative")
    if args.max_records < 0:
        ap.error("--max-records must be non-negative")
    if args.max_pairs < 0:
        ap.error("--max-pairs must be non-negative")
    if args.min_score_gap < 0.0:
        ap.error("--min-score-gap must be non-negative")
    _pairs, report = build_preference_pairs(
        args.manifest, root=args.root, split=args.split, score_field=args.score_field,
        group_by=args.group_by, mode=args.mode, min_score_gap=args.min_score_gap,
        max_pairs_per_group=args.max_pairs_per_group, max_records=args.max_records,
        max_pairs=args.max_pairs, out=args.out, report_out=args.report_out)
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
