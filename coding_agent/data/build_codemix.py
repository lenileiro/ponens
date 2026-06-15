"""Build a CODE-HEAVY *continuation* manifest for autoregressive next-token training.

Proper fix for the chunk->chunk memorization collapse: each record is a contiguous
window of a single source file, with target == the NEXT window's text (stride == W), and
meta {source: category, chunk_index: monotonic}. This lets the trainer's `auto` inferer
detect a continuation manifest -> trains CAUSALLY (text+target as one next-token sequence),
and lets source-balanced sampling group by category. No more disjoint seq2seq pairs.
"""
import re, json, os, glob, random

TOK = re.compile(r"[a-z]+|[0-9]+|[^\sa-z0-9]")
W = 64                       # window; causal sequence length = 2*W = 128 tokens (fits a full tool transcript)
OUT = "/tmp/codemix/manifest.jsonl"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

# (source dir, target MB of text, category)  -- code-heavy mixture preserved
SOURCES = [
    ("/tmp/code", 45, "code"),          # raw code, many langs
    ("/tmp/agentdata", 28, "agent"),    # MCP / tool frameworks / agent code
    ("/tmp/toolfmt", 28, "tool"),       # bash-ReAct protocol transcripts (write/run/fix)
    ("/tmp/se", 22, "code_qa"),         # code Q&A
    ("/tmp/instruct", 14, "instruct_tool"),  # glaive tool-calls + alpaca/dolly
    ("/tmp/arxiv", 8, "tech_nl"),       # research NL
    ("/tmp/wikibooks", 5, "tech_nl"),   # textbooks
    ("/tmp/corpus1gb", 6, "general"),   # general English
]


def contiguous_text(src_dir, target_bytes):
    """Yield (filepath, contiguous_text_slice) per file, splitting the byte budget
    evenly across files. One CONTIGUOUS slice per file preserves next-token order."""
    files = sorted(glob.glob(os.path.join(src_dir, "*.txt")))
    if not files:
        return
    per = max(1, target_bytes // len(files))
    for fp in files:
        sz = os.path.getsize(fp)
        with open(fp, encoding="utf-8", errors="ignore") as f:
            if sz > per:                       # start past any license/header boilerplate
                f.seek(min(sz - per, sz // 10))
                f.readline()                   # align to a line boundary
            yield fp, f.read(per)


def main():
    rng = random.Random(0)
    rows = []
    cat_records = {}
    chunk_index = {}                           # per-category monotonic order
    for src, mb, cat in SOURCES:
        ci = chunk_index.get(cat, 0)
        n0 = len(rows)
        for _fp, text in contiguous_text(src, int(mb * 1e6)):
            toks = TOK.findall(text.lower())
            # stride W: text=toks[i:i+W], target=toks[i+W:i+2W]  => target == next.text
            for i in range(0, len(toks) - W, W):
                target = toks[i + W:i + 2 * W]
                if not target:
                    break
                rows.append({
                    "split": "eval" if ci % 50 == 0 else "train",
                    "text": toks[i:i + W],
                    "target": target,
                    "meta": {"source": cat, "chunk_index": ci},
                })
                ci += 1
        chunk_index[cat] = ci
        cat_records[cat] = cat_records.get(cat, 0) + (len(rows) - n0)
        print(f"  {src} ({cat}): {len(rows)-n0} windows", flush=True)
    rng.shuffle(rows)                          # on-disk shuffle is fine; auto re-groups by source+chunk_index
    with open(OUT, "w") as o:
        for r in rows:
            o.write(json.dumps(r) + "\n")
    tot = len(rows)
    print("\n=== mixture (by record count) ===")
    for c, n in sorted(cat_records.items(), key=lambda x: -x[1]):
        print(f"  {c:14} {n:9d}  {100*n/tot:5.1f}%")
    print(f"rows: {tot} | W={W} (causal seq {2*W}) | manifest MB: {os.path.getsize(OUT)/1e6:.0f}")


if __name__ == "__main__":
    main()
