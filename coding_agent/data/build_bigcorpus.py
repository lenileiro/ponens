"""Build a MULTI-GB code-heavy continuation manifest by STREAMING (low local RAM).
Same per-file contiguous-window causal format as build_codemix (W=64, source+chunk_index meta,
target==next.text so auto-detects causal), but with large byte targets and incremental writes
(no holding all rows in memory, no global shuffle — multimodal samples records randomly at
train time, so on-disk order doesn't matter). Output: /tmp/bigcorpus/manifest.jsonl

This manifest is too big for git-archive; ship it with launcher --multimodal-upload-manifest (scp).
"""
import re, json, os, glob

TOK = re.compile(r"[a-z]+|[0-9]+|[^\sa-z0-9]")
W = 64
OUT = "/tmp/bigcorpus/manifest.jsonl"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

# (dir, target MB of TEXT, category) — large targets; raw sources: code 429M, se 2.9G,
# agentdata 106M, instruct 273M, arxiv 144M, corpus1gb 959M, wikibooks 48M, toolfmt 40M
SOURCES = [
    ("/tmp/code", 350, "code"),
    ("/tmp/code2", 700, "code"),          # more diverse code (priority for a coding agent)
    ("/tmp/devtools", 200, "devtools"),   # bash/git/docker/pkg/build-test/multi-step tool mastery
    ("/tmp/toolfmt", 38, "tool"),         # write/run/fix protocol
    ("/tmp/se", 400, "code_qa"),
    ("/tmp/instruct", 150, "instruct_tool"),
    ("/tmp/agentdata", 100, "agent"),
    ("/tmp/arxiv", 80, "tech_nl"),
    ("/tmp/wikibooks", 40, "tech_nl"),
    ("/tmp/corpus1gb", 100, "general"),
]


def main():
    ci = {}
    total_rows = 0
    cat_rows = {}
    with open(OUT, "w") as o:
        for src, mb, cat in SOURCES:
            files = sorted(glob.glob(os.path.join(src, "*.txt")))
            if not files:
                print(f"  {src}: no files", flush=True); continue
            per = max(1, int(mb * 1e6) // len(files))
            c = ci.get(cat, 0); start = total_rows
            for fp in files:
                sz = os.path.getsize(fp)
                with open(fp, encoding="utf-8", errors="ignore") as f:
                    if sz > per:
                        f.seek(min(sz - per, sz // 10)); f.readline()
                    text = f.read(per)
                toks = TOK.findall(text.lower())
                for i in range(0, len(toks) - W, W):
                    tgt = toks[i + W:i + 2 * W]
                    if not tgt:
                        break
                    o.write(json.dumps({
                        "split": "eval" if c % 100 == 0 else "train",
                        "text": toks[i:i + W], "target": tgt,
                        "meta": {"source": cat, "chunk_index": c}}) + "\n")
                    c += 1; total_rows += 1
            ci[cat] = c
            cat_rows[cat] = cat_rows.get(cat, 0) + (total_rows - start)
            print(f"  {src} ({cat}): {total_rows - start} windows | running {total_rows}", flush=True)
    gb = os.path.getsize(OUT) / 1e9
    print(f"\nrows={total_rows} | W={W} (seq {2*W}) | manifest {gb:.2f} GB")
    for cat, n in sorted(cat_rows.items(), key=lambda x: -x[1]):
        print(f"  {cat:14} {n:9d}  {100*n/total_rows:5.1f}%")


if __name__ == "__main__":
    main()
