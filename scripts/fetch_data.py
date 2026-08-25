#!/usr/bin/env python3
"""Download HH-RLHF.

Pulls from the anthropics/hh-rlhf GitHub repository rather than the model hub, because
the GitHub copy is reachable from environments where huggingface.co is not. The helpful
subsets are stored in git-lfs and come from the media endpoint; harmless-base is stored
directly in the tree and comes from raw. Both are the same files the hub serves.
"""
import argparse, hashlib, gzip, os, sys, urllib.request

SUBSETS = ("helpful-base", "helpful-online", "helpful-rejection-sampled", "harmless-base")
LFS = "https://media.githubusercontent.com/media/anthropics/hh-rlhf/master/{sub}/{split}.jsonl.gz"
RAW = "https://raw.githubusercontent.com/anthropics/hh-rlhf/master/{sub}/{split}.jsonl.gz"
EXPECTED_ROWS = {
    ("helpful-base", "train"): 43835, ("helpful-base", "test"): 2354,
    ("helpful-online", "train"): 22007, ("helpful-online", "test"): 1137,
    ("helpful-rejection-sampled", "train"): 52421, ("helpful-rejection-sampled", "test"): 2749,
    ("harmless-base", "train"): 42537, ("harmless-base", "test"): 2312,
}


def fetch(url, dest):
    req = urllib.request.Request(url, headers={"User-Agent": "rm-robustness/0.1"})
    with urllib.request.urlopen(req, timeout=300) as r:
        body = r.read()
    if len(body) < 1024 or body[:2] != b"\x1f\x8b":
        return False
    with open(dest, "wb") as f:
        f.write(body)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--subsets", default=",".join(SUBSETS))
    a = ap.parse_args()
    raw = os.path.join(a.data_dir, "raw")
    os.makedirs(raw, exist_ok=True)
    ok = True
    for sub in a.subsets.split(","):
        for split in ("train", "test"):
            dest = os.path.join(raw, f"{sub}_{split}.jsonl.gz")
            if os.path.exists(dest) and os.path.getsize(dest) > 1024:
                print(f"have {dest}")
            else:
                got = False
                for tmpl in (LFS, RAW):
                    try:
                        got = fetch(tmpl.format(sub=sub, split=split), dest)
                        if got:
                            break
                    except Exception as e:
                        print(f"  {tmpl.split('/')[2]}: {e}", file=sys.stderr)
                if not got:
                    print(f"FAILED {sub}/{split}", file=sys.stderr)
                    ok = False
                    continue
                print(f"fetched {dest} ({os.path.getsize(dest)} bytes)")
            n = sum(1 for _ in gzip.open(dest, "rt", encoding="utf-8"))
            exp = EXPECTED_ROWS.get((sub, split))
            flag = "" if exp is None or n == exp else f"  !! expected {exp}"
            print(f"  {n} rows{flag}")
            if exp is not None and n != exp:
                ok = False
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
