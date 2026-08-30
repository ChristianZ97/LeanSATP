#!/usr/bin/env python3
"""Distill the 92 verified v2 configs to minimal form via aesop?'s Try-this.

v4.26 aesop prints Try-this as annotated step lines `[kind]   (tactic)` — the
v1-era _distill_aesop.py parser predates this format (its substitutions
produced non-tactics like `[apply]   (nlinarith)`), so this targets the new
format directly. One aesop block per file, whole-tail replacement:

  1. NNN_<name>.lean --(aesop -> aesop?)--> capture `[kind] (tactic)` steps
  2. minimal candidate = statement head + step tactics, one per line
  3. lake-verify candidate with `#print axioms <name>` appended in the SAME
     run; accept iff rc==0, no sorry, axioms within the standard trio
     (the shipped file then drops the #print line — removing a trailing
     top-level command cannot un-verify the theorem above it)
  4. any failure -> fallback: verbatim config (itself verified/axiom-clean)

Outputs: minimal/<name>.lean + manifest.jsonl + summary on stdout."""
import json
import os
import re
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
# In-place default: this file ships at <repo>/proofs/minif2f_test_v2/.
LAKE = os.environ.get("SATP_LAKE_DIR") or os.path.abspath(f"{HERE}/../..")
OUT = f"{HERE}/minimal"
TR = f"{HERE}/traced"
STD = {"propext", "Classical.choice", "Quot.sound"}
os.makedirs(OUT, exist_ok=True)
os.makedirs(TR, exist_ok=True)

srcs = sorted(f for f in os.listdir(HERE) if re.match(r"^\d{3}_.*\.lean$", f)) or sorted(
    f"config/{f}" for f in os.listdir(f"{HERE}/config") if f.endswith(".lean"))
print(f"[distill] {len(srcs)} files", flush=True)


def parse_steps(log):
    """Extract the tactic sequence from v4.26 aesop?'s Try-this block(s).

    Shape: `  [kind]   <tactic>` opens a step; deeper-indented following
    lines are further tactics of the same step (linear sequence). Flatten
    everything to a col-2 sequence; re-join lines left bracket-unbalanced
    by pretty-printer wrapping. Verification arbitrates correctness."""
    steps, in_block = [], False
    for ln in log.splitlines():
        if ln.rstrip().endswith("Try this:"):
            in_block = True
            continue
        if not in_block:
            continue
        if not ln.strip():
            continue
        if re.match(r"^\S+.*:\d+:\d+:", ln):  # next lean message header
            in_block = False
            continue
        m = re.match(r"^\s*\[\w+\]\s+(.*)$", ln)
        t = (m.group(1) if m else ln).strip()
        if steps and (steps[-1].count("[") > steps[-1].count("]")
                      or steps[-1].count("(") > steps[-1].count(")")):
            steps[-1] += " " + t
        else:
            steps.append(t)
    return steps


def lake(path, timeout=300):
    try:
        p = subprocess.run(["lake", "env", "lean", path], cwd=LAKE,
                           capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout + p.stderr
    except subprocess.TimeoutExpired:
        return -1, "TIMEOUT"


def one(fn):
    src = open(f"{HERE}/{fn}").read()
    name = re.search(r"^\s*(?:theorem|lemma)\s+([^\s:({\[]+)", src, re.M).group(1)
    dst = f"{OUT}/{name}.lean"
    m = re.search(r"^\s*aesop \(config", src, re.M)
    if not m:
        open(dst, "w").write(src)
        return {"file": fn, "name": name, "status": "no_config_block"}
    head = src[:m.start()]
    traced = f"{TR}/{os.path.basename(fn)}"
    open(traced, "w").write(head + src[m.start():].replace("aesop (config", "aesop? (config", 1))
    rc, log = lake(traced)
    open(f"{traced}.log", "w").write(log)
    steps = parse_steps(log)
    reason = None
    if rc == 0 and steps:
        body = "  " + "\n  ".join(steps) + "\n"
        open(dst, "w").write(head + body + f"\n#print axioms {name}\n")
        vrc, vout = lake(dst)
        am = re.search(r"depends on axioms: \[([^\]]*)\]", vout)
        ax = {a.strip() for a in am.group(1).split(",") if a.strip()} if am else set()
        if vrc == 0 and "sorry" not in vout and am and not (ax - STD):
            open(dst, "w").write(head + body)
            return {"file": fn, "name": name, "status": "minimal", "steps": steps,
                    "axioms": sorted(ax), "bytes": (len(src), len(head + body))}
        reason = f"verify_failed(rc={vrc},axioms={sorted(ax - STD)})"
    else:
        reason = "no_steps" if rc == 0 else f"trace_failed(rc={rc})"
    open(dst, "w").write(src)
    return {"file": fn, "name": name, "status": "fallback", "reason": reason}


res = []
with ThreadPoolExecutor(12) as ex:
    for r in ex.map(one, srcs):
        res.append(r)
        tag = r["status"] if r["status"] != "minimal" else f"minimal({len(r['steps'])} steps)"
        print(f"[{len(res):2d}/{len(srcs)}] {r['name']:<45} {tag}"
              + (f"  <- {r['reason']}" if r.get("reason") else ""), flush=True)

# manifest_run.jsonl, not manifest.jsonl: the shipped manifest is the
# dataset-enriched view (uuid/dataset_index) and must survive re-runs.
with open(f"{HERE}/manifest_run.jsonl", "w") as f:
    for r in res:
        f.write(json.dumps(r) + "\n")

n_min = sum(r["status"] == "minimal" for r in res)
shrunk = [r for r in res if r["status"] == "minimal"]
tot_src = sum(r["bytes"][0] for r in shrunk)
tot_dst = sum(r["bytes"][1] for r in shrunk)
tac = Counter(s.split()[0].rstrip(";") for r in shrunk for s in r["steps"])
print(f"\n[distill] minimal {n_min}/{len(res)}  "
      f"(fallback {sum(r['status'] == 'fallback' for r in res)}, "
      f"no_config {sum(r['status'] == 'no_config_block' for r in res)})")
if shrunk:
    print(f"[distill] size on minimal subset: {tot_src} -> {tot_dst} bytes "
          f"(-{(1 - tot_dst / tot_src) * 100:.0f}%)")
print(f"[distill] head-tactic histogram: {dict(tac.most_common())}")
