"""Inventory ds002345 (Narratives) BOLD coverage on the volume: per story, how many
subjects have a MATERIALIZED bold file (>1 MB), and per subject how many stories.
Tells us the real training pool. Read-only.  Run: modal run scripts/inventory_ds002345.py
"""
import os
import re
from collections import defaultdict

import modal

app = modal.App("mary-inventory-ds002345")
vol = modal.Volume.from_name("sapient-data")
TASK_RE = re.compile(r"task-([A-Za-z0-9]+)")


@app.function(volumes={"/data": vol}, timeout=600)
def inv():
    root = "/data/raw/ds002345"
    per_story = defaultdict(set)        # story -> set(subjects) with real bytes
    per_subject = defaultdict(set)      # subject -> set(stories) with real bytes
    pointer_only = defaultdict(int)     # story -> count of annex-pointer (<1MB) files
    subs = sorted(d for d in os.listdir(root) if d.startswith("sub-"))
    for sub in subs:
        func = os.path.join(root, sub, "func")
        if not os.path.isdir(func):
            continue
        for f in os.listdir(func):
            if not f.endswith("_bold.nii.gz"):
                continue
            m = TASK_RE.search(f)
            if not m:
                continue
            story = m.group(1)
            p = os.path.join(func, f)
            try:
                sz = os.path.getsize(p)
            except OSError:
                sz = 0
            if sz > 1_000_000:           # real BOLD (>1MB)
                per_story[story].add(sub)
                per_subject[sub].add(story)
            else:
                pointer_only[story] += 1
    print(f"Subjects scanned: {len(subs)}")
    print("\n=== Stories by #subjects with MATERIALIZED bold (>1MB) ===")
    for story, ss in sorted(per_story.items(), key=lambda kv: -len(kv[1])):
        print(f"  {story:24s} {len(ss):3d} subjects")
    print("\n=== Subjects with >=2 materialized stories (good for training) ===")
    multi = {s: st for s, st in per_subject.items() if len(st) >= 2}
    for s in sorted(multi, key=lambda s: -len(multi[s]))[:40]:
        print(f"  {s}: {len(multi[s])} stories -> {sorted(multi[s])}")
    print(f"\nSubjects with >=2 stories: {len(multi)} / {len(subs)}")
    tot = sum(len(v) for v in per_subject.values())
    print(f"Total materialized (subject,story) pairs: {tot}")
    print(f"Pointer-only (unmaterialized) by story (top): "
          f"{dict(sorted(pointer_only.items(), key=lambda kv:-kv[1])[:10])}")


@app.local_entrypoint()
def main():
    inv.remote()
