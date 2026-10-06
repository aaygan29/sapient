"""Fetch CNeuroMod Friends MOVIE STIMULI (.mkv) via git-annex from the PUBLIC CC0
remotes (s3unf / conp-ria-storage-http) — no DUA token needed (probe confirmed
5 public copies, e.g. s3unf.cneuromod.friends.stimuli).

The friends stimuli live in their OWN git-annex repo (subdataset) on the volume at
  /data/raw/algonauts2025/stimuli/movies/friends   (origin = courtois-neuromod/friends.stimuli)
The .mkv files are dangling annex symlinks until pulled. This runs
`git annex get <rel>` for the requested season/episode subset, materializing the
real bytes so the 6 stream extractors can decode them.

Idempotent: annex skips files already present. CPU only (cheap).

  modal run --detach data/fetch_friends_stimuli.py --season s1 --episodes 1-12
  modal run data/fetch_friends_stimuli.py::report --season s1
"""
from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path

import modal

vol = modal.Volume.from_name("sapient-data", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.11").apt_install(
    "git", "git-annex", "openssh-client", "curl", "wget", "ca-certificates"
)
app = modal.App("mary-fetch-friends-stimuli")

FRIENDS = "/data/raw/algonauts2025/stimuli/movies/friends"


def _materialized(p: Path) -> bool:
    try:
        return p.stat().st_size > 1_000_000
    except OSError:
        return False


def _ep_num(name: str) -> int | None:
    # friends_s01e03a.mkv -> 3
    m = re.search(r"e(\d+)[a-z]?\.mkv$", name)
    return int(m.group(1)) if m else None


@app.function(image=image, cpu=4, memory=8192, timeout=12 * 3600,
              volumes={"/data": vol})
def fetch(season: str = "s1", episodes: str = "1-12") -> dict:
    fr = Path(FRIENDS)
    sdir = fr / season
    if not sdir.is_dir():
        return {"error": f"no season dir {sdir}"}
    # git identity so annex's post-get state commit doesn't abort
    subprocess.run(["git", "config", "user.email", "mary@sapient.local"],
                   cwd=fr, check=False)
    subprocess.run(["git", "config", "user.name", "mary-data"], cwd=fr, check=False)

    lo, hi = (episodes.split("-") + [episodes])[:2]
    lo, hi = int(lo), int(hi)
    mkvs = sorted(sdir.glob("*.mkv"))
    todo = [p for p in mkvs
            if (_ep_num(p.name) is not None and lo <= _ep_num(p.name) <= hi
                and not _materialized(p))]
    print(f"[{season}] {len(mkvs)} segments, episodes {lo}-{hi}, "
          f"{len(todo)} to fetch", flush=True)
    got, failed = 0, []
    t0 = time.time()
    for i, p in enumerate(todo):
        rel = str(p.relative_to(fr))
        r = subprocess.run(["git", "annex", "get", rel], cwd=fr,
                           capture_output=True, text=True)
        ok = _materialized(p)
        if ok:
            got += 1
        else:
            failed.append({"file": rel, "rc": r.returncode,
                           "err": (r.stdout + r.stderr)[-300:]})
        print(f"  {i+1}/{len(todo)} got={got} {p.name} ok={ok}", flush=True)
        vol.commit()
    vol.commit()
    return {"season": season, "episodes": episodes,
            "n_segments_season": len(mkvs), "fetched_this_run": got,
            "n_failed": len(failed), "failed_sample": failed[:5],
            "secs": round(time.time() - t0, 1)}


@app.function(image=image, cpu=2, memory=4096, timeout=20 * 60,
              volumes={"/data": vol})
def report(season: str = "s1") -> dict:
    fr = Path(FRIENDS)
    out = {}
    for sd in sorted(fr.glob("s*")):
        if not sd.is_dir():
            continue
        mkvs = sorted(sd.glob("*.mkv"))
        mat = [p for p in mkvs if _materialized(p)]
        out[sd.name] = {"n_segments": len(mkvs), "n_materialized": len(mat),
                        "materialized": sorted(p.name for p in mat)[:60]}
    return out


@app.local_entrypoint()
def main(season: str = "s1", episodes: str = "1-12"):
    print(json.dumps(fetch.remote(season, episodes), indent=2))


@app.local_entrypoint()
def report_ep(season: str = "s1"):
    print(json.dumps(report.remote(season), indent=2))
