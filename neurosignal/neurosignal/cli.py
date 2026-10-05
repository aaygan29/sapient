"""Command-line interface:  neurosignal detect [--networks | --parcels | --image] ...

Examples:
  neurosignal detect --networks "Visual=1.0,Limbic=0.9,VentralAttention=0.2,Frontoparietal=0.3"
  neurosignal detect --parcels betas.npy --network-ids yeo7_ids.npy
  neurosignal detect --parcels betas.npy --schaefer          # auto-load atlas ([atlas] extra)
  neurosignal detect --image ad.png --model geo_subj05.pkl --repo /path/to/digital-brain --trusted
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from .detect import detect_from_networks
from .inputs.parcels import detect_from_parcels
from .types import DetectionResult


def _parse_networks(spec: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for pair in spec.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            sys.exit(f"bad --networks entry {pair!r}; use Name=value")
        k, v = pair.split("=", 1)
        out[k.strip()] = float(v)
    return out


def _run(args) -> DetectionResult:
    if args.networks:
        return detect_from_networks(_parse_networks(args.networks), source="manual network activations")
    if args.parcels:
        pa = np.load(args.parcels)
        if args.schaefer:
            from .inputs.fmri import detect_from_schaefer
            return detect_from_schaefer(pa)
        if not args.network_ids:
            sys.exit("--parcels requires --network-ids FILE.npy (or --schaefer to auto-load)")
        return detect_from_parcels(pa, np.load(args.network_ids))
    if args.image:
        if not args.model:
            sys.exit("--image requires --model geo_subjXX.pkl (and usually --repo to unpickle)")
        from .inputs.digital_brain import detect_from_image
        return detect_from_image(args.image, model_path=args.model, repo_path=args.repo, trusted=args.trusted)
    sys.exit("provide one of: --networks, --parcels, or --image")


def _print(res: DetectionResult, as_json: bool) -> None:
    if as_json:
        print(json.dumps(res.to_dict(), indent=2))
        return
    print(f"\n  BUY/SELL: {res.recommendation}  "
          f"(score {res.buy_sell_score}/100 · confidence {res.confidence} · coverage {int(res.coverage*100)}%)")
    print(f"  source: {res.source}\n")
    for c in res.constructs:
        cell = f"{c.score:5.1f}" if c.covered else "  n/a"
        proxy = "  (cortical proxy)" if c.cortical_proxy else ""
        print(f"   {c.label:30s} {cell}{proxy}")
    for n in res.notes:
        print("\n  note:", n)


def _print_analysis(a, as_json: bool) -> None:
    if as_json:
        print(json.dumps(a.to_dict(), indent=2))
        return
    print(f"\n  neuro-analysis  ·  modalities: {', '.join(a.modalities)}  ·  coverage {int(a.coverage*100)}%  ·  confidence {a.confidence}")
    print(f"  source: {a.source}\n")
    print("  METRICS")
    for m in a.metrics:
        print(f"   {m.label:28s} {m.score:6.1f}  [{m.unit:9s}] {m.interpretation}")
    print("\n  CONSTRUCTS (activation)")
    for c in a.constructs:
        cell = f"{c.score:5.1f}" if c.covered else "  n/a"
        print(f"   {c.label:30s} {cell}" + ("  (cortical proxy)" if c.cortical_proxy else ""))
    for n in a.notes:
        print("\n  note:", n)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        prog="neurosignal",
        description="Detect neuro-behavioral constructs + metrics (valence, arousal, manipulation, sycophancy, buy/sell).",
    )
    p.add_argument("command", nargs="?", default="detect", choices=["detect", "analyze"])
    p.add_argument("--text", help="text to analyze (analyze command); multimodal audio/video via the library API")
    p.add_argument("--encoder", default="reference", choices=["reference", "learned"], help="encoder backend")
    p.add_argument("--networks", help='inline Yeo-7 activations, e.g. "Visual=1.0,Limbic=0.9,VentralAttention=0.2"')
    p.add_argument("--parcels", help=".npy parcel activation (n_parcels,) or (T, n_parcels)")
    p.add_argument("--network-ids", help=".npy Yeo-7 network id per parcel")
    p.add_argument("--schaefer", action="store_true", help="auto-load Schaefer-1000 -> Yeo-7 ids ([atlas] extra)")
    p.add_argument("--image", help="image path -> Digital Brain visual-cortex encoder ([encoder] extra)")
    p.add_argument("--model", help="geo_subjXX.pkl path (for --image)")
    p.add_argument("--repo", help="digital-brain repo root (for --image unpickling)")
    p.add_argument("--trusted", action="store_true", help="allow unpickling the model (only for trusted files)")
    p.add_argument("--json", action="store_true", help="emit JSON")
    args = p.parse_args(argv)
    if args.command == "analyze":
        if not args.text:
            sys.exit("analyze requires --text (audio/video via the library API: neurosignal.analyze)")
        from .analyze import analyze
        _print_analysis(analyze(text=args.text, encoder=args.encoder), args.json)
        return
    _print(_run(args), args.json)


if __name__ == "__main__":
    main()
