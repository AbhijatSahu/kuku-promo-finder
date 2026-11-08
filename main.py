"""CLI Pipeline entrypoint for Kuku AI Promo Finder."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

# Automatically load environment variables from .env
load_dotenv(find_dotenv(), override=False)

from promo_finder.parser import parse_episode_full, export_parsed_episodes, generate_validation_report, Episode
from promo_finder.candidates import continuous_candidates, stitched_candidates, Candidate
from promo_finder.evaluate import deterministic_eval, llm_eval, get_api_key, MODEL_FALLBACKS, RUBRIC
from promo_finder.rank import validate, rank_candidates, export_html_report

def main() -> None:
    ap = argparse.ArgumentParser(description="Find, evaluate, and rank microdrama promo candidates.")
    ap.add_argument("--input", default="data/episodes", help="Folder containing .txt screenplay files")
    ap.add_argument("--output", default="outputs", help="Output directory for reports and evaluations")
    ap.add_argument("--min-seconds", type=float, default=20.0, help="Minimum total footage duration (default: 20)")
    ap.add_argument("--max-seconds", type=float, default=60.0, help="Maximum total footage duration (default: 60)")
    ap.add_argument("--max-segments", type=int, default=3, help="Maximum number of segments for stitched promos (default: 3)")
    ap.add_argument("--top", type=int, default=30, help="Number of top ranked promos to output (default: 30)")
    ap.add_argument("--continuous-only", action="store_true", help="Generate continuous candidates only (skip stitching)")
    ap.add_argument("--limit", type=int, default=0, help="Optional cap on generated candidates for quick smoke testing")
    ap.add_argument("--dry-run", action="store_true", help="Run deterministic heuristic pipeline without calling LLM APIs")
    ap.add_argument("--parse-only", action="store_true", help="Parse and validate screenplays only, exporting JSON reports without candidate generation")
    ap.add_argument("--llm-max-candidates", type=int, default=0, help="Max top unique candidates to evaluate with LLM (default: same as --top)")
    ap.add_argument("--model", default=None, help="Override LLM model name (default: PROMO_MODEL or gpt-4o-mini)")
    ap.add_argument("--require-llm", action="store_true", help="Fail instead of silently falling back to heuristics when the LLM is unavailable")
    args = ap.parse_args()

    files = sorted(Path(args.input).glob("*.txt"))
    if not files:
        raise SystemExit(f"No .txt files found in '{args.input}'. Please place screenplay files there.")

    print(f"=== Kuku AI Promo Finder ===")
    print(f"Reading {len(files)} screenplay files from {args.input}...")

    t0 = time.time()
    episodes: list[Episode] = []
    events_by_episode = {}
    valid_episodes = set()
    episode_durations = {}

    for f in files:
        try:
            ep = parse_episode_full(f)
            episodes.append(ep)
            events_by_episode[ep.episode_id] = ep.events
            valid_episodes.add(ep.episode_id)
            episode_durations[ep.episode_id] = ep.duration
        except Exception as exc:
            print(f"WARNING: Failed to parse {f.name}: {exc}")

    total_events = sum(len(e) for e in events_by_episode.values())
    dialogue_events = sum(sum(1 for ev in evs if ev.kind == 'dialogue') for evs in events_by_episode.values())
    print(f"Parsed {len(episodes)} episodes ({total_events} events, {dialogue_events} dialogue lines) in {time.time() - t0:.2f}s.")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Export parsed episode JSON and validation report
    parsed_json_path = out_dir / "parsed_episodes.json"
    val_rep_path = out_dir / "parser_validation_report.json"
    export_parsed_episodes(episodes, parsed_json_path)
    val_report = generate_validation_report(episodes, val_rep_path)

    total_anomalies = val_report["summary"]["total_anomalies"]
    print(f"Validation Report: {val_report['summary']['total_scenes']} scenes, {total_anomalies} anomalies detected.")
    if total_anomalies > 0:
        flagged = [ep for ep in val_report["episodes"] if ep["status"] == "FLAGGED"]
        print(f"  Flagged {len(flagged)} episodes with warnings/anomalies (details in {val_rep_path.name})")

    if args.parse_only:
        print(f"\nParse-only complete. Outputs written to {out_dir.resolve()}:")
        print(f"  - {parsed_json_path.name}")
        print(f"  - {val_rep_path.name}")
        return

    # 1. Candidate Generation
    print("Generating candidate promos...")
    candidates: list[Candidate] = []

    # Continuous candidates
    for ep in episodes:
        candidates.extend(
            continuous_candidates(
                ep.events,
                min_seconds=args.min_seconds,
                max_seconds=args.max_seconds,
                max_candidates_per_episode=30,
            )
        )
    cont_count = len(candidates)
    print(f"Generated {cont_count} continuous candidates.")

    # Stitched candidates (intra-episode and cross-episode)
    if not args.continuous_only:
        stitched = stitched_candidates(
            events_by_episode,
            min_seconds=args.min_seconds,
            max_seconds=args.max_seconds,
            max_segments=args.max_segments,
            max_windows_per_episode=6,
            allow_cross_episode=True,
        )
        candidates.extend(stitched)
        print(f"Generated {len(stitched)} stitched candidates (intra & cross-episode).")

    # 2. Strict Validation Gate
    valid_candidates: list[Candidate] = []
    invalid_count = 0
    for c in candidates:
        errs = validate(
            c,
            min_seconds=args.min_seconds,
            max_seconds=args.max_seconds,
            valid_episodes=valid_episodes,
            episode_durations=episode_durations,
        )
        if not errs:
            valid_candidates.append(c)
        else:
            invalid_count += 1

    candidates = valid_candidates
    print(f"Validated candidates: {len(candidates)} valid ({invalid_count} rejected).")

    if args.limit and args.limit > 0:
        candidates = candidates[:args.limit]
        print(f"Limited pool to {len(candidates)} candidates via --limit.")

    if not candidates:
        raise SystemExit("No valid promo candidates generated. Check duration bounds or input screenplays.")

    # 3. Two-Stage Evaluation
    has_api_key = bool(get_api_key())
    run_llm = has_api_key and not args.dry_run
    if args.require_llm and not args.dry_run and not has_api_key:
        raise SystemExit("ERROR: --require-llm was set but no API key is configured (PROMO_API_KEY).")

    evaluations: dict[str, dict] = {}

    if not run_llm:
        mode_label = "DRY RUN (Heuristic)" if args.dry_run else "Heuristic (no API key)"
        print(f"Evaluating {len(candidates)} candidates using {mode_label}...")
        t_eval = time.time()
        for idx, c in enumerate(candidates, 1):
            evaluations[c.candidate_id] = deterministic_eval(c)
            if idx % 500 == 0 or idx == len(candidates):
                print(f"Evaluated {idx}/{len(candidates)} candidates...")
        print(f"Heuristic evaluation completed in {time.time() - t_eval:.2f}s.")
        print(f"Ranking candidates and applying temporal IoU diversity filter (target top {args.top})...")
        ranked = rank_candidates(candidates, evaluations, top_n=args.top, diversity=True)
    else:
        chosen_model = args.model or os.getenv('PROMO_MODEL', 'gemini-3.5-flash-lite')
        effective_model = MODEL_FALLBACKS.get(chosen_model, chosen_model)

        # Step 1: Pre-ranking candidates with deterministic heuristic to extract top unique promos
        print(f"Step 1: Heuristic screening across all {len(candidates)} candidates...")
        t_pre = time.time()
        pre_evals = {c.candidate_id: deterministic_eval(c) for c in candidates}
        print(f"Heuristic screening completed in {time.time() - t_pre:.2f}s.")

        target_llm_count = args.llm_max_candidates if args.llm_max_candidates > 0 else args.top
        print(f"Applying temporal IoU diversity filter to select top {target_llm_count} unique promo candidates...")
        pre_ranked = rank_candidates(candidates, pre_evals, top_n=target_llm_count, diversity=True)
        top_ids = [r['candidate_id'] for r in pre_ranked]
        cand_map = {c.candidate_id: c for c in candidates}
        llm_targets = [cand_map[cid] for cid in top_ids if cid in cand_map]

        print(f"Step 2: Calling API to evaluate {len(llm_targets)} unique candidates with {effective_model}...")
        for i, c in enumerate(llm_targets, 1):
            evaluations[c.candidate_id] = llm_eval(c, model=effective_model)
            if i % 2 == 0 or i == len(llm_targets):
                print(f"API evaluated {i}/{len(llm_targets)} promos...")

        llm_ok = sum(1 for c in llm_targets if evaluations[c.candidate_id].get("evaluation_mode") == "llm")
        print(f"LLM evaluations succeeded: {llm_ok}/{len(llm_targets)} (others used heuristic fallback).")
        if args.require_llm and llm_targets and llm_ok == 0:
            reasons = {evaluations[c.candidate_id].get("reasoning", "") for c in llm_targets}
            raise SystemExit("ERROR: every LLM evaluation failed; refusing to report heuristic scores as LLM results.\n" + "\n".join(sorted(reasons)))

        # Step 3: Re-rank only the API-evaluated candidates based on final LLM scores
        print(f"Ranking final API-evaluated candidates (target top {args.top})...")
        llm_ranked = rank_candidates(llm_targets, evaluations, top_n=args.top, diversity=True)
        # Fill remaining slots with heuristic‑only candidates if needed
        missing = args.top - len(llm_ranked)
        if missing > 0:
            # IDs of candidates already evaluated by LLM
            llm_ids = {c.candidate_id for c in llm_targets}
            # Remaining candidates (not evaluated by LLM) still have deterministic scores in pre_evals
            remaining_candidates = [c for c in candidates if c.candidate_id not in llm_ids]
            heuristic_ranked = rank_candidates(remaining_candidates, pre_evals, top_n=missing, diversity=True)
            ranked = llm_ranked + heuristic_ranked
        else:
            ranked = llm_ranked

    # 5. Export Outputs
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_evals_path = out_dir / "all_evaluations.json"
    json_ranked_path = out_dir / "ranked_promos.json"
    csv_ranked_path = out_dir / "ranked_promos.csv"
    html_ranked_path = out_dir / "ranked_promos.html"

    json_evals_path.write_text(json.dumps(evaluations, ensure_ascii=False, indent=2), encoding="utf-8")
    json_ranked_path.write_text(json.dumps(ranked, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_fields = [
        "rank",
        "candidate_id",
        "title",
        "generation",
        "is_cross_episode",
        "duration_seconds",
        "raw_score",
        "adjusted_score",
        "hook_quote",
        "cliffhanger_quote",
        "reasoning",
        "risk",
        "evaluation_mode",
        "model_name",
        "scores",
        "segments",
        "transcript",
    ]

    with csv_ranked_path.open("w", newline="", encoding="utf-8-sig") as fp:
        writer = csv.DictWriter(fp, fieldnames=csv_fields)
        writer.writeheader()
        for i, r in enumerate(ranked, 1):
            row_data = {"rank": i}
            for k in csv_fields:
                if k == "rank":
                    continue
                val = r.get(k, "")
                if isinstance(val, (dict, list)):
                    row_data[k] = json.dumps(val, ensure_ascii=False)
                else:
                    row_data[k] = val
            writer.writerow(row_data)

    export_html_report(ranked, html_ranked_path)

    print(f"\nSUCCESS: Pipeline finished! Showing {len(ranked)} final evaluated results:\n")
    for idx, r in enumerate(ranked, 1):
        gen_type = "Cross-Episode" if r.get("is_cross_episode") else r.get("generation", "").capitalize()
        scores = r.get("scores", {})
        print(f"#{idx} [{r.get('raw_score', 0):.1f} pts] [{gen_type}] \"{r.get('title')}\" ({r.get('duration_seconds')}s)")
        print(f"    Mode: {r.get('evaluation_mode')} ({r.get('model_name')}) | Grounded: {r.get('quotes_grounded')}")
        print(f"    Rubric: Hook={scores.get('hook')}/10, Cliffhanger={scores.get('cliffhanger')}/10, Clarity={scores.get('standalone_clarity')}/10, Coherence={scores.get('narrative_coherence')}/10, Impact={scores.get('emotional_impact')}/10")
        print(f"    Hook: \"{r.get('hook_quote')}\"")
        print(f"    Cliffhanger: \"{r.get('cliffhanger_quote')}\"")
        print(f"    Rationale: {r.get('reasoning')}")
        if r.get('risk'):
            print(f"    Risk/Note: {r.get('risk')}")
        
        segs = r.get("segments", [])
        seg_strs = [f"{s.get('episode')}: {s.get('start'):.1f}s-{s.get('end'):.1f}s ({s.get('duration', 0):.1f}s)" for s in segs]
        print(f"    Segments ({len(segs)}): " + " -> ".join(seg_strs))
        print("-" * 80)

    print(f"\nOutputs written to {out_dir.resolve()}:")
    print(f"  - {json_ranked_path.name}")
    print(f"  - {csv_ranked_path.name}")
    print(f"  - {html_ranked_path.name}")
    print(f"  - {json_evals_path.name}")

if __name__ == "__main__":
    main()
