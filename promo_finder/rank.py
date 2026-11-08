"""Ranking, validation, temporal IoU deduplication, and rich report generation."""
from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any
from .evaluate import calculate_weighted_score, RUBRIC

def calculate_temporal_iou(cand_a: Any, cand_b: Any) -> float:
    """Calculate exact Intersection-over-Union (IoU) of selected footage across two candidates."""
    segs_a = cand_a.segments if hasattr(cand_a, 'segments') else cand_a.get('segments', [])
    segs_b = cand_b.segments if hasattr(cand_b, 'segments') else cand_b.get('segments', [])

    intersection_seconds = 0.0
    for sa in segs_a:
        sa_ep = sa.episode if hasattr(sa, 'episode') else sa.get('episode')
        sa_start = sa.start if hasattr(sa, 'start') else sa.get('start', 0.0)
        sa_end = sa.end if hasattr(sa, 'end') else sa.get('end', 0.0)

        for sb in segs_b:
            sb_ep = sb.episode if hasattr(sb, 'episode') else sb.get('episode')
            sb_start = sb.start if hasattr(sb, 'start') else sb.get('start', 0.0)
            sb_end = sb.end if hasattr(sb, 'end') else sb.get('end', 0.0)

            if sa_ep == sb_ep:
                inter = max(0.0, min(sa_end, sb_end) - max(sa_start, sb_start))
                intersection_seconds += inter

    dur_a = cand_a.duration if hasattr(cand_a, 'duration') else cand_a.get('duration_seconds', 0.0)
    dur_b = cand_b.duration if hasattr(cand_b, 'duration') else cand_b.get('duration_seconds', 0.0)

    union_seconds = dur_a + dur_b - intersection_seconds
    if union_seconds <= 0.0:
        return 0.0
    return intersection_seconds / union_seconds

def validate(
    candidate: Any,
    min_seconds: float = 20.0,
    max_seconds: float = 60.0,
    valid_episodes: set[str] | list[str] | None = None,
    episode_durations: dict[str, float] | None = None,
) -> list[str]:
    """Validate candidate duration, segment ordering, and boundary constraints."""
    errors: list[str] = []
    dur = candidate.duration if hasattr(candidate, 'duration') else candidate.get('duration_seconds', 0.0)
    segments = candidate.segments if hasattr(candidate, 'segments') else candidate.get('segments', [])

    # Tolerance for floating point precision
    eps = 0.01
    if dur < (min_seconds - eps) or dur > (max_seconds + eps):
        errors.append(f"duration {dur:.2f}s outside allowed range {min_seconds}-{max_seconds}s")

    if not segments:
        errors.append("candidate contains no segments")
        return errors

    # Check sum of segment durations equals candidate duration
    segment_dur_sum = sum(
        (s.duration if hasattr(s, 'duration') else (s.get('end', 0.0) - s.get('start', 0.0)))
        for s in segments
    )
    if abs(dur - segment_dur_sum) > eps:
        errors.append(f"candidate duration {dur:.2f}s does not match sum of segments {segment_dur_sum:.2f}s")

    valid_ep_set = set(valid_episodes) if valid_episodes else None

    # Validate each segment
    for idx, s in enumerate(segments):
        ep = s.episode if hasattr(s, 'episode') else s.get('episode')
        start = s.start if hasattr(s, 'start') else s.get('start', 0.0)
        end = s.end if hasattr(s, 'end') else s.get('end', 0.0)

        if end <= start:
            errors.append(f"segment {idx} end ({end:.2f}s) must be strictly after start ({start:.2f}s)")

        if start < 0.0:
            errors.append(f"segment {idx} start ({start:.2f}s) cannot be negative")

        if valid_ep_set and ep not in valid_ep_set:
            errors.append(f"segment {idx} episode '{ep}' is not in valid episodes manifest")

        if episode_durations and ep in episode_durations:
            ep_max_dur = episode_durations[ep]
            if end > ep_max_dur + 2.0:
                errors.append(f"segment {idx} end ({end:.2f}s) exceeds episode duration ({ep_max_dur:.2f}s)")

    # Validate that segments from the same episode do not overlap
    for i in range(len(segments) - 1):
        s1 = segments[i]
        s2 = segments[i + 1]
        ep1 = s1.episode if hasattr(s1, 'episode') else s1.get('episode')
        ep2 = s2.episode if hasattr(s2, 'episode') else s2.get('episode')
        end1 = s1.end if hasattr(s1, 'end') else s1.get('end', 0.0)
        start2 = s2.start if hasattr(s2, 'start') else s2.get('start', 0.0)

        if ep1 == ep2 and start2 < end1:
            errors.append(f"segments {i} and {i+1} in episode '{ep1}' overlap: [{end1:.2f}s > {start2:.2f}s]")

    return errors

def rank_candidates(
    candidates: list[Any],
    evaluations: dict[str, dict[str, Any]],
    top_n: int = 30,
    diversity: bool = True,
    iou_threshold: float = 0.70,
    rubric: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Rank candidates preserving raw scores while applying transparent diversity filtering."""
    rows: list[dict[str, Any]] = []

    for c in candidates:
        cand_dict = c.to_dict() if hasattr(c, 'to_dict') else dict(c)
        cand_id = cand_dict['candidate_id']
        ev = evaluations.get(cand_id, {})
        scores = ev.get('scores', {})

        # Compute raw weighted score
        raw_weighted = calculate_weighted_score(scores, rubric)

        rows.append({
            **cand_dict,
            'title': ev.get('title', 'Untitled promo'),
            'scores': scores,
            'raw_score': raw_weighted,
            'weighted_score': raw_weighted,
            'adjusted_score': raw_weighted,
            'hook_quote': ev.get('hook_quote', ''),
            'cliffhanger_quote': ev.get('cliffhanger_quote', ''),
            'reasoning': ev.get('reasoning', ''),
            'risk': ev.get('risk', ''),
            'evaluation_mode': ev.get('evaluation_mode', 'heuristic'),
            'model_name': ev.get('model_name', 'heuristic-rules-v1'),
            'quotes_grounded': ev.get('quotes_grounded', True),
        })

    # Sort descending by raw score
    rows.sort(key=lambda r: r['raw_score'], reverse=True)

    if not diversity:
        return rows[:top_n]

    # Diversity selection using exact temporal IoU
    chosen: list[dict[str, Any]] = []
    for candidate in rows:
        # Check IoU overlap against already chosen candidates
        is_duplicate = False
        for old in chosen:
            iou = calculate_temporal_iou(candidate, old)
            if iou >= iou_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            chosen.append(candidate)
            if len(chosen) >= top_n:
                break

    return chosen

def export_html_report(ranked: list[dict[str, Any]], output_path: str | Path, title: str = "Kuku AI Promo Finder — Ranked Promos Report") -> None:
    """Generate an interactive, human-readable HTML report of ranked promo candidates."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    cards_html = []
    for idx, p in enumerate(ranked, 1):
        scores = p.get('scores', {})
        segments = p.get('segments', [])
        seg_rows = []
        for s in segments:
            seg_rows.append(
                f"<tr><td><code>{html.escape(str(s.get('episode', '')))}</code></td>"
                f"<td>{s.get('start', 0.0):.2f}s</td>"
                f"<td>{s.get('end', 0.0):.2f}s</td>"
                f"<td>{s.get('duration', 0.0):.2f}s</td></tr>"
            )
        seg_table = (
            "<table class='seg-table'><thead><tr><th>Episode</th><th>In</th><th>Out</th><th>Duration</th></tr></thead><tbody>"
            + "".join(seg_rows)
            + "</tbody></table>"
        )

        gen_badge = f"<span class='badge badge-{p.get('generation', 'continuous')}'>{p.get('generation', '').upper()}</span>"
        mode_badge = f"<span class='badge badge-mode'>{html.escape(p.get('evaluation_mode', 'heuristic')).upper()} ({html.escape(p.get('model_name', ''))})</span>"
        cross_badge = "<span class='badge badge-cross'>CROSS-EPISODE</span>" if p.get('is_cross_episode') else ""

        card = f"""
        <div class="promo-card">
            <div class="card-header">
                <div class="rank-title">
                    <span class="rank-num">#{idx}</span>
                    <h3 class="promo-title">{html.escape(p.get('title', 'Untitled Promo'))}</h3>
                </div>
                <div class="badges">
                    {gen_badge}
                    {cross_badge}
                    {mode_badge}
                </div>
            </div>
            <div class="score-grid">
                <div class="score-box main-score">
                    <div class="score-label">WEIGHTED SCORE</div>
                    <div class="score-val">{p.get('raw_score', 0.0):.1f}</div>
                </div>
                <div class="score-box">
                    <div class="score-label">Hook (25%)</div>
                    <div class="score-val">{scores.get('hook', 0)}/10</div>
                </div>
                <div class="score-box">
                    <div class="score-label">Cliffhanger (25%)</div>
                    <div class="score-val">{scores.get('cliffhanger', 0)}/10</div>
                </div>
                <div class="score-box">
                    <div class="score-label">Clarity (20%)</div>
                    <div class="score-val">{scores.get('standalone_clarity', 0)}/10</div>
                </div>
                <div class="score-box">
                    <div class="score-label">Coherence (20%)</div>
                    <div class="score-val">{scores.get('narrative_coherence', 0)}/10</div>
                </div>
                <div class="score-box">
                    <div class="score-label">Emotion (10%)</div>
                    <div class="score-val">{scores.get('emotional_impact', 0)}/10</div>
                </div>
            </div>
            <div class="quotes-section">
                <div class="quote-box">
                    <strong>Opening Hook Quote:</strong> <em>"{html.escape(p.get('hook_quote', ''))}"</em>
                </div>
                <div class="quote-box">
                    <strong>Closing Cliffhanger Quote:</strong> <em>"{html.escape(p.get('cliffhanger_quote', ''))}"</em>
                </div>
            </div>
            <div class="details-section">
                <p><strong>Editorial Rationale:</strong> {html.escape(p.get('reasoning', ''))}</p>
                <p><strong>Context/Spoiler Risk:</strong> <span class="risk-text">{html.escape(p.get('risk', ''))}</span></p>
            </div>
            <div class="segments-section">
                <h4>Footage Segments (Total Selected Duration: {p.get('duration_seconds', 0.0):.2f}s)</h4>
                {seg_table}
            </div>
            <details class="transcript-details">
                <summary>View Complete Assembled Transcript</summary>
                <pre class="transcript-box">{html.escape(p.get('transcript', ''))}</pre>
            </details>
        </div>
        """
        cards_html.append(card)

    template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html.escape(title)}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 24px; }}
        .container {{ max-width: 1100px; margin: 0 auto; }}
        header {{ margin-bottom: 24px; border-bottom: 1px solid #334155; padding-bottom: 16px; }}
        h1 {{ margin: 0 0 8px 0; font-size: 28px; color: #38bdf8; }}
        p.subtitle {{ margin: 0; color: #94a3b8; font-size: 15px; }}
        .promo-card {{ background: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 20px; margin-bottom: 24px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }}
        .card-header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #334155; padding-bottom: 12px; margin-bottom: 16px; }}
        .rank-title {{ display: flex; align-items: center; gap: 12px; }}
        .rank-num {{ background: #38bdf8; color: #0f172a; font-weight: bold; border-radius: 6px; padding: 4px 10px; font-size: 16px; }}
        .promo-title {{ margin: 0; font-size: 20px; color: #f1f5f9; }}
        .badges {{ display: flex; gap: 8px; }}
        .badge {{ font-size: 11px; font-weight: bold; padding: 4px 8px; border-radius: 4px; text-transform: uppercase; }}
        .badge-continuous {{ background: #0284c7; color: white; }}
        .badge-stitched {{ background: #7c3aed; color: white; }}
        .badge-cross {{ background: #e11d48; color: white; }}
        .badge-mode {{ background: #334155; color: #cbd5e1; }}
        .score-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 12px; margin-bottom: 16px; }}
        .score-box {{ background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 10px; text-align: center; }}
        .score-box.main-score {{ background: #0c4a6e; border-color: #0284c7; }}
        .score-box.main-score .score-val {{ color: #38bdf8; font-size: 24px; }}
        .score-label {{ font-size: 11px; text-transform: uppercase; color: #94a3b8; margin-bottom: 4px; }}
        .score-val {{ font-size: 18px; font-weight: bold; color: #f8fafc; }}
        .quotes-section {{ background: #0f172a; border-left: 4px solid #38bdf8; border-radius: 4px; padding: 12px 16px; margin-bottom: 16px; }}
        .quote-box {{ margin-bottom: 6px; font-size: 14px; color: #e2e8f0; }}
        .quote-box:last-child {{ margin-bottom: 0; }}
        .details-section {{ font-size: 14px; line-height: 1.5; color: #cbd5e1; margin-bottom: 16px; }}
        .risk-text {{ color: #f59e0b; }}
        .segments-section h4 {{ margin: 0 0 8px 0; font-size: 14px; color: #94a3b8; }}
        .seg-table {{ width: 100%; border-collapse: collapse; font-size: 13px; margin-bottom: 12px; }}
        .seg-table th, .seg-table td {{ border: 1px solid #334155; padding: 6px 10px; text-align: left; }}
        .seg-table th {{ background: #0f172a; color: #94a3b8; }}
        .transcript-details summary {{ cursor: pointer; color: #38bdf8; font-size: 13px; font-weight: 500; margin-top: 8px; }}
        .transcript-box {{ background: #0f172a; border: 1px solid #334155; border-radius: 6px; padding: 12px; white-space: pre-wrap; font-family: monospace; font-size: 12px; color: #e2e8f0; margin-top: 8px; }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>{html.escape(title)}</h1>
            <p class="subtitle">Generated {len(ranked)} ranked promos with exact timestamps, evidence quotes, and editorial scores.</p>
        </header>
        {"".join(cards_html)}
    </div>
</body>
</html>
"""
    out_file.write_text(template, encoding='utf-8')
