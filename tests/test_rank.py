import pytest
from pathlib import Path
from promo_finder.candidates import Segment, Candidate
from promo_finder.rank import (
    calculate_temporal_iou,
    validate,
    rank_candidates,
    export_html_report,
)

def test_calculate_temporal_iou():
    seg1 = Segment("ep1", 0.0, 20.0, "text1")
    seg2 = Segment("ep1", 10.0, 30.0, "text2")
    c1 = Candidate("C1", [seg1], "text1", "continuous")
    c2 = Candidate("C2", [seg2], "text2", "continuous")

    # Overlap: 10 to 20 = 10s. Union: 20 + 20 - 10 = 30s. IoU = 10/30 = 0.333
    iou = calculate_temporal_iou(c1, c2)
    assert abs(iou - (10.0 / 30.0)) < 0.01

    # Identical candidate
    assert calculate_temporal_iou(c1, c1) == 1.0

    # Disjoint candidates
    seg3 = Segment("ep1", 40.0, 60.0, "text3")
    c3 = Candidate("C3", [seg3], "text3", "continuous")
    assert calculate_temporal_iou(c1, c3) == 0.0

    # Different episodes
    seg4 = Segment("ep2", 0.0, 20.0, "text4")
    c4 = Candidate("C4", [seg4], "text4", "continuous")
    assert calculate_temporal_iou(c1, c4) == 0.0

def test_validate_candidate():
    # Valid candidate (25s)
    seg_valid = Segment("ep1", 0.0, 25.0, "test")
    c_valid = Candidate("C_VAL", [seg_valid], "test", "continuous")
    assert validate(c_valid, valid_episodes={"ep1"}, episode_durations={"ep1": 100.0}) == []

    # Invalid duration (< 20s)
    seg_short = Segment("ep1", 0.0, 15.0, "test")
    c_short = Candidate("C_SHORT", [seg_short], "test", "continuous")
    errs = validate(c_short)
    assert any("duration" in e for e in errs)

    # Inverted timestamps
    seg_inv = Segment("ep1", 20.0, 10.0, "test")
    c_inv = Candidate("C_INV", [seg_inv], "test", "continuous")
    errs = validate(c_inv)
    assert any("strictly after start" in e for e in errs)

    # Overlapping segments within same episode
    s1 = Segment("ep1", 0.0, 15.0, "part 1")
    s2 = Segment("ep1", 10.0, 25.0, "part 2")  # overlaps with s1
    c_overlap = Candidate("C_OVER", [s1, s2], "part 1 part 2", "stitched")
    errs = validate(c_overlap)
    assert any("overlap" in e for e in errs)

    # Unknown episode
    errs = validate(c_valid, valid_episodes={"ep2", "ep3"})
    assert any("not in valid episodes" in e for e in errs)

def test_rank_candidates_diversity():
    s1 = Segment("ep1", 0.0, 30.0, "promo 1")
    c1 = Candidate("C1", [s1], "promo 1", "continuous")

    # Near duplicate of C1 (0.5s shift)
    s2 = Segment("ep1", 0.5, 30.5, "promo 2")
    c2 = Candidate("C2", [s2], "promo 2", "continuous")

    # Distinct promo from ep1 later
    s3 = Segment("ep1", 60.0, 90.0, "promo 3")
    c3 = Candidate("C3", [s3], "promo 3", "continuous")

    evaluations = {
        "C1": {"title": "Top Promo", "scores": {"hook": 9, "cliffhanger": 9, "standalone_clarity": 8, "narrative_coherence": 8, "emotional_impact": 8}},
        "C2": {"title": "Near Duplicate", "scores": {"hook": 8, "cliffhanger": 8, "standalone_clarity": 8, "narrative_coherence": 8, "emotional_impact": 8}},
        "C3": {"title": "Distinct Promo", "scores": {"hook": 7, "cliffhanger": 7, "standalone_clarity": 7, "narrative_coherence": 7, "emotional_impact": 7}},
    }

    # With diversity: C2 should be pruned because it overlaps with C1 (> 0.7 IoU)
    ranked = rank_candidates([c1, c2, c3], evaluations, top_n=10, diversity=True)
    ranked_ids = [r["candidate_id"] for r in ranked]
    assert "C1" in ranked_ids
    assert "C2" not in ranked_ids, "Near-duplicate should be pruned by IoU diversity"
    assert "C3" in ranked_ids

    # Check that raw score is preserved
    assert ranked[0]["raw_score"] > 80.0

def test_export_html_report(tmp_path):
    ranked = [{
        "rank": 1,
        "candidate_id": "CONT_EP1_1234",
        "title": "Dramatic Altar Betrayal",
        "generation": "continuous",
        "duration_seconds": 25.5,
        "raw_score": 85.0,
        "is_cross_episode": False,
        "evaluation_mode": "heuristic",
        "model_name": "heuristic-rules-v1",
        "scores": {"hook": 9, "cliffhanger": 9, "standalone_clarity": 8, "narrative_coherence": 8, "emotional_impact": 8},
        "hook_quote": "You should turn around, Juliana.",
        "cliffhanger_quote": "Not anymore.",
        "reasoning": "Incredible tension.",
        "risk": "Low risk.",
        "segments": [{"episode": "ep1", "start": 0.0, "end": 25.5, "duration": 25.5}],
        "transcript": 'MATRIARCH: "You should turn around, Juliana." LADY JULIANA: "Why?" MATRIARCH: "Not anymore."',
    }]

    out_file = tmp_path / "report.html"
    export_html_report(ranked, out_file)

    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")
    assert "Dramatic Altar Betrayal" in content
    assert "You should turn around, Juliana." in content
    assert "85.0" in content
