import pytest
from promo_finder.parser import Event
from promo_finder.candidates import Segment, Candidate
from promo_finder.evaluate import (
    calculate_weighted_score,
    deterministic_eval,
    verify_quote_grounding,
    ScoresModel,
    EvaluationResult,
    RUBRIC,
)

def make_test_candidate():
    seg = Segment(
        episode="ep1",
        start=10.0,
        end=35.0,
        text='MATRIARCH (Cold): "You should turn around, Juliana." LADY JULIANA: "Why?" MATRIARCH: "Not anymore."',
        dialogue_count=3,
    )
    return Candidate(
        candidate_id="CONT_ep1_TEST1234",
        segments=[seg],
        source_text=seg.text,
        generation="continuous",
    )

def test_calculate_weighted_score():
    perfect_scores = {
        "hook": 10,
        "cliffhanger": 10,
        "standalone_clarity": 10,
        "narrative_coherence": 10,
        "emotional_impact": 10,
    }
    assert calculate_weighted_score(perfect_scores) == 100.0

    zero_scores = {
        "hook": 0,
        "cliffhanger": 0,
        "standalone_clarity": 0,
        "narrative_coherence": 0,
        "emotional_impact": 0,
    }
    assert calculate_weighted_score(zero_scores) == 0.0

    # Custom weights
    custom_rubric = {
        "hook": 0.50,
        "cliffhanger": 0.50,
        "standalone_clarity": 0.0,
        "narrative_coherence": 0.0,
        "emotional_impact": 0.0,
    }
    scores = {"hook": 8, "cliffhanger": 6}
    # (8 * 0.5 + 6 * 0.5) * 10 = 70.0
    assert calculate_weighted_score(scores, custom_rubric) == 70.0

def test_verify_quote_grounding():
    transcript = 'MATRIARCH: "You should turn around, Juliana." LADY JULIANA: "Why?"'
    assert verify_quote_grounding("You should turn around, Juliana.", transcript) is True
    assert verify_quote_grounding("You should turn around", transcript) is True
    assert verify_quote_grounding("Why?", transcript) is True
    # Hallucinated quote
    assert verify_quote_grounding("Juliana, I hate you with all my heart!", transcript) is False

def test_deterministic_eval():
    cand = make_test_candidate()
    eval_res = deterministic_eval(cand)

    assert eval_res["evaluation_mode"] == "heuristic"
    assert eval_res["model_name"] == "heuristic-rules-v1"
    assert eval_res["quotes_grounded"] is True
    assert "heuristic" in eval_res["reasoning"].lower()
    assert 0.0 <= eval_res["weighted_score"] <= 100.0

    scores = eval_res["scores"]
    for dim in RUBRIC:
        assert dim in scores
        assert 0 <= scores[dim] <= 10

    # Verify quotes are extracted and grounded in source text
    assert eval_res["hook_quote"]
    assert eval_res["cliffhanger_quote"]
    assert verify_quote_grounding(eval_res["hook_quote"], cand.source_text)
    assert verify_quote_grounding(eval_res["cliffhanger_quote"], cand.source_text)

def test_scores_model_clamping():
    sm = ScoresModel(
        hook=15,  # Should clamp to 10
        cliffhanger=-3,  # Should clamp to 0
        standalone_clarity=8.2,  # Should round to 8
        narrative_coherence="7",  # Should parse to 7
        emotional_impact=10,
    )
    assert sm.hook == 10
    assert sm.cliffhanger == 0
    assert sm.standalone_clarity == 8
    assert sm.narrative_coherence == 7
    assert sm.emotional_impact == 10

def test_normalize_base_url_and_fallbacks():
    from promo_finder.evaluate import normalize_base_url, MODEL_FALLBACKS
    
    # Gemini base URL normalization
    assert normalize_base_url("https://generativelanguage.googleapis.com") == "https://generativelanguage.googleapis.com/v1beta/openai/"
    assert normalize_base_url("https://generativelanguage.googleapis.com/") == "https://generativelanguage.googleapis.com/v1beta/openai/"
    assert normalize_base_url("https://generativelanguage.googleapis.com/v1beta/openai") == "https://generativelanguage.googleapis.com/v1beta/openai/"
    assert normalize_base_url("https://generativelanguage.googleapis.com/v1beta/openai/") == "https://generativelanguage.googleapis.com/v1beta/openai/"

    # Other OpenAI base URLs should just ensure trailing slash
    assert normalize_base_url("https://api.openai.com/v1") == "https://api.openai.com/v1/"
    assert normalize_base_url(None) is None

    # Deprecated model mapping
    assert MODEL_FALLBACKS["gemini-2.5-flash"] == "gemini-3.5-flash-lite"
    assert MODEL_FALLBACKS["gemini-2.0-flash"] == "gemini-3.5-flash-lite"

