import pytest
from pathlib import Path
from promo_finder.parser import Event, parse_episode
from promo_finder.candidates import (
    Segment,
    Candidate,
    generate_candidate_id,
    continuous_candidates,
    stitched_candidates,
    extract_moment_windows,
    score_event_hook,
    score_event_cliffhanger,
    normalize_character,
    characters_match,
    segments_share_characters,
)

def make_dummy_events(episode="ep1", count=10, character_a="ALICE", character_b="BOB"):
    events = []
    t = 0.0
    for i in range(count):
        events.append(
            Event(
                episode=episode,
                start=t,
                end=t + 4.0,
                text=f"Line {i} spoken by character.",
                character=character_a if i % 2 == 0 else character_b,
                kind="dialogue",
                source_line=i + 1,
            )
        )
        t += 5.0
    return events

def test_generate_candidate_id():
    seg1 = Segment("ep1", 0.0, 10.0, "text1")
    seg2 = Segment("ep1", 15.0, 30.0, "text2")

    id1 = generate_candidate_id([seg1, seg2], "stitched")
    id2 = generate_candidate_id([seg1, seg2], "stitched")
    assert id1 == id2, "ID generation must be deterministic"

    id3 = generate_candidate_id([seg1], "continuous")
    assert id1 != id3, "Different generation/segments must yield different IDs"
    assert id1.startswith("STITCH_")
    assert id3.startswith("CONT_")

def test_score_event_hook_and_cliffhanger():
    e_hook = Event(
        episode="ep1",
        start=0.0,
        end=3.0,
        text="Why did you betray me?",
        character="JULIANA",
        emotion="Furious",
        kind="dialogue"
    )
    assert score_event_hook(e_hook) >= 8.0

    e_cliff = Event(
        episode="ep1",
        start=50.0,
        end=53.0,
        text="I changed my mind. I am never marrying you.",
        character="GERALD",
        emotion="Cold",
        kind="dialogue"
    )
    assert score_event_cliffhanger(e_cliff) >= 8.5

def test_character_matching_and_overlap():
    assert normalize_character("LADY JULIANA") == "JULIANA"
    assert normalize_character("THE FORMER GROOM") == "THE FORMER GROOM"
    assert characters_match("LADY JULIANA", "JULIANA") is True
    assert characters_match("THE BEARDED STRANGER", "MYSTERIOUS BEARDED MAN") is True
    assert characters_match("LADY ASHCROFT", "LORD ASHCROFT") is True
    assert characters_match("LADY JULIANA", "CATHERINE") is False

    assert segments_share_characters(["LADY JULIANA", "GERALD"], ["JULIANA", "CLARISSA"]) is True
    assert segments_share_characters(["CATHERINE", "THOMAS"], ["JULIANA", "GERALD"]) is False

def test_continuous_candidates_bounds():
    events = make_dummy_events("ep1", 12)  # 60s total
    cands = continuous_candidates(events, min_seconds=20.0, max_seconds=40.0)

    assert len(cands) > 0
    for c in cands:
        assert c.generation == "continuous"
        assert len(c.segments) == 1
        assert 20.0 <= c.duration <= 40.0
        assert "ALICE" in c.source_text or "BOB" in c.source_text

def test_stitched_candidates_intra_and_cross():
    events_ep1 = make_dummy_events("ep1", 16, character_a="ALICE", character_b="BOB")
    events_ep2 = make_dummy_events("ep2", 16, character_a="ALICE", character_b="CHARLIE")  # Shares ALICE
    all_events = events_ep1 + events_ep2

    cands = stitched_candidates(all_events, min_seconds=20.0, max_seconds=50.0, max_windows_per_episode=6)
    assert len(cands) > 0

    has_intra = any(not c.is_cross_episode for c in cands)
    has_cross = any(c.is_cross_episode for c in cands)

    assert has_intra, "Should have intra-episode stitched candidates"
    assert has_cross, "Should have cross-episode stitched candidates"

    for c in cands:
        assert c.generation == "stitched"
        assert 20.0 <= c.duration <= 50.0
        # Check no overlapping segments within same episode
        for i in range(len(c.segments) - 1):
            s1 = c.segments[i]
            s2 = c.segments[i + 1]
            if s1.episode == s2.episode:
                assert s1.end <= s2.start, f"Segments in same episode must not overlap: {s1} vs {s2}"

def test_cross_episode_character_filtering():
    # ep1 has ALICE and BOB
    events_ep1 = make_dummy_events("ep1", 12, character_a="ALICE", character_b="BOB")
    # ep2 has completely unrelated characters XAVIER and YVONNE
    events_ep2 = make_dummy_events("ep2", 12, character_a="XAVIER", character_b="YVONNE")

    # With require_character_overlap=True, no cross-episode combos should be formed
    cands = stitched_candidates(events_ep1 + events_ep2, min_seconds=20.0, max_seconds=50.0, require_character_overlap=True)
    cross_cands = [c for c in cands if c.is_cross_episode]
    assert len(cross_cands) == 0, "Disjoint characters should not be stitched across episodes"

def test_real_episode_candidates_smoke():
    ep1_file = Path("data/episodes/episode-001-Episode 1671703.txt")
    ep2_file = Path("data/episodes/episode-002-Episode 1671704.txt")
    assert ep1_file.exists() and ep2_file.exists()

    events1 = parse_episode(ep1_file)
    events2 = parse_episode(ep2_file)

    cont_cands = continuous_candidates(events1, 20.0, 60.0, max_candidates_per_episode=20)
    assert 5 <= len(cont_cands) <= 20
    for c in cont_cands:
        assert 20.0 <= c.duration <= 60.0
        assert c.source_text
        assert c.hook_score >= 0.0
        assert c.cliffhanger_score >= 0.0

    stitched = stitched_candidates(events1 + events2, 20.0, 60.0, max_windows_per_episode=6)
    assert len(stitched) > 0
    for s in stitched:
        assert 20.0 <= s.duration <= 60.0
        assert s.source_text
