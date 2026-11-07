import json
import pytest
from pathlib import Path
from promo_finder.parser import (
    to_seconds,
    parse_character_heading,
    parse_episode,
    parse_episode_full,
    export_parsed_episodes,
    generate_validation_report,
    Event,
    Scene,
    Episode,
    ValidationAnomaly,
)

def test_to_seconds():
    assert to_seconds("00:00.00") == 0.0
    assert to_seconds("00:11.30") == 11.3
    assert to_seconds("(01:30.50)") == 90.5
    assert to_seconds("01:02:03.50") == 3723.5
    assert to_seconds("12:34") == 754.0

def test_parse_character_heading():
    name, emotion, kind = parse_character_heading("**MATRIARCH (Cold, Commanding):**")
    assert name == "MATRIARCH"
    assert emotion == "Cold, Commanding"
    assert kind == "dialogue"

    name, emotion, kind = parse_character_heading("**LADY JULIANA:**")
    assert name == "LADY JULIANA"
    assert emotion == ""
    assert kind == "dialogue"

    name, emotion, kind = parse_character_heading("**CAMERA:**")
    assert name == ""
    assert emotion == ""
    assert kind == "camera"

    name, emotion, kind = parse_character_heading("**ACTION:**")
    assert name == ""
    assert emotion == ""
    assert kind == "action"

    name, emotion, kind = parse_character_heading("**SOUND FX:**")
    assert name == ""
    assert emotion == ""
    assert kind == "sfx"

def test_event_formatting_and_attributes():
    e_diag = Event(
        episode_id="ep1",
        episode_number="Episode 1",
        scene_id="SCENE 01",
        scene_number=1,
        scene_heading="INT. ROOM - DAY",
        start=10.0,
        end=12.0,
        raw_start="00:10.00",
        raw_end="00:12.00",
        kind="dialogue",
        text="Turn around.",
        character="MATRIARCH",
        emotion="Cold",
        dialogue="Turn around.",
        action_description="",
        raw_text='"Turn around."',
        source_line_start=15,
        source_line_end=17,
    )
    assert e_diag.episode == "ep1"
    assert e_diag.source_line == 15
    assert e_diag.duration == 2.0
    assert e_diag.formatted_line() == 'MATRIARCH (Cold): "Turn around."'

    e_act = Event(
        episode_id="ep1",
        episode_number="Episode 1",
        scene_id="SCENE 01",
        scene_number=1,
        scene_heading="INT. ROOM - DAY",
        start=1.0,
        end=3.0,
        raw_start="00:01.00",
        raw_end="00:03.00",
        kind="action",
        text="She walks down the aisle.",
        character="",
        emotion="",
        dialogue="",
        action_description="She walks down the aisle.",
        raw_text="She walks down the aisle.",
        source_line_start=10,
        source_line_end=12,
    )
    assert e_act.formatted_line() == '[ACTION] She walks down the aisle.'

def test_parse_real_episode_smoke():
    real_file = Path("data/episodes/episode-001-Episode 1671703.txt")
    assert real_file.exists(), "Episode 1 file should exist"

    ep = parse_episode_full(real_file)
    assert ep.episode_number == "Episode 1"
    assert ep.external_id == "1671703"
    assert "Lady Juliana" in ep.logline
    assert len(ep.scenes) == 1
    assert ep.scenes[0].scene_id == "SCENE 01"
    assert ep.scenes[0].start == 0.0
    assert ep.scenes[0].end == 138.0  # 02:18.00 = 138s

    assert len(ep.events) == 95
    # First dialogue event in episode 1
    dialogues = [e for e in ep.events if e.kind == "dialogue"]
    assert len(dialogues) > 30

    first_diag = dialogues[0]
    assert first_diag.character == "MATRIARCH"
    assert first_diag.emotion == "Cold, Commanding"
    assert first_diag.dialogue == "You should turn around, Juliana."
    assert first_diag.text == "You should turn around, Juliana."
    assert first_diag.raw_start == "00:11.30"
    assert first_diag.raw_end == "00:13.10"
    assert first_diag.source_line_start == 42
    assert first_diag.source_line_end == 45

# Synthetic Edge Cases

def test_synthetic_missing_timestamps(tmp_path):
    text = """EPISODE NUMBER: Episode 99
EPISODE LOGLINE: Test missing timestamp.
CONTINUITY NOTES: None.

--- SCRIPT START ---
SCENE 01
Scene heading: INT. ROOM - DAY
Scene start: 00:00.00
Scene end: 01:00.00

**JOHN (Angry):**
"Why did you do that?"

--- SCRIPT END ---
"""
    f = tmp_path / "ep99.txt"
    f.write_text(text, encoding="utf-8")
    ep = parse_episode_full(f)
    assert len(ep.anomalies) > 0
    assert any(a.anomaly_type == "missing_timestamp" for a in ep.anomalies)

def test_synthetic_invalid_interval(tmp_path):
    text = """EPISODE NUMBER: Episode 98
EPISODE LOGLINE: Test invalid interval.
CONTINUITY NOTES: None.

--- SCRIPT START ---
SCENE 01
Scene heading: INT. ROOM - DAY
Scene start: 00:00.00
Scene end: 01:00.00

**JOHN (Angry):**
Start: (00:20.00) | End: (00:10.00)
"This is an inverted timestamp."

--- SCRIPT END ---
"""
    f = tmp_path / "ep98.txt"
    f.write_text(text, encoding="utf-8")
    ep = parse_episode_full(f)
    assert len(ep.anomalies) > 0
    assert any(a.anomaly_type == "invalid_interval" and a.severity == "error" for a in ep.anomalies)

def test_synthetic_overlapping_dialogue(tmp_path):
    text = """EPISODE NUMBER: Episode 97
EPISODE LOGLINE: Test overlapping dialogue.
CONTINUITY NOTES: None.

--- SCRIPT START ---
SCENE 01
Scene heading: INT. ROOM - DAY
Scene start: 00:00.00
Scene end: 01:00.00

**JOHN (Angry):**
Start: (00:10.00) | End: (00:15.00)
"First line speaking."

**MARY (Screaming):**
Start: (00:13.00) | End: (00:18.00)
"Second line interrupting!"

--- SCRIPT END ---
"""
    f = tmp_path / "ep97.txt"
    f.write_text(text, encoding="utf-8")
    ep = parse_episode_full(f)
    assert len(ep.anomalies) > 0
    assert any(a.anomaly_type == "overlapping_events" for a in ep.anomalies)

def test_synthetic_multi_scene_and_export(tmp_path):
    text = """EPISODE NUMBER: Episode 96
EPISODE LOGLINE: Multi scene episode.
CONTINUITY NOTES: None.

--- SCRIPT START ---
SCENE 01
Scene heading: INT. ROOM - DAY
Scene start: 00:00.00
Scene end: 00:30.00

**ACTION:**
Start: (00:00.00) | End: (00:05.00)
John looks out the window.

SCENE 02
Scene heading: EXT. GARDEN - NIGHT
Scene start: 00:30.00
Scene end: 01:00.00

**MARY (Whispering):**
Start: (00:30.00) | End: (00:35.00)
"Are we safe?"

--- SCRIPT END ---
"""
    f = tmp_path / "ep96.txt"
    f.write_text(text, encoding="utf-8")
    ep = parse_episode_full(f)
    assert len(ep.scenes) == 2
    assert ep.scenes[0].heading == "INT. ROOM - DAY"
    assert ep.scenes[1].heading == "EXT. GARDEN - NIGHT"
    assert len(ep.events) == 2
    assert ep.events[0].scene_number == 1
    assert ep.events[1].scene_number == 2

    # Test export
    out_json = tmp_path / "parsed.json"
    export_parsed_episodes([ep], out_json)
    assert out_json.exists()
    loaded = json.loads(out_json.read_text(encoding="utf-8"))
    assert len(loaded) == 1
    assert loaded[0]["episode_number"] == "Episode 96"

    # Test validation report
    rep_json = tmp_path / "validation_rep.json"
    rep = generate_validation_report([ep], rep_json)
    assert rep_json.exists()
    assert rep["summary"]["total_episodes"] == 1
    assert rep["summary"]["total_scenes"] == 2
    assert rep["summary"]["total_events"] == 2
