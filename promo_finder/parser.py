"""Parse and validate timestamped screenplay text into rich scenes and timed events."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any

TECHNICAL_HEADINGS = {
    'ACTION': 'action',
    'CAMERA': 'camera',
    'SOUND FX': 'sfx',
    'SFX': 'sfx',
    'MUSIC': 'music',
    'VISUAL EFFECTS': 'visual_effects',
    'VFX': 'visual_effects',
    'TRANSITION': 'transition',
    'NOTE': 'action',
}

@dataclass
class ValidationAnomaly:
    anomaly_type: str  # missing_timestamp, invalid_interval, overlapping_events, malformed_section, unrecognized_heading
    severity: str      # error, warning
    line_number: int
    description: str
    context: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass
class Scene:
    scene_id: str
    scene_number: int
    heading: str
    start: float
    end: float
    raw_start: str = ""
    raw_end: str = ""
    source_line: int = 0

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict[str, Any]:
        return {
            'scene_id': self.scene_id,
            'scene_number': self.scene_number,
            'heading': self.heading,
            'start': round(self.start, 3),
            'end': round(self.end, 3),
            'duration': round(self.duration, 3),
            'raw_start': self.raw_start,
            'raw_end': self.raw_end,
            'source_line': self.source_line,
        }

@dataclass
class Event:
    episode_id: str = ""
    start: float = 0.0
    end: float = 0.0
    text: str = ""
    episode_number: str = ""
    scene_id: str = "SCENE 01"
    scene_number: int = 1
    scene_heading: str = ""
    raw_start: str = ""
    raw_end: str = ""
    kind: str = "dialogue"  # dialogue, action, camera, sfx, music, transition
    character: str = ""
    emotion: str = ""
    dialogue: str = ""
    action_description: str = ""
    raw_text: str = ""
    source_line_start: int = 0
    source_line_end: int = 0
    episode: str = ""      # constructor compatibility
    source_line: int = 0  # constructor compatibility

    def __post_init__(self):
        if not self.episode_id and self.episode:
            self.episode_id = self.episode
        elif not self.episode and self.episode_id:
            self.episode = self.episode_id

        if not self.source_line_start and self.source_line:
            self.source_line_start = self.source_line
        elif not self.source_line and self.source_line_start:
            self.source_line = self.source_line_start

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def formatted_line(self, include_brackets: bool = True) -> str:
        """Render a readable transcript line with speaker/action context."""
        if self.kind == "dialogue":
            speaker = self.character or "SPEAKER"
            speech = self.dialogue or self.text
            if self.emotion:
                return f'{speaker} ({self.emotion}): "{speech}"'
            return f'{speaker}: "{speech}"'
        elif self.kind == "action":
            desc = self.action_description or self.text
            return f"[ACTION] {desc}" if include_brackets else desc
        elif self.kind == "camera":
            desc = self.action_description or self.text
            return f"[CAMERA] {desc}" if include_brackets else desc
        elif self.kind in {"sfx", "sound fx"}:
            return f"[SFX] {self.text}" if include_brackets else self.text
        return f"[{self.kind.upper()}] {self.text}" if include_brackets else self.text

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["duration"] = round(self.duration, 3)
        return d

@dataclass
class Episode:
    episode_id: str
    episode_number: str
    external_id: str
    logline: str
    continuity_notes: str
    scenes: list[Scene]
    events: list[Event]
    anomalies: list[ValidationAnomaly] = field(default_factory=list)

    @property
    def duration(self) -> float:
        if not self.events:
            return 0.0
        return max(e.end for e in self.events)

    def to_dict(self) -> dict[str, Any]:
        return {
            'episode_id': self.episode_id,
            'episode_number': self.episode_number,
            'external_id': self.external_id,
            'logline': self.logline,
            'continuity_notes': self.continuity_notes,
            'duration': round(self.duration, 3),
            'scenes': [s.to_dict() for s in self.scenes],
            'events': [e.to_dict() for e in self.events],
            'anomalies': [a.to_dict() for a in self.anomalies],
        }

def to_seconds(value: str) -> float:
    """Convert timestamp strings like '00:12.34' or '01:02:03.40' to seconds."""
    val = value.strip('() \t')
    parts = val.split(':')
    if len(parts) == 2:  # MM:SS or MM:SS.mmm
        minutes, sec = parts
        return int(minutes) * 60 + float(sec)
    elif len(parts) == 3:  # HH:MM:SS or HH:MM:SS.mmm
        hours, minutes, sec = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(sec)
    return float(val)

def parse_character_heading(heading_text: str) -> tuple[str, str, str]:
    """Parse character name, emotion/delivery note, and event kind.
    
    Returns:
        (character_name, emotion, kind)
    """
    cleaned = heading_text.strip()
    cleaned = re.sub(r"^\*+|\*+$", "", cleaned).strip().rstrip(':').strip()
    cleaned = re.sub(r"^\*+|\*+$", "", cleaned).strip().rstrip(':').strip()

    emotion_match = re.search(r"\(([^)]+)\)", cleaned)
    emotion = emotion_match.group(1).strip() if emotion_match else ""

    base_label = re.sub(r"\s*\([^)]*\)", "", cleaned).strip().rstrip(':').strip()
    upper_base = base_label.upper()

    if upper_base in TECHNICAL_HEADINGS:
        return "", "", TECHNICAL_HEADINGS[upper_base]

    return base_label, emotion, "dialogue"

def parse_episode_full(path: str | Path) -> Episode:
    """Parse an episode file into an Episode model with scenes, events, and validation anomalies."""
    path = Path(path)
    lines = path.read_text(encoding='utf-8', errors='replace').splitlines()

    episode_number = ""
    external_id = ""
    logline = ""
    continuity_notes = ""
    scenes: list[Scene] = []
    events: list[Event] = []
    anomalies: list[ValidationAnomaly] = []

    # Extract external ID from filename if present (e.g. episode-001-Episode 1671703)
    ext_m = re.search(r"Episode\s+(\d+)", path.stem, re.IGNORECASE)
    if ext_m:
        external_id = ext_m.group(1)

    current_scene_id = "SCENE 01"
    current_scene_num = 1
    current_scene_heading = ""
    in_script_body = False

    i = 0
    while i < len(lines):
        raw_line = lines[i]
        line = raw_line.strip()
        line_num = i + 1

        if line.startswith("EPISODE NUMBER:"):
            episode_number = line.split(":", 1)[1].strip()
            i += 1
            continue
        elif line.startswith("EPISODE LOGLINE:"):
            logline = line.split(":", 1)[1].strip()
            i += 1
            continue
        elif line.startswith("CONTINUITY NOTES:"):
            continuity_notes = line.split(":", 1)[1].strip()
            i += 1
            continue
        elif line == "--- SCRIPT START ---":
            in_script_body = True
            i += 1
            continue
        elif line == "--- SCRIPT END ---":
            in_script_body = False
            i += 1
            continue

        # Scene detection
        if line.startswith("SCENE ") and not any(line.startswith(p) for p in ("SCENE HEADING", "SCENE START", "SCENE END")):
            current_scene_id = line
            sm = re.search(r"\d+", line)
            if sm:
                current_scene_num = int(sm.group())
            i += 1
            continue
        elif line.lower().startswith("scene heading:"):
            current_scene_heading = line.split(":", 1)[1].strip()
            i += 1
            continue

        # Scene boundaries
        scene_start_match = re.match(r"(?:scene start|SCENE START):\s*([0-9:.]+)", line, re.IGNORECASE)
        if scene_start_match:
            raw_s = scene_start_match.group(1)
            s_start = to_seconds(raw_s)
            raw_e = raw_s
            s_end = s_start
            if i + 1 < len(lines):
                e_m = re.match(r"(?:scene end|SCENE END):\s*([0-9:.]+)", lines[i+1].strip(), re.IGNORECASE)
                if e_m:
                    raw_e = e_m.group(1)
                    s_end = to_seconds(raw_e)
            scenes.append(Scene(current_scene_id, current_scene_num, current_scene_heading, s_start, s_end, raw_s, raw_e, line_num))

        # Heading detection (**CAMERA:**, **ACTION:**, **CHARACTER (Emotion):**)
        heading_match = re.match(r"^\*\*([^*]+)\*\*:?$", line)
        if heading_match:
            heading_raw = heading_match.group(1).strip()
            heading_line = line_num

            # Check next line for timestamps
            if i + 1 >= len(lines):
                anomalies.append(ValidationAnomaly("missing_timestamp", "error", heading_line, f"Heading '{heading_raw}' at EOF without timestamps."))
                i += 1
                continue

            ts_line = lines[i+1].strip()
            ts_match = re.match(r"Start:\s*\(?([0-9:.]+)\)?\s*\|\s*End:\s*\(?([0-9:.]+)\)?", ts_line)
            if not ts_match:
                anomalies.append(ValidationAnomaly("missing_timestamp", "error", i + 2, f"Expected 'Start: ... | End: ...' after heading '{heading_raw}', got: '{ts_line}'."))
                i += 1
                continue

            raw_start_str = ts_match.group(1)
            raw_end_str = ts_match.group(2)
            try:
                start = to_seconds(raw_start_str)
                end = to_seconds(raw_end_str)
            except Exception as e:
                anomalies.append(ValidationAnomaly("invalid_interval", "error", i + 2, f"Failed to parse timestamp '{raw_start_str}' or '{raw_end_str}': {e}"))
                i += 2
                continue

            if end < start:
                anomalies.append(ValidationAnomaly("invalid_interval", "error", i + 2, f"End time ({end:.2f}s) is before start time ({start:.2f}s)."))
            elif end == start:
                anomalies.append(ValidationAnomaly("invalid_interval", "warning", i + 2, f"Zero-duration event at ({start:.2f}s). Clamped to +0.1s."))
                end = start + 0.1

            # Accumulate content lines
            content_lines: list[str] = []
            j = i + 2
            while j < len(lines):
                nxt = lines[j].strip()
                if re.match(r"^Start:\s*", nxt) or re.match(r"^\*\*[^*]+\*\*:?$", nxt):
                    break
                if nxt.startswith("SCENE ") or nxt.startswith("---"):
                    break
                if nxt:
                    content_lines.append(nxt)
                j += 1

            raw_text = "\n".join(content_lines).strip()
            clean_text = " ".join(re.sub(r"^\*\*|\*\*$", "", cl).strip() for cl in content_lines).strip()
            clean_text = clean_text.strip('"').strip()

            if not clean_text:
                anomalies.append(ValidationAnomaly("malformed_section", "warning", heading_line, f"Empty content under heading '{heading_raw}'."))
            else:
                char_name, emotion, kind = parse_character_heading(heading_raw)
                dialogue_text = clean_text if kind == "dialogue" else ""
                action_text = clean_text if kind in {"action", "camera"} else ""

                events.append(
                    Event(
                        episode_id=path.stem,
                        episode_number=episode_number or path.stem,
                        scene_id=current_scene_id,
                        scene_number=current_scene_num,
                        scene_heading=current_scene_heading,
                        start=round(start, 3),
                        end=round(end, 3),
                        raw_start=raw_start_str,
                        raw_end=raw_end_str,
                        kind=kind,
                        text=clean_text,
                        character=char_name,
                        emotion=emotion,
                        dialogue=dialogue_text,
                        action_description=action_text,
                        raw_text=raw_text,
                        source_line_start=heading_line,
                        source_line_end=j,
                    )
                )

            i = j
        else:
            i += 1

    # Check for overlapping events of the same kind
    for idx in range(len(events) - 1):
        e1 = events[idx]
        e2 = events[idx + 1]
        if e2.start < e1.end and e1.kind == e2.kind:
            overlap_duration = round(e1.end - e2.start, 3)
            anomalies.append(
                ValidationAnomaly(
                    anomaly_type="overlapping_events",
                    severity="warning",
                    line_number=e2.source_line_start,
                    description=f"Consecutive {e1.kind} events overlap by {overlap_duration:.2f}s: [{e1.start:.2f}-{e1.end:.2f}] and [{e2.start:.2f}-{e2.end:.2f}].",
                    context=f"{e1.text[:30]}... vs {e2.text[:30]}...",
                )
            )

    return Episode(
        episode_id=path.stem,
        episode_number=episode_number or path.stem,
        external_id=external_id,
        logline=logline,
        continuity_notes=continuity_notes,
        scenes=scenes,
        events=events,
        anomalies=anomalies,
    )

def parse_episode(path: str | Path) -> list[Event]:
    """Parse an episode file and return its events (backwards-compatible)."""
    return parse_episode_full(path).events

def export_parsed_episodes(episodes: list[Episode], output_path: str | Path) -> None:
    """Export parsed episode models as structured JSON."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    data = [ep.to_dict() for ep in episodes]
    out_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def generate_validation_report(episodes: list[Episode], output_path: str | Path | None = None) -> dict[str, Any]:
    """Generate a validation and statistics report across parsed episodes."""
    from collections import Counter
    total_scenes = sum(len(ep.scenes) for ep in episodes)
    total_events = sum(len(ep.events) for ep in episodes)
    total_anomalies = sum(len(ep.anomalies) for ep in episodes)

    kind_counts = Counter(e.kind for ep in episodes for e in ep.events)
    anomaly_counts = Counter(a.anomaly_type for ep in episodes for a in ep.anomalies)

    ep_summaries = []
    for ep in episodes:
        ep_kinds = Counter(e.kind for e in ep.events)
        ep_summaries.append({
            "episode_id": ep.episode_id,
            "episode_number": ep.episode_number,
            "external_id": ep.external_id,
            "scenes_count": len(ep.scenes),
            "events_count": len(ep.events),
            "dialogue_count": ep_kinds.get("dialogue", 0),
            "action_count": ep_kinds.get("action", 0),
            "camera_count": ep_kinds.get("camera", 0),
            "sfx_count": ep_kinds.get("sfx", 0),
            "duration_seconds": round(ep.duration, 2),
            "anomalies_count": len(ep.anomalies),
            "anomalies": [a.to_dict() for a in ep.anomalies],
            "status": "FLAGGED" if len(ep.anomalies) > 0 else "VALID",
        })

    report = {
        "summary": {
            "total_episodes": len(episodes),
            "total_scenes": total_scenes,
            "total_events": total_events,
            "total_anomalies": total_anomalies,
            "events_by_kind": dict(kind_counts),
            "anomalies_by_type": dict(anomaly_counts),
        },
        "episodes": ep_summaries,
    }

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    return report

def split_character(body: str) -> tuple[str, str]:
    """Legacy helper for splitting character from colon-separated line."""
    m = re.match(r"^([A-Z][A-Z0-9 .'-]{1,35})(?:\s*\([^)]*\))?\s*[:—-]\s*(.+)$", body)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return "", body

def classify(text: str) -> str:
    """Classify text kind when heading is unavailable."""
    t = text.lower()
    if any(x in t for x in ['cut to black', 'fade out', 'blackout']):
        return 'transition'
    if text.startswith('(') or text.startswith('['):
        return 'action'
    return 'dialogue'
