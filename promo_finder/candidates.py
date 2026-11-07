"""Generate beat-anchored continuous and narrative-aware stitched promo candidates."""
from __future__ import annotations

import hashlib
import re
from itertools import combinations
from dataclasses import dataclass, field
from .parser import Event

def score_event_hook(event: Event) -> float:
    """Score how effectively an event acts as an opening hook."""
    score = 2.0
    t = event.text.lower()
    if event.kind == "dialogue":
        score += 2.0
        if "?" in t:
            score += 2.0
        if any(w in t for w in ["why", "stop", "wait", "look", "turn around", "who", "never", "how dare", "what did you", "marry", "cheat", "bastard"]):
            score += 3.0
        if event.emotion and any(e in event.emotion.lower() for e in ["cold", "furious", "angry", "shock", "desperate", "mocking", "gloating", "spiteful"]):
            score += 1.5
    elif event.kind == "action":
        if any(w in t for w in ["slap", "shoves", "drops", "knees", "weeps", "tears", "shocks", "door", "runs", "flees", "discovers"]):
            score += 2.0
    return min(10.0, score)

def score_event_cliffhanger(event: Event) -> float:
    """Score how effectively an event acts as a closing cliffhanger."""
    score = 2.0
    t = event.text.lower()
    if event.kind == "dialogue":
        score += 2.0
        if "?" in t:
            score += 3.0
        if any(w in t for w in ["not anymore", "never", "changed my mind", "marrying me", "leave", "pay", "kill", "die", "destroy", "mistake", "choice", "begging", "secret"]):
            score += 3.5
        if event.emotion and any(e in event.emotion.lower() for e in ["cold", "heartless", "venomous", "cruel", "choked", "broken", "resolute", "steely"]):
            score += 1.5
    elif event.kind in {"sfx", "action"}:
        if any(w in t for w in ["gasp", "screams", "terror", "blackout", "horror", "freezes", "shriek"]):
            score += 2.5
    return min(10.0, score)

def normalize_character(name: str) -> str:
    """Normalize character name by stripping titles and descriptor prefixes."""
    cleaned = re.sub(
        r"\b(LADY|LORD|THE|NOBLE|YOUNG|ELDER|DOWAGER|DUCHESS|FORMER|GROOM|MAN|WOMAN|VOICE|OFF-SCREEN)\b",
        "",
        name.upper(),
    ).strip()
    return cleaned if cleaned else name.upper()

def characters_match(c1: str, c2: str) -> bool:
    """Check if two character labels refer to the same character."""
    if not c1 or not c2:
        return False
    n1 = normalize_character(c1)
    n2 = normalize_character(c2)
    if not n1 or not n2:
        return False
    if n1 in n2 or n2 in n1:
        return True
    tokens1 = set(n1.split())
    tokens2 = set(n2.split())
    return bool(tokens1 & tokens2)

def segments_share_characters(chars_a: list[str], chars_b: list[str]) -> bool:
    """Verify that two segments share at least one recurring character."""
    if not chars_a or not chars_b:
        return False
    for ca in chars_a:
        for cb in chars_b:
            if characters_match(ca, cb):
                return True
    return False

@dataclass
class Segment:
    episode: str
    start: float
    end: float
    text: str
    dialogue_count: int = 0
    scene_number: int = 1
    characters: list[str] = field(default_factory=list)
    hook_potential: float = 0.0
    cliffhanger_potential: float = 0.0

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict:
        return {
            'episode': self.episode,
            'start': round(self.start, 3),
            'end': round(self.end, 3),
            'duration': round(self.duration, 3),
            'text': self.text,
            'dialogue_count': self.dialogue_count,
            'scene_number': self.scene_number,
            'characters': self.characters,
            'hook_potential': round(self.hook_potential, 2),
            'cliffhanger_potential': round(self.cliffhanger_potential, 2),
        }

@dataclass
class Candidate:
    candidate_id: str
    segments: list[Segment]
    source_text: str
    generation: str  # 'continuous' or 'stitched'
    characters: list[str] = field(default_factory=list)
    hook_score: float = 0.0
    cliffhanger_score: float = 0.0
    metadata: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return sum(s.duration for s in self.segments)

    @property
    def is_cross_episode(self) -> bool:
        return len(set(s.episode for s in self.segments)) > 1

    def to_dict(self) -> dict:
        return {
            'candidate_id': self.candidate_id,
            'generation': self.generation,
            'duration_seconds': round(self.duration, 3),
            'is_cross_episode': self.is_cross_episode,
            'characters': self.characters,
            'hook_score': round(self.hook_score, 2),
            'cliffhanger_score': round(self.cliffhanger_score, 2),
            'segments': [s.to_dict() for s in self.segments],
            'transcript': self.source_text,
            'metadata': self.metadata,
        }

def generate_candidate_id(segments: list[Segment], generation: str) -> str:
    """Generate a reproducible, deterministic candidate ID using content hashing."""
    parts = [f"{s.episode}:{s.start:.2f}-{s.end:.2f}" for s in segments]
    key_str = f"{generation}:" + "|".join(parts)
    h = hashlib.sha256(key_str.encode('utf-8')).hexdigest()[:8].upper()
    prefix = "CONT" if generation == "continuous" else "STITCH"
    ep_tag = segments[0].episode.split('-')[0].replace(" ", "_")
    return f"{prefix}_{ep_tag}_{h}"

def render_segment_transcript(events: list[Event], include_camera: bool = False) -> str:
    """Render events into a clean, speaker-attributed transcript."""
    lines: list[str] = []
    for e in events:
        if e.kind == 'dialogue':
            lines.append(e.formatted_line())
        elif e.kind in {'action', 'sfx', 'transition'}:
            lines.append(e.formatted_line())
        elif include_camera and e.kind == 'camera':
            lines.append(e.formatted_line())
    return ' '.join(lines).strip()

def _build_segment(events: list[Event], start: float, end: float) -> Segment:
    """Extract events overlapping [start, end] and format into a Segment with beat metrics."""
    chosen = [e for e in events if e.start < end and e.end > start]
    text = render_segment_transcript(chosen)
    diag_cnt = sum(1 for e in chosen if e.kind == 'dialogue')
    scene_num = chosen[0].scene_number if chosen else 1
    chars = sorted(set(e.character for e in chosen if e.character))

    # Hook potential based on first 1-2 events
    opening_events = chosen[:min(len(chosen), 2)]
    hook_pot = max((score_event_hook(e) for e in opening_events), default=2.0)

    # Cliffhanger potential based on last 1-2 events
    closing_events = chosen[-min(len(chosen), 2):]
    cliff_pot = max((score_event_cliffhanger(e) for e in closing_events), default=2.0)

    return Segment(
        episode=events[0].episode,
        start=round(start, 3),
        end=round(end, 3),
        text=text,
        dialogue_count=diag_cnt,
        scene_number=scene_num,
        characters=chars,
        hook_potential=hook_pot,
        cliffhanger_potential=cliff_pot,
    )

def continuous_candidates(
    events: list[Event],
    min_seconds: float = 20.0,
    max_seconds: float = 60.0,
    step: float = 2.5,
    max_candidates_per_episode: int = 40,
) -> list[Candidate]:
    """Generate beat-anchored continuous candidates scored by dramatic potential."""
    if not events:
        return []
    events = sorted(events, key=lambda e: e.start)
    d_events = [e for e in events if e.kind in {'dialogue', 'action'}]
    if not d_events:
        return []

    pool: list[tuple[float, Candidate]] = []
    seen: set[tuple[float, float]] = set()
    last_accepted_start = -999.0

    for i, e_start in enumerate(d_events):
        start = e_start.start
        if start - last_accepted_start < step and e_start.kind != 'dialogue':
            continue

        for e_end in d_events[i:]:
            end = e_end.end
            dur = end - start
            if dur < min_seconds:
                continue
            if dur > max_seconds:
                break

            key = (round(start, 1), round(end, 1))
            if key in seen:
                continue
            seen.add(key)

            seg = _build_segment(events, start, end)
            if not seg.text.strip() or seg.dialogue_count == 0:
                continue

            cand_id = generate_candidate_id([seg], "continuous")
            dramatic_score = 0.45 * seg.hook_potential + 0.45 * seg.cliffhanger_potential + min(1.0, seg.dialogue_count / 5.0)

            c = Candidate(
                candidate_id=cand_id,
                segments=[seg],
                source_text=seg.text,
                generation='continuous',
                characters=seg.characters,
                hook_score=seg.hook_potential,
                cliffhanger_score=seg.cliffhanger_potential,
                metadata={'dialogue_count': seg.dialogue_count, 'dramatic_score': round(dramatic_score, 2)},
            )
            pool.append((dramatic_score, c))
            last_accepted_start = start

    # Sort descending by dramatic score
    pool.sort(key=lambda x: x[0], reverse=True)

    # Apply within-episode diversity (IoU < 0.75) to return distinct dramatic windows
    accepted: list[Candidate] = []
    for score, cand in pool:
        seg = cand.segments[0]
        is_dup = False
        for prev in accepted:
            p_seg = prev.segments[0]
            inter = max(0.0, min(seg.end, p_seg.end) - max(seg.start, p_seg.start))
            union = max(seg.end, p_seg.end) - min(seg.start, p_seg.start)
            if union > 0 and (inter / union) >= 0.75:
                is_dup = True
                break
        if not is_dup:
            accepted.append(cand)
            if len(accepted) >= max_candidates_per_episode:
                break

    return accepted

def extract_moment_windows(
    events: list[Event],
    min_dur: float = 6.0,
    max_dur: float = 25.0,
    max_windows: int = 8,
) -> list[Segment]:
    """Extract diverse, high-value moment windows (6-25s) from an episode."""
    d_events = [e for e in events if e.kind in {'dialogue', 'action'}]
    if not d_events:
        return []

    windows: list[tuple[float, Segment]] = []
    seen_intervals: list[tuple[float, float]] = []

    for i, e_start in enumerate(d_events):
        start = e_start.start
        for e_end in d_events[i:]:
            end = e_end.end
            dur = end - start
            if dur < min_dur:
                continue
            if dur > max_dur:
                break

            diag_events = [e for e in d_events[i : d_events.index(e_end) + 1] if e.kind == 'dialogue']
            if not diag_events:
                continue

            overlap = False
            for o_s, o_e in seen_intervals:
                overlap_len = max(0.0, min(end, o_e) - max(start, o_s))
                if overlap_len > 3.0:
                    overlap = True
                    break

            if not overlap:
                seg = _build_segment(events, start, end)
                if seg.text.strip():
                    score = seg.hook_potential + seg.cliffhanger_potential + min(2.0, seg.dialogue_count)
                    windows.append((score, seg))
                    seen_intervals.append((start, end))

    # Sort windows by dramatic score, but retain temporal spread
    windows.sort(key=lambda x: x[0], reverse=True)
    selected_segments = [w[1] for w in windows[:max_windows]]
    selected_segments.sort(key=lambda s: s.start)
    return selected_segments

def stitched_candidates(
    events: list[Event] | dict[str, list[Event]],
    min_seconds: float = 20.0,
    max_seconds: float = 60.0,
    max_segments: int = 3,
    max_windows_per_episode: int = 6,
    allow_cross_episode: bool = True,
    require_character_overlap: bool = True,
    max_gap: float = 60.0,
) -> list[Candidate]:
    """Create combinations of compact scene windows; total selected footage is summed.

    Supports intra-episode stitching (chronological) and character-coherent
    adjacent cross-episode stitching.
    """
    if isinstance(events, list):
        by_ep: dict[str, list[Event]] = {}
        for e in events:
            by_ep.setdefault(e.episode, []).append(e)
    else:
        by_ep = events

    windows_by_ep: dict[str, list[Segment]] = {}
    for ep, evs in by_ep.items():
        windows_by_ep[ep] = extract_moment_windows(evs, max_windows=max_windows_per_episode)

    candidates: list[Candidate] = []
    seen: set[tuple[tuple[str, float, float], ...]] = set()
    ep_keys = sorted(windows_by_ep.keys())

    # 1. Intra-episode stitching (within same episode)
    for ep in ep_keys:
        wins = sorted(windows_by_ep[ep], key=lambda s: s.start)
        for n in range(2, max_segments + 1):
            for combo in combinations(wins, n):
                if any(combo[k + 1].start < combo[k].end for k in range(len(combo) - 1)):
                    continue
                if max_gap and any(combo[k + 1].start - combo[k].end > max_gap for k in range(len(combo) - 1)):
                    continue

                tot_dur = sum(s.duration for s in combo)
                if not (min_seconds <= tot_dur <= max_seconds):
                    continue

                sig = tuple((s.episode, round(s.start, 1), round(s.end, 1)) for s in combo)
                if sig in seen:
                    continue
                seen.add(sig)

                cand_id = generate_candidate_id(list(combo), "stitched")
                transcript = " … ".join(s.text for s in combo)
                combo_chars = sorted(set(c for s in combo for c in s.characters))
                h_score = combo[0].hook_potential
                c_score = combo[-1].cliffhanger_potential

                candidates.append(
                    Candidate(
                        candidate_id=cand_id,
                        segments=list(combo),
                        source_text=transcript,
                        generation="stitched",
                        characters=combo_chars,
                        hook_score=h_score,
                        cliffhanger_score=c_score,
                        metadata={"type": "intra_episode", "segment_count": len(combo)},
                    )
                )

    # 2. Cross-episode stitching (between adjacent episodes with character overlap)
    if allow_cross_episode and len(ep_keys) > 1:
        for idx in range(len(ep_keys) - 1):
            ep_a = ep_keys[idx]
            ep_b = ep_keys[idx + 1]
            wins_a = windows_by_ep[ep_a]
            wins_b = windows_by_ep[ep_b]

            # 2-segment cross combos (1 from A, 1 from B)
            for sa in wins_a:
                for sb in wins_b:
                    if require_character_overlap and not segments_share_characters(sa.characters, sb.characters):
                        continue

                    tot_dur = sa.duration + sb.duration
                    if min_seconds <= tot_dur <= max_seconds:
                        combo = [sa, sb]
                        sig = tuple((s.episode, round(s.start, 1), round(s.end, 1)) for s in combo)
                        if sig in seen:
                            continue
                        seen.add(sig)

                        cand_id = generate_candidate_id(combo, "stitched")
                        transcript = f"{sa.text} … [NEXT EPISODE] {sb.text}"
                        combo_chars = sorted(set(sa.characters + sb.characters))
                        shared = [ca for ca in sa.characters for cb in sb.characters if characters_match(ca, cb)]

                        candidates.append(
                            Candidate(
                                candidate_id=cand_id,
                                segments=combo,
                                source_text=transcript,
                                generation="stitched",
                                characters=combo_chars,
                                hook_score=sa.hook_potential,
                                cliffhanger_score=sb.cliffhanger_potential,
                                metadata={
                                    "type": "cross_episode",
                                    "episodes": [ep_a, ep_b],
                                    "shared_characters": list(set(shared)),
                                },
                            )
                        )

            # 3-segment cross combos (2 from A + 1 from B, or 1 from A + 2 from B)
            if max_segments >= 3:
                # 2 from A, 1 from B
                for combo_a in combinations(wins_a, 2):
                    if combo_a[1].start < combo_a[0].end:
                        continue
                    for sb in wins_b:
                        a_chars = set(combo_a[0].characters + combo_a[1].characters)
                        if require_character_overlap and not segments_share_characters(list(a_chars), sb.characters):
                            continue

                        tot_dur = sum(s.duration for s in combo_a) + sb.duration
                        if min_seconds <= tot_dur <= max_seconds:
                            combo = [combo_a[0], combo_a[1], sb]
                            sig = tuple((s.episode, round(s.start, 1), round(s.end, 1)) for s in combo)
                            if sig in seen:
                                continue
                            seen.add(sig)

                            cand_id = generate_candidate_id(combo, "stitched")
                            transcript = f"{combo_a[0].text} … {combo_a[1].text} … [NEXT EPISODE] {sb.text}"
                            combo_chars = sorted(a_chars | set(sb.characters))

                            candidates.append(
                                Candidate(
                                    candidate_id=cand_id,
                                    segments=combo,
                                    source_text=transcript,
                                    generation="stitched",
                                    characters=combo_chars,
                                    hook_score=combo_a[0].hook_potential,
                                    cliffhanger_score=sb.cliffhanger_potential,
                                    metadata={"type": "cross_episode", "episodes": [ep_a, ep_b]},
                                )
                            )

                # 1 from A, 2 from B
                for sa in wins_a:
                    for combo_b in combinations(wins_b, 2):
                        if combo_b[1].start < combo_b[0].end:
                            continue
                        b_chars = set(combo_b[0].characters + combo_b[1].characters)
                        if require_character_overlap and not segments_share_characters(sa.characters, list(b_chars)):
                            continue

                        tot_dur = sa.duration + sum(s.duration for s in combo_b)
                        if min_seconds <= tot_dur <= max_seconds:
                            combo = [sa, combo_b[0], combo_b[1]]
                            sig = tuple((s.episode, round(s.start, 1), round(s.end, 1)) for s in combo)
                            if sig in seen:
                                continue
                            seen.add(sig)

                            cand_id = generate_candidate_id(combo, "stitched")
                            transcript = f"{sa.text} … [NEXT EPISODE] {combo_b[0].text} … {combo_b[1].text}"
                            combo_chars = sorted(set(sa.characters) | b_chars)

                            candidates.append(
                                Candidate(
                                    candidate_id=cand_id,
                                    segments=combo,
                                    source_text=transcript,
                                    generation="stitched",
                                    characters=combo_chars,
                                    hook_score=sa.hook_potential,
                                    cliffhanger_score=combo_b[-1].cliffhanger_potential,
                                    metadata={"type": "cross_episode", "episodes": [ep_a, ep_b]},
                                )
                            )

    return candidates
