"""Candidate evaluation module with Pydantic validation and anti-hallucination guardrails."""
from __future__ import annotations

import json
import os
import re
import time
from typing import Any
from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel, Field, field_validator

# Load .env without overriding variables already set (e.g. a key entered in the UI).
load_dotenv(find_dotenv(), override=False)

# Default rubric weights mandated by the assignment
RUBRIC = {
    'hook': 0.25,
    'cliffhanger': 0.25,
    'standalone_clarity': 0.20,
    'narrative_coherence': 0.20,
    'emotional_impact': 0.10,
}

SYSTEM_PROMPT = """You are a professional TV microdrama promo editor. Evaluate the COMPLETE assembled promo transcript.
You must return ONLY a structured JSON object with the following fields:
{
  "title": "short catchy editorial promo title (under 8 words)",
  "scores": {
    "hook": integer between 0 and 10 (strength of opening line/action in grabbing immediate attention),
    "cliffhanger": integer between 0 and 10 (strength of the final punchline/unresolved tension leaving viewer wanting more),
    "standalone_clarity": integer between 0 and 10 (can a new viewer understand the core stakes and characters?),
    "narrative_coherence": integer between 0 and 10 (does the sequence flow logically, without confusing jumps?),
    "emotional_impact": integer between 0 and 10 (visceral intensity, melodrama, betrayal, shock)
  },
  "hook_quote": "exact verbatim quote from the transcript that serves as the hook",
  "cliffhanger_quote": "exact verbatim quote from the transcript that serves as the cliffhanger",
  "reasoning": "concise editorial rationale (under 40 words)",
  "risk": "spoiler, confusion, or context risk note (under 30 words)"
}
CRITICAL RULES:
1. Every score must be an integer between 0 and 10.
2. hook_quote and cliffhanger_quote MUST BE verbatim substrings from the supplied transcript. DO NOT invent dialogue or paraphrase quotes!
3. Reward high-conflict dialogue, shocking betrayal, and abrupt cliffhangers.
4. Penalize missing context, abrupt unmotivated cuts, and weak endings.
"""

class ScoresModel(BaseModel):
    hook: int = Field(..., ge=0, le=10)
    cliffhanger: int = Field(..., ge=0, le=10)
    standalone_clarity: int = Field(..., ge=0, le=10)
    narrative_coherence: int = Field(..., ge=0, le=10)
    emotional_impact: int = Field(..., ge=0, le=10)

    @field_validator('*', mode='before')
    @classmethod
    def clamp_and_int(cls, v: Any) -> int:
        try:
            val = int(round(float(v)))
            return max(0, min(10, val))
        except (ValueError, TypeError):
            return 5

class EvaluationResult(BaseModel):
    title: str = "Untitled promo"
    scores: ScoresModel
    hook_quote: str = ""
    cliffhanger_quote: str = ""
    reasoning: str
    risk: str = ""
    evaluation_mode: str  # "llm" or "heuristic"
    model_name: str
    weighted_score: float = 0.0
    quotes_grounded: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            'title': self.title,
            'scores': self.scores.model_dump(),
            'hook_quote': self.hook_quote,
            'cliffhanger_quote': self.cliffhanger_quote,
            'reasoning': self.reasoning,
            'risk': self.risk,
            'evaluation_mode': self.evaluation_mode,
            'model_name': self.model_name,
            'weighted_score': self.weighted_score,
            'quotes_grounded': self.quotes_grounded,
        }

def calculate_weighted_score(scores: dict[str, int | float] | ScoresModel, rubric: dict[str, float] | None = None) -> float:
    """Calculate the final weighted score (0–100) using configurable weights."""
    weights = rubric or RUBRIC
    if isinstance(scores, ScoresModel):
        scores_dict = scores.model_dump()
    else:
        scores_dict = scores

    total = sum(float(scores_dict.get(k, 0)) * w for k, w in weights.items())
    return round(total * 10.0, 2)

def extract_grounded_quotes(text: str) -> tuple[str, str]:
    """Extract verbatim dialogue quotes from transcript as candidate hook/cliffhanger quotes."""
    quotes = re.findall(r'"([^"]+)"', text)
    if quotes:
        return quotes[0].strip(), quotes[-1].strip()
    # Fallback to sentence fragments if no quotes exist
    fragments = [s.strip() for s in re.split(r'[[\]!?…\n]+', text) if len(s.strip()) > 8]
    if fragments:
        return fragments[0], fragments[-1]
    return "", ""

def verify_quote_grounding(quote: str, transcript: str) -> bool:
    """Verify that a cited quote is an actual substring of the candidate transcript."""
    if not quote:
        return True
    def norm(s: str) -> str:
        return re.sub(r"[^\w\s]", "", s.lower()).strip()
    return norm(quote) in norm(transcript)

def deterministic_eval(candidate: Any, rubric: dict[str, float] | None = None) -> dict[str, Any]:
    """Deterministic, transparent heuristic evaluation clearly labeled as non-editorial."""
    t = candidate.source_text.lower()
    words = re.findall(r"\b[\w'-]+\b", t)
    first_quote, last_quote = extract_grounded_quotes(candidate.source_text)

    # Hook indicators in early transcript
    early_text = t[:min(len(t), 240)]
    base_hook = getattr(candidate, 'hook_score', 4.0)
    hook_score = int(round(base_hook)) if base_hook > 0 else 4
    if '?' in early_text:
        hook_score = max(hook_score, hook_score + 1)
    if any(k in early_text for k in ['turn around', 'why', 'wait', 'look', 'betray', 'marry', 'stop', 'listen', 'who are you', 'how dare']):
        hook_score = max(hook_score, 8)
    hook_score = max(1, min(10, hook_score))

    # Cliffhanger indicators in late transcript
    late_text = t[-min(len(t), 240):]
    base_cliff = getattr(candidate, 'cliffhanger_score', 3.0)
    cliff_score = int(round(base_cliff)) if base_cliff > 0 else 3
    if '?' in late_text:
        cliff_score = max(cliff_score, cliff_score + 1)
    if any(k in late_text for k in ['not anymore', 'marrying me', 'changed my mind', 'never going to', 'mistake', 'tell them', 'price', 'leave where']):
        cliff_score = max(cliff_score, 9)
    if any(k in late_text for k in ['no...', 'father', 'begging', 'secret', 'die', 'kill', 'destroy', 'pay']):
        cliff_score = max(cliff_score, cliff_score + 1)
    cliff_score = max(1, min(10, cliff_score))

    # Standalone clarity
    clarity_score = 7
    if len(words) < 22:
        clarity_score -= 2
    elif len(words) >= 35:
        clarity_score += 1

    chars = getattr(candidate, 'characters', [])
    if len(chars) >= 2:
        clarity_score += 1  # Multi-character confrontation aids scene comprehension
    if getattr(candidate, 'is_cross_episode', False):
        shared = candidate.metadata.get('shared_characters', [])
        if not shared:
            clarity_score -= 2
    clarity_score = max(2, min(10, clarity_score))

    # Narrative coherence
    coherence_score = 8
    if len(candidate.segments) > 2:
        coherence_score -= 1
    # If candidate is stitched with shared characters and strong beats, reward clean transition
    if getattr(candidate, 'generation', '') == 'stitched':
        if getattr(candidate, 'hook_score', 0) >= 7.0 and getattr(candidate, 'cliffhanger_score', 0) >= 7.0:
            coherence_score = min(10, coherence_score + 1)
        if getattr(candidate, 'is_cross_episode', False) and not candidate.metadata.get('shared_characters', []):
            coherence_score -= 2
    coherence_score = max(2, min(10, coherence_score))

    # Emotional impact
    emotion_keywords = [
        'love', 'hate', 'sister', 'father', 'mother', 'swore', 'believe',
        'tears', 'crying', 'shock', 'betrayal', 'marry', 'marrying', 'never',
        'ruined', 'disgrace', 'liar', 'cheated', 'mistress', 'shame',
        'heartbroken', 'devastated', 'cruel', 'gloating', 'spiteful', 'furious',
        'dead', 'alive', 'truth', 'secret', 'falsified', 'disown', 'revenge'
    ]
    emotion_count = sum(k in t for k in emotion_keywords)
    impact_score = max(2, min(10, 4 + min(5, emotion_count)))

    scores_data = {
        'hook': hook_score,
        'cliffhanger': cliff_score,
        'standalone_clarity': clarity_score,
        'narrative_coherence': coherence_score,
        'emotional_impact': impact_score,
    }
    w_score = calculate_weighted_score(scores_data, rubric)

    title = f"Promo: {first_quote[:32]}..." if first_quote else "Untitled microdrama promo"
    result = EvaluationResult(
        title=title,
        scores=ScoresModel(**scores_data),
        hook_quote=first_quote,
        cliffhanger_quote=last_quote,
        weighted_score=w_score,
        reasoning="Heuristic baseline calculated from narrative beat markers and dialogue density.",
        risk="Heuristic scores are rule-based initial estimates, not validated editor judgments.",
        evaluation_mode="heuristic",
        model_name="heuristic-rules-v1",
        quotes_grounded=True,
    )
    return result.to_dict()

def get_api_key() -> str | None:
    """Retrieve API key from supported environment variable names."""
    return (
        os.getenv('PROMO_API_KEY')
        or os.getenv('GEMINI_API_KEY')
        or os.getenv('GOOGLE_API_KEY')
        or os.getenv('OPENAI_API_KEY')
    )

def normalize_base_url(url: str | None) -> str | None:
    """Normalize base URL for OpenAI SDK compatibility, specifically for Gemini endpoints."""
    if not url:
        return None
    url = url.strip().rstrip("/")
    if "generativelanguage.googleapis.com" in url:
        if not url.endswith("/v1beta/openai"):
            url = f"{url}/v1beta/openai"
        return f"{url}/"
    return f"{url}/" if not url.endswith("/") else url

# Recommended model fallbacks when deprecated/retired model IDs are requested
MODEL_FALLBACKS: dict[str, str] = {
    "gemini-2.5-flash": "gemini-3.5-flash-lite",
    "gemini-2.0-flash": "gemini-3.5-flash-lite",
    "gemini-2.5-flash-lite": "gemini-3.5-flash-lite",
    "gemini-2.5-pro": "gemini-3.1-pro-preview",
}

def resolve_model_name(model: str | None) -> str:
    """Return the model ID that will actually be called (after deprecated-ID remapping)."""
    raw_model = (model or os.getenv('PROMO_MODEL') or 'gemini-3.5-flash-lite').strip()
    return MODEL_FALLBACKS.get(raw_model, raw_model)


def check_llm_connection(
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float = 20.0,
) -> tuple[bool, str]:
    """Make one tiny request to verify the model/API key work before a full run.

    Uses the same OpenAI-compatible client, base URL normalisation and model
    remapping as ``llm_eval`` so a passing check means evaluation will work.
    Returns ``(ok, message)``; the message never contains the API key.
    """
    key = (api_key or "").strip() or get_api_key()
    if not key:
        return False, "No API key provided. Enter one or set PROMO_API_KEY."
    model_name = resolve_model_name(model)
    url = normalize_base_url(base_url or os.getenv('PROMO_BASE_URL'))
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key, base_url=url, timeout=timeout, max_retries=0)
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{'role': 'user', 'content': 'Reply with the word OK.'}],
            max_tokens=5,
        )
        if not resp.choices:
            return False, f"Model '{model_name}' returned an empty response."
        return True, f"Model '{model_name}' is reachable."
    except Exception as exc:  # noqa: BLE001 - surface any SDK/network error to the UI
        detail = str(exc).replace(key, "***")
        return False, f"Model '{model_name}' check failed ({type(exc).__name__}): {detail[:400]}"


def llm_eval(
    candidate: Any,
    model: str | None = None,
    rubric: dict[str, float] | None = None,
    max_retries: int = 4,
) -> dict[str, Any]:
    """Call an OpenAI-compatible endpoint with strict schema validation, retry attempts (3-5x), and grounding checks."""
    key = get_api_key()
    if not key:
        return deterministic_eval(candidate, rubric)

    model_name = resolve_model_name(model)
    base_url = normalize_base_url(os.getenv('PROMO_BASE_URL'))

    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            from openai import OpenAI
            client = OpenAI(
                api_key=key,
                base_url=base_url,
                timeout=25.0,
            )

            user_payload = {
                'transcript': candidate.source_text,
                'duration_seconds': round(candidate.duration, 2),
                'segments': [s.to_dict() for s in candidate.segments],
            }

            response = client.chat.completions.create(
                model=model_name,
                temperature=0.2,
                response_format={'type': 'json_object'},
                messages=[
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': json.dumps(user_payload, ensure_ascii=False)},
                ],
            )

            raw_content = response.choices[0].message.content or "{}"
            data = json.loads(raw_content)

            # Enforce Pydantic schema validation
            raw_scores = data.get('scores', {})
            scores_model = ScoresModel(**raw_scores)

            hook_q = str(data.get('hook_quote', '')).strip()
            cliff_q = str(data.get('cliffhanger_quote', '')).strip()

            # Anti-hallucination verification: check if quotes exist in transcript
            hook_grounded = verify_quote_grounding(hook_q, candidate.source_text)
            cliff_grounded = verify_quote_grounding(cliff_q, candidate.source_text)
            all_grounded = hook_grounded and cliff_grounded

            risk_note = str(data.get('risk', '')).strip()
            if not all_grounded:
                risk_note = f"[Warning: Cited quotes may not be exact transcript text.] {risk_note}".strip()

            w_score = calculate_weighted_score(scores_model, rubric)

            result = EvaluationResult(
                title=str(data.get('title', 'Untitled promo')).strip(),
                scores=scores_model,
                hook_quote=hook_q,
                cliffhanger_quote=cliff_q,
                reasoning=str(data.get('reasoning', 'LLM semantic evaluation.')).strip(),
                risk=risk_note,
                evaluation_mode="llm",
                model_name=model_name,
                weighted_score=w_score,
                quotes_grounded=all_grounded,
            )
            return result.to_dict()

        except Exception as exc:
            last_exc = exc
            if attempt < max_retries:
                # Exponential backoff with small pause before retrying
                time.sleep(1.0 * attempt)
            else:
                break

    # If all retry attempts failed, gracefully fall back to deterministic heuristic evaluation
    fallback = deterministic_eval(candidate, rubric)
    fallback['reasoning'] = f"LLM evaluation failed after {max_retries} attempts ({type(last_exc).__name__}); deterministic fallback used."
    fallback['risk'] = f"[Fallback] {fallback['risk']}"
    return fallback

def weighted_score(scores: dict[str, int | float], rubric: dict[str, float] | None = None) -> float:
    """Convenience alias for calculate_weighted_score for backward compatibility."""
    return calculate_weighted_score(scores, rubric)
