# Kuku Promo Finder — Antigravity project instructions

## Project goal
Build a reliable, rerunnable tool for Kuku Problem 1. It reads timestamped microdrama screenplays and proposes multiple 20–60 second promos. A promo may be one continuous clip or a sequence of timestamped segments from one or more episodes. It must return source episode IDs, exact in/out timestamps, hook, cliffhanger, explanation, score breakdown, risks, and ranking.

## Work safely
- First inspect the entire existing repository and explain its current architecture, entry points, assumptions, and known bugs. Do not rewrite files before reporting the audit and proposing a staged plan.
- Preserve working behavior. Make small, reviewable changes and run tests after each change.
- Never fabricate screenplay content, timestamps, episode IDs, scores, or evaluation results.
- Treat screenplay files as confidential. Do not upload them to public services or commit them to Git.
- Never place API keys in source code. Use environment variables and `.env` (gitignored); provide `.env.example` only.
- Ask before deleting files, changing dependencies substantially, making network calls, or using a paid API.

## Functional requirements
1. Support the actual input-pack format. Inspect real files before changing the parser. Preserve original timestamps and source line references.
2. Validate every segment: end > start; segment belongs to a real episode; timestamps are within the episode; selected footage duration is the sum of segment durations and must be 20–60 seconds.
3. Generate diverse candidates: continuous windows and stitched candidates with 2–3 segments initially. Avoid combinatorial explosion with configurable caps and transparent pruning. Do not assume cross-episode order is chronological or narratively valid.
4. Evaluate the complete assembled candidate, not isolated segments only. Use an LLM for semantic judgments when configured; retain a deterministic fallback clearly labeled as heuristic, never present heuristic scores as model/human judgments.
5. Ask the evaluator for structured JSON: hook, cliffhanger, standalone clarity, narrative coherence, emotional impact (0–10), concise rationale, weakness/context risk, and evidence quotes tied to supplied transcript text. Reject malformed or unsupported output and retry/fallback safely.
6. Calculate weighted scores in code. Default weights: hook 25%, cliffhanger 25%, standalone clarity 20%, narrative coherence 20%, emotional impact 10%. Make weights configurable and document that they are initial assumptions, not validated performance predictors.
7. Rank candidates with raw scores preserved. Add deduplication/diversity as a separate, explainable adjustment; never overwrite raw score.
8. Export raw candidate/evaluation JSON and a human-readable CSV or HTML report. Include all segments in order, total selected duration, hook, cliffhanger, rationale, weaknesses, score dimensions, raw score, adjusted score, and evaluation mode/model.
9. Include tests for timestamp parsing, duration calculation, invalid boundaries, duplicate detection, stitched candidates, cross-episode behavior, JSON validation, scoring, and exports.
10. Provide a simple local UI only after the core pipeline is reliable. Prefer Streamlit for a fast prototype unless the repository has a good reason otherwise.

## Engineering standards
- Python 3.11+; type hints and clear module boundaries.
- Keep parsing, candidate generation, evaluation, ranking, and reporting separate.
- Use deterministic IDs and reproducible runs; record config and model metadata.
- Add logging and progress reporting; handle one bad episode without losing the whole run.
- Avoid expensive all-combinations generation. Make candidate caps, window sizes, and segment count configurable.
- Add a `--dry-run` or equivalent mode that parses and generates candidates without LLM calls.
- Do not claim full-pack success until all supplied episodes have been processed and output counts are verified.

## Definition of done for each task
- Explain what changed and why.
- Run relevant automated tests and show actual results.
- Run a smoke test on a small fixture.
- List known limitations and next steps.
- Do not silently change scoring weights or candidate-generation assumptions.
