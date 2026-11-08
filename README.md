# Kuku Promo Finder

**Generate, evaluate, and rank short promotional clips (20–60 seconds) from timestamped micro‑drama screenplays.**

---

## Table of Contents
1. [Overview](#overview)
2. [Installation](#installation)
3. [Quickstart](#quickstart)
4. [Command‑Line Interface](#command‑line-interface)
5. [Streamlit Dashboard](#streamlit-dashboard)
6. [Configuration & Secrets](#configuration--secrets)
7. [Outputs](#outputs)
8. [Testing](#testing)
9. [Deployment (Streamlit Community Cloud)](#deployment-streamlit‑community‑cloud)
10. [License](#license)

---

## Overview
The **Kuku Promo Finder** is a production‑grade pipeline that:
- Parses timestamped screenplay files.
- Generates continuous and stitched promo candidates (1‑3 segments).
- Validates that each promo is between **20 s and 60 s** and that segment timestamps stay within episode bounds.
- Evaluates candidates with a Large Language Model (LLM) using a 5‑dimension rubric **or** a deterministic heuristic fallback.
- Ranks promos with configurable rubric weights and an optional LLM‑score boost.
- Exports the results as **JSON**, **CSV**, and an **interactive HTML report**.
- Offers an intuitive **Streamlit UI** for editorial review and re‑running the pipeline.

---

## Installation
```bash
# 1️⃣ Clone the repository (once you have pushed the cleaned history)
git clone https://github.com/AbhijatSahu/kuku-promo-finder.git
cd kuku-promo-finder

# 2️⃣ (Optional) Create a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3️⃣ Install the required Python packages
pip install -r requirements.txt
```

> **Note** – The `data/episodes/` folder is ignored by Git (see `.gitignore`). Populate it locally with your screenplay files before running the pipeline.

---

## Quickstart
```bash
# 1️⃣ Place your screenplay `.txt` files in the folder:
#    data/episodes/
#    (Each file must follow the timestamped format described in the docs.)

# 2️⃣ Add your API credentials (see the "Configuration & Secrets" section).

# 3️⃣ Run the pipeline and ask for the top 10 promos:
python main.py --top 10

# 4️⃣ Open the Streamlit UI to explore the results:
streamlit run app.py
```

---

## Command‑Line Interface
```
python main.py [OPTIONS]
```
| Flag | Default | Description |
|------|---------|-------------|
| `--input` | `data/episodes` | Directory containing screenplay `.txt` files |
| `--output` | `outputs` | Directory where reports and JSON files are written |
| `--min-seconds` | `20.0` | Minimum selected footage duration |
| `--max-seconds` | `60.0` | Maximum selected footage duration |
| `--max-segments` | `3` | Maximum number of segments in a stitched promo |
| `--top` | `30` | Number of top‑ranked promos to return |
| `--continuous-only` | `False` | Generate only single‑episode continuous clips |
| `--limit` | `0` (off) | Cap candidate pool for fast debugging |
| `--dry-run` | `False` | Skip LLM calls – use deterministic heuristic evaluation |
| `--llm-max-candidates` | `50` | Upper bound for LLM‑evaluated candidates |
| `--model` | `gpt-4o-mini` | OpenAI‑compatible model identifier |
| `--llm-retries` | `4` | Number of retry attempts for LLM evaluation |
| `--llm-weight-boost` | `5.0` | Additive boost applied to LLM‑evaluated promos during ranking |

---

## Streamlit Dashboard (`app.py`)
The UI provides:
- **Ranked promo explorer** with filters (format, duration, score, episode). 
- **Score progress bars** for Hook (25 %), Cliffhanger (25 %), Clarity (20 %), Coherence (20 %), Emotional Impact (10 %).
- **Quote cards** that display the exact transcript excerpts used for hook and cliffhanger, with anti‑hallucination verification.
- **Download buttons** (JSON & CSV) at the bottom of the page.
- **Live sliders** for LLM retries and ranking boost, so you can experiment without leaving the UI.

Run it locally with `streamlit run app.py` and follow the on‑screen instructions.

---

## Configuration & Secrets
Create a **`.env`** file in the project root (this file is ignored by Git). Example template:

```ini
# Google Gemini (or OpenAI) credentials
PROMO_API_KEY=YOUR_GCP_API_KEY          # replace with your real key
PROMO_BASE_URL=https://generativelanguage.googleapis.com
PROMO_MODEL=gemini-3.5-flash-lite
```

The pipeline automatically selects the correct endpoint based on `PROMO_BASE_URL`. **Never commit `.env`** – it is excluded via `.gitignore`.

---

## Outputs
All artefacts are written to the `outputs/` directory:
1. `ranked_promos.html` – interactive visual report.
2. `ranked_promos.json` – full structured data.
3. `ranked_promos.csv` – spreadsheet‑ready view.
4. `all_evaluations.json` – raw evaluation records for every candidate.

---

## Testing
The repository includes a `pytest` suite covering parsing, candidate generation, validation, scoring, and export logic.
```bash
python -m pytest tests/ -v
```
All tests should pass before you push a new version.

---

## License
This project is released under the **MIT License**. See the `LICENSE` file for details.

---

*Created by the Kuku competition team – © 2026*
