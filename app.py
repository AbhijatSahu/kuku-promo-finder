"""Streamlit UI for Kuku AI Promo Finder.

Interactive dashboard to explore, filter, analyze, and generate promotional clips
from timestamped microdrama screenplays.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, List, Dict

import streamlit as st

from promo_finder.evaluate import check_llm_connection, get_api_key, resolve_model_name

# -------------------------------------------------
# Page configuration
st.set_page_config(
    page_title="Kuku AI Promo Finder",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------------------------
# Paths
OUTPUTS_DIR = Path("outputs")
RANKED_JSON = OUTPUTS_DIR / "ranked_promos.json"
RANKED_CSV = OUTPUTS_DIR / "ranked_promos.csv"
HTML_REPORT = OUTPUTS_DIR / "ranked_promos.html"
VAL_REPORT = OUTPUTS_DIR / "parser_validation_report.json"
EPISODES_DIR = Path("data/episodes")

# -------------------------------------------------
# Custom CSS
st.markdown(
    """
    <style>
    .metric-card {background: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.02) 100%);
                border: 1px solid rgba(255,255,255,0.1);border-radius:10px;padding:16px;text-align:center;}
    .badge-continuous {background:#1e3a8b;color:#93c5fd;padding:4px 10px;border-radius:6px;font-size:.82rem;}
    .badge-stitched {background:#3b2064;color:#d8b4fe;padding:4px 10px;border-radius:6px;font-size:.82rem;}
    .badge-cross {background:#831843;color:#fbcfe8;padding:4px 10px;border-radius:6px;font-size:.82rem;}
    .badge-llm {background:#064e3b;color:#6ee7b7;padding:4px 10px;border-radius:6px;font-size:.82rem;}
    .badge-heuristic {background:#713f12;color:#fde047;padding:4px 10px;border-radius:6px;font-size:.82rem;}
    .quote-box-hook{background:rgba(16,185,129,0.1);border-left:4px solid #10b981;padding:10px 14px;border-radius:0 8px 8px 0;margin-bottom:8px;}
    .quote-box-cliff{background:rgba(239,68,68,0.1);border-left:4px solid #ef4444;padding:10px 14px;border-radius:0 8px 8px 0;margin-bottom:8px;}
    .promo-card{background:#181c24;border:1px solid #2d3748;border-radius:12px;padding:20px;margin-bottom:24px;}
    .st-expander > summary,
    .st-expanderHeader,
    .streamlit-expanderHeader,
    details.st-expander > summary {
        font-size: 1.8rem !important;
        font-weight: 600 !important;
        line-height: 1.4;
    }
    .scrollbox {
        max-height: 150px;
        overflow-y: auto;
        border: 1px solid #444;
        padding: 5px;
        background: #181c24;
        border-radius: 5px;
        margin-top: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------
def format_time(seconds: float) -> str:
    """Convert seconds to MM:SS.s format."""
    m = int(seconds // 60)
    s = seconds % 60
    return f"{m:02d}:{s:04.1f}"

@st.cache_data
def load_data() -> List[Dict[str, Any]]:
    if not RANKED_JSON.exists():
        return []
    try:
        with open(RANKED_JSON, "r", encoding="utf-8") as fp:
            return json.load(fp)
    except Exception as exc:
        st.error(f"Error loading {RANKED_JSON}: {exc}")
        return []

@st.cache_data
def load_validation_data() -> Dict[str, Any]:
    if not VAL_REPORT.exists():
        return {}
    try:
        with open(VAL_REPORT, "r", encoding="utf-8") as fp:
            return json.load(fp)
    except Exception:
        return {}

def filter_promos(raw_promos: List[Dict[str, Any]], selections: Dict[str, Any] = None, **kwargs) -> List[Dict[str, Any]]:
    """Filter promos based on user selections.

    This function now accepts either a ``selections`` dict (the original
    signature) *or* keyword arguments directly. If ``selections`` is ``None``
    the provided ``kwargs`` are used as the selections dictionary. This fixes
    the ``unexpected keyword argument 'format_choice'`` error caused by the UI
    calling ``filter_promos`` with a single ``format_choice`` argument.
    """
    # Preserve backward compatibility – build a selections dict from kwargs if needed
    if selections is None:
        selections = kwargs
    # Default values for filters that are not yet exposed in the UI
    # Default values for filters that are not yet exposed in the UI
    fmt = selections.get("format_choice", "All")
    dur_min, dur_max = selections.get("dur_range", (0.0, float('inf')))
    min_score = selections.get("min_score", 0.0)
    ep_filter = selections.get("ep_filter", "All Episodes")
    query = selections.get("search_query", "")
    out: List[Dict[str, Any]] = []
    for p in raw_promos:
        dur = p.get("duration_seconds", 0.0)
        score = p.get("raw_score", 0.0)
        is_cross = p.get("is_cross_episode", False)
        gen = p.get("generation", "")
        if not (dur_min <= dur <= dur_max):
            continue
        if score < min_score:
            continue
        if fmt == "Continuous Only" and (gen != "continuous" or is_cross):
            continue
        if fmt == "Intra-Episode Stitched" and (gen != "stitched" or is_cross):
            continue
        if fmt == "Cross-Episode Stitched" and not is_cross:
            continue
        if ep_filter != "All Episodes":
            cand_eps = [s.get("episode") for s in p.get("segments", [])]
            if ep_filter not in cand_eps:
                continue
        if query:
            if not any(
                query in (p.get(k, "").lower())
                for k in ["title", "hook_quote", "cliffhanger_quote", "transcript"]
            ) and query not in " ".join(p.get("characters", [])).lower():
                continue
        out.append(p)
    return out

def render_filter_sidebar(raw_promos: List[Dict[str, Any]]) -> Dict[str, Any]:
    format_options = ["All Formats", "Continuous Only", "Intra-Episode Stitched", "Cross-Episode Stitched"]
    format_choice = st.selectbox("Promo Format", format_options, index=0)
    durations = [p.get("duration_seconds", 0.0) for p in raw_promos] if raw_promos else [20.0, 60.0]
    min_d, max_d = min(durations), max(durations)
    dur_range = st.slider(
        "Duration Range (seconds)",
        min_value=20.0,
        max_value=60.0,
        value=(float(min(min_d, 20.0)), float(max(max_d, 60.0))),
        step=0.5,
    )
    min_score = st.slider(
        "Minimum Weighted Score",
        min_value=0.0,
        max_value=100.0,
        value=0.0,
        step=1.0,
    )
    all_eps = {s.get("episode") for p in raw_promos for s in p.get("segments", []) if s.get("episode")}
    ep_filter = st.selectbox("Filter by Episode", ["All Episodes"] + sorted(all_eps), index=0)
    search_query = st.text_input("Search (Title, Quote, Dialogue, Character)", "").strip().lower()
    return {
        "format_choice": format_choice,
        "dur_range": dur_range,
        "min_score": min_score,
        "ep_filter": ep_filter,
        "search_query": search_query,
    }

def run_pipeline(
    *,
    use_heuristic: bool,
    model_choice: str,
    api_key_input: str,
    target_top: int,
    min_sec: float,
    max_sec: float
) -> None:
    """Validate inputs, check the LLM (unless heuristic), run main.py and refresh the UI."""
    if min_sec >= max_sec:
        st.error("Minimum duration must be less than maximum duration.")
        return
    if not any(EPISODES_DIR.glob("*.txt")):
        st.error("No screenplay files found. Upload at least one .txt file above.")
        return

    cmd = [
        sys.executable, "main.py",
        "--top", str(target_top),
        "--min-seconds", str(min_sec),
        "--max-seconds", str(max_sec),
    ]

    # Pass the key only to the child process; never write it into this server's os.environ.
    env = os.environ.copy()
    if use_heuristic:
        cmd.append("--dry-run")
        mode_label = "Heuristic"
    else:
        model_name = (model_choice or "").strip()
        if not model_name:
            st.error("⚠️ Enter a model name, or tick 'Use heuristic ranking'.")
            return
        api_key = (api_key_input or "").strip() or (get_api_key() or "")
        if not api_key:
            st.error("⚠️ No API key provided. Enter one, set PROMO_API_KEY, or tick 'Use heuristic ranking'.")
            return
        with st.spinner(f"Checking model '{model_name}'…"):
            ok, msg = check_llm_connection(model=model_name, api_key=api_key)
        if not ok:
            st.error(f"⚠️ {msg}\n\nPlease check the model name and API key. The pipeline was not run.")
            return
        st.success(f"✅ {msg}")
        env["PROMO_API_KEY"] = api_key
        cmd.extend(["--model", model_name, "--require-llm"])
        mode_label = resolve_model_name(model_name)

    with st.spinner(f"🚀 Running the promo-finder pipeline ({mode_label}) – please wait…"):
        t_start = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", env=env)
        elapsed = time.time() - t_start

    if proc.returncode != 0:
        st.error("Pipeline failed with error:")
        st.code(proc.stderr or proc.stdout)
        return

    load_data.clear()
    new_promos = load_data()
    llm_rows = sum(1 for p in new_promos if p.get("evaluation_mode") == "llm")
    st.session_state["pipeline_ran"] = True
    st.session_state["llm_success"] = llm_rows > 0
    notices = [("success", f"Pipeline executed successfully in {elapsed:.2f}s ({mode_label}).")]
    if not use_heuristic and llm_rows < len(new_promos):
        notices.append((
            "warning",
            f"{len(new_promos) - llm_rows} of {len(new_promos)} ranked promos use heuristic scores "
            "(LLM call failed for them, or extra slots were filled after diversity filtering). "
            "They are labelled 'Heuristic'.",
        ))
    st.session_state["run_notices"] = notices
    st.session_state["run_log"] = proc.stdout
    st.rerun()


def main() -> None:
    """Entry point for the Streamlit application."""
    # -------------------------------------------------
    # 1️⃣ Initialise session flag only if it doesn't exist
    # -------------------------------------------------
    if "pipeline_ran" not in st.session_state:
        st.session_state["pipeline_ran"] = False

    # -------------------------------------------------
    # 2️⃣ Perform cleanup **only when the pipeline has NOT run yet**
    # -------------------------------------------------
    if not st.session_state["pipeline_ran"]:
        # Clear uploaded files state
        if "uploaded_files" in st.session_state:
            del st.session_state["uploaded_files"]
        # Clear cached promo data
        load_data.clear()
        # Delete generated result files
        for path in [RANKED_JSON, RANKED_CSV, HTML_REPORT,
                     Path("outputs/all_evaluations.json")]:
            try:
                if path.exists():
                    path.unlink()
            except Exception:
                pass
        # Delete stored screenplay episodes
        episodes_dir = Path("data/episodes")
        if episodes_dir.exists():
            for ep_file in episodes_dir.iterdir():
                try:
                    ep_file.unlink()
                except Exception:
                    pass

    # -------------------------------------------------
    # 3️⃣ Load data (fresh validation report)
    # -------------------------------------------------
    raw_promos = load_data()
    try:
        with open(VAL_REPORT, "r", encoding="utf-8") as fp:
            val_data = json.load(fp)
    except Exception:
        val_data = {}

    # UI
    st.title("Kuku Promo Finder")
    st.caption(
        "Automated Microdrama Promo Discovery, Multi‑Segment Stitching & Semantic Evaluation"
    )
    col_main, col_filters = st.columns([3, 1])
    with col_filters:
        selections = render_filter_sidebar(raw_promos)

    # Determine filtered promos – reload data after pipeline run to capture LLM results
    if st.session_state.get("pipeline_ran", False):
        # Refresh promo data after pipeline execution
        raw_promos = load_data()
        filtered = filter_promos(raw_promos, selections)
    else:
        filtered = []

        filtered = filter_promos(raw_promos, selections)

    # -----------------------------------------------------------------
    # Tab layout
    # If a pipeline just finished, force the UI to switch to the first tab (Ranked Promos)
    if st.session_state.get("_goto_first_tab"):
        st.markdown(
            """
            <script>
            const tabs = document.querySelectorAll('[role="tab"]');
            if (tabs.length > 0) {
                tabs[0].click();
                // Delay scrolling to allow the tab content to render
                setTimeout(() => { window.scrollTo(0, 0); }, 100);
            }
            </script>
            """,
            unsafe_allow_html=True,
        )
        # Reset flag so we don't keep re‑clicking on subsequent reruns
        del st.session_state["_goto_first_tab"]
    
    with col_main:
        tab1, tab2, tab3 = st.tabs(
            ["🏆 Ranked Promos", "📜 Screenplay Corpus", "⚡ Run Pipeline"]
        )

        # -------------------------------------------------
        # Replace the placeholder in Tab 1
        # -------------------------------------------------
        with tab1:
            if not st.session_state.get("pipeline_ran", False):
                st.info("Run the pipeline to generate and view promo results.")
            else:
                # --- Render filtered promos (same as original code) ---
                # Scroll to top when entering Tab 1
                st.markdown("<script>window.scrollTo(0,0);</script>", unsafe_allow_html=True)
                eval_modes = {
                    p.get("model_name", "LLM") if p.get("evaluation_mode") == "llm" else "Heuristic"
                    for p in filtered
                } or {"Heuristic"}
                st.markdown(f"### **Evaluation Mode:** {' / '.join(sorted(eval_modes))}")

                for idx, p in enumerate(filtered, 1):
                    title = p.get("title", "Untitled Promo")
                    gen = p.get("generation", "")
                    is_cross = p.get("is_cross_episode", False)
                    characters = p.get("characters", [])
                    scores = p.get("scores", {})

                    if is_cross:
                        badge_fmt = '<span class="badge-cross">Cross‑Episode Stitched</span>'
                    elif gen == "stitched":
                        badge_fmt = '<span class="badge-stitched">Intra‑Episode Stitched</span>'
                    else:
                        badge_fmt = '<span class="badge-continuous">Continuous Clip</span>'

                    exp_label = f"#{idx} {title}"
                    with st.expander(exp_label, expanded=False):
                        st.markdown(f"### {title}")

                        badges = f"{badge_fmt}<br>👥 {', '.join(characters) if characters else 'Unspecified'}"
                        st.markdown(badges, unsafe_allow_html=True)

                        # Score columns
                        s_cols = st.columns(5)
                        s_cols[0].caption("Hook (25%)")
                        s_cols[0].progress(scores.get("hook", 0) / 10.0,
                                        text=f"**{scores.get('hook', 0)}/10**")
                        s_cols[1].caption("Cliffhanger (25%)")
                        s_cols[1].progress(scores.get("cliffhanger", 0) / 10.0,
                                        text=f"**{scores.get('cliffhanger', 0)}/10**")
                        s_cols[2].caption("Clarity (20%)")
                        s_cols[2].progress(scores.get("standalone_clarity", 0) / 10.0,
                                        text=f"**{scores.get('standalone_clarity', 0)}/10**")
                        s_cols[3].caption("Coherence (20%)")
                        s_cols[3].progress(scores.get("narrative_coherence", 0) / 10.0,
                                        text=f"**{scores.get('narrative_coherence', 0)}/10**")
                        s_cols[4].caption("Emotion (10%)")
                        s_cols[4].progress(scores.get("emotional_impact", 0) / 10.0,
                                        text=f"**{scores.get('emotional_impact', 0)}/10**")

                        # Quotes & rationale
                        st.markdown(f"**Editorial Rationale:** {p.get('reasoning', '')}")
                        if p.get("risk"):
                            st.markdown(f"**⚠️ Editorial Risk/Context:** {p.get('risk')}")
                        if not p.get("quotes_grounded", True):
                            st.warning("Warning: Cited quotes were not verified 100% verbatim against transcript text.")

                        # Segments table
                        segs = p.get("segments", [])
                        dur = p.get("duration_seconds", 0.0)
                        st.subheader(f"📋 Segments ({len(segs)}) – {dur:.1f}s total")
                        if segs:
                            rows = []
                            for s_idx, s in enumerate(segs, 1):
                                rows.append({
                                    "Seg #": s_idx,
                                    "Episode": s.get("episode"),
                                    "Start Time": f"{format_time(s.get('start',0.0))} ({s.get('start',0.0):.1f}s)",
                                    "End Time": f"{format_time(s.get('end',0.0))} ({s.get('end',0.0):.1f}s)",
                                    "Duration": f"{s.get('duration',0.0):.1f}s",
                                    "Lines": s.get("dialogue_count", 0),
                                    "Characters": ", ".join(s.get("characters", [])),
                                })
                            st.table(rows)
                        else:
                            st.info("No segments available.")

                        # Transcript
                        st.subheader("📖 Assembled Promo Transcript")
                        st.text_area(
                            label="Transcript",
                            value=p.get("transcript", ""),
                            height=140,
                            disabled=True,
                            key=f"trans_{p.get('candidate_id', idx)}_{idx}",
                        )
                # End of Tab 1 rendering

        # -------------------------------------------------
        # Replace the placeholder in Tab 2
        # -------------------------------------------------
        with tab2:
            st.subheader("📜 Screenplay Corpus & Validation Report")
            if not st.session_state.get("pipeline_ran", False):
                st.info("Run the pipeline to generate the screenplay corpus.")
            elif not val_data or "summary" not in val_data:
                st.info(
                    "Validation report not found. Run `python main.py --parse-only` to generate it."
                )
            else:
                summ = val_data["summary"]
                vc1, vc2, vc3, vc4 = st.columns(4)
                vc1.metric("Total Episodes", summ.get("total_episodes", 0))
                vc2.metric("Total Scenes", summ.get("total_scenes", 0))
                vc3.metric("Total Events", summ.get("total_events", 0))
                vc4.metric(
                    "Spoken Dialogue Lines",
                    summ.get("events_by_kind", {}).get("dialogue", 0),
                )
                st.markdown("---")
                st.markdown("#### Episodes Manifest & Status")
                rows = []
                for ep in val_data.get("episodes", []):
                    rows.append(
                        {
                            "Episode ID": ep.get("episode_id"),
                            "Number": ep.get("episode_number"),
                            "Duration": f"{ep.get('duration_seconds',0.0):.1f}s ({format_time(ep.get('duration_seconds',0.0))})",
                            "Scenes": ep.get("scenes_count", 0),
                            "Dialogue Lines": ep.get("dialogue_count", 0),
                            "Actions": ep.get("action_count", 0),
                            "Status": ep.get("status", "VALID"),
                            "Anomalies": ep.get("anomalies_count", 0),
                        }
                    )
                st.dataframe(rows, use_container_width=True)

        # ------------------- Tab 3 -------------------
        with tab3:
            st.subheader("⚡ Execute Promo Finder Pipeline")
            st.caption("Re‑run candidate generation, heuristic screening, and LLM evaluation with custom parameters.")
            uploaded_files = st.file_uploader(
                "Upload screenplay .txt files (will be saved to `data/episodes/`)",
                type=["txt"],
                accept_multiple_files=True,
                key="screenplay_uploader",
            )
            if uploaded_files:
                if "uploaded_files" not in st.session_state:
                    st.session_state.uploaded_files = []
                for uf in uploaded_files:
                    if not any(f["name"] == uf.name for f in st.session_state.uploaded_files):
                        st.session_state.uploaded_files.append({"name": uf.name, "data": uf.getvalue()})
                st.success(f"Added {len(uploaded_files)} file(s).")
            for f in st.session_state.get("uploaded_files", []):
                save_path = Path("data/episodes") / f["name"]
                with open(save_path, "wb") as out_f:
                    out_f.write(f["data"])


            # ---------- Evaluation method (outside the form so it toggles live) ----------
            use_heuristic = st.checkbox(
                "Use heuristic ranking (no API calls)",
                value=False,
                key="dry_run_chk",
                help="Rank promos with the deterministic rule-based scorer. Uncheck to score with an LLM.",
            )
            model_choice = st.text_input(
                "Model name (e.g., gemini-3.5-flash-lite)",
                value=os.getenv("PROMO_MODEL", "gemini-3.5-flash-lite"),
                key="model_name_input",
                disabled=use_heuristic,
            )
            api_key_input = st.text_input(
                "LLM API Key",
                type="password",
                key="llm_api_key_input",
                disabled=use_heuristic,
                help="Leave blank to use PROMO_API_KEY from .env / Streamlit secrets.",
            )

            # ---------- Pipeline parameters form ----------
            with st.form("pipeline_run_form"):
                target_top = st.number_input("Top Promos to Output", min_value=1, max_value=50, value=10, step=1, key="target_top_input")
                min_sec = st.number_input("Minimum Duration (s)", min_value=15.0, max_value=60.0, value=20.0, step=1.0, key="min_sec_input")
                max_sec = st.number_input("Maximum Duration (s)", min_value=20.0, max_value=90.0, value=60.0, step=1.0, key="max_sec_input")
                submitted = st.form_submit_button("🚀 Run Promo Finder Pipeline", use_container_width=True)

            if submitted:
                run_pipeline(
                    use_heuristic=use_heuristic,
                    model_choice=model_choice,
                    api_key_input=api_key_input,
                    target_top=int(target_top),
                    min_sec=float(min_sec),
                    max_sec=float(max_sec)
                )

    # Bottom download section
    st.markdown("---")
    col0, col1, col2, col3, col4 = st.columns([1, 2, 0.5, 2, 1])
    with col1:
        if RANKED_JSON.exists():
            st.download_button(
                label="📥 Download JSON",
                data=RANKED_JSON.read_text(encoding="utf-8"),
                file_name="ranked_promos.json",
                mime="application/json",
            )
    with col3:
        if RANKED_CSV.exists():
            st.download_button(
                label="📊 Download CSV",
                data=RANKED_CSV.read_text(encoding="utf-8-sig"),
                file_name="ranked_promos.csv",
                mime="text/csv",
            )

if __name__ == "__main__":
    main()
