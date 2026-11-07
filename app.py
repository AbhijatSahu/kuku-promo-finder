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
        font-size: 1.8rem !important;   /* adjust as needed (1.2rem, 1.8rem, etc.) */
        font-weight: 600 !important;   /* optional – makes it bold */
        line-height: 1.4;               /* optional – a bit more vertical spacing */
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

def filter_promos(raw_promos: List[Dict[str, Any]], selections: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Apply filtering based on user selections."""
    fmt = selections["format_choice"]
    dur_min, dur_max = selections["dur_range"]
    min_score = selections["min_score"]
    ep_filter = selections["ep_filter"]
    query = selections["search_query"]
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
    """Render filter widgets and return selections. Used when filters appear on the right column."""
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

    # Episode filter
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

def main() -> None:
    st.title("🎬 Kuku AI Promo Finder")
    st.caption("Automated Microdrama Promo Discovery, Multi-Segment Stitching & Semantic Evaluation")

    raw_promos = load_data()
    val_data = load_validation_data()

    # Layout: left for main content, right for filters (only on tabs 1‑3)
    col_main, col_filters = st.columns([3, 1])

    # Render filters in the right column (will be used for tabs 1‑3)
    with col_filters:
        selections = render_filter_sidebar(raw_promos)

    filtered = filter_promos(raw_promos, selections)

    # Tabs live in the left column
    with col_main:
        tab1, tab2, tab3 = st.tabs([
            "🏆 Ranked Promos",
            "📜 Screenplay Corpus",
            "⚡ Run Pipeline",
        ])

        # ------------------- Tab 1: Ranked Promos -------------------
        with tab1:
            # -------------------------------------------------
            # ️⃣  Evaluation mode + model name (large text, no counts)
            # -------------------------------------------------
            eval_modes = set()               # we only need the distinct names
            for p in filtered:
                mode = p.get("evaluation_mode", "heuristic")
                if mode == "llm":
                    # keep the LLM name (model_name) together with the mode
                    model = p.get("model_name", "unknown")
                    eval_modes.add(f"{model}")
                else:
                    eval_modes.add("Heuristic")

            # Join the distinct entries with a slash – e.g. “LLM (gemini-3.5‑flash‑lite) / Heuristic”
            display_text = " / ".join(sorted(eval_modes))

            # Render it as a large heading (you can adjust ## → #, ###, etc. for size)
            st.markdown(f"### **Evaluation Mode:** {display_text}")
            if not filtered:
                st.info("No promos matched the selected filters. Try broadening the duration or score threshold.")
            else:
                # Render each promo
                for idx, p in enumerate(filtered, 1):
                    raw_score = p.get("raw_score", 0.0)
                    dur = p.get("duration_seconds", 0.0)
                    title = p.get("title", "Untitled Promo")
                    gen = p.get("generation", "continuous")
                    is_cross = p.get("is_cross_episode", False)
                    mode = p.get("evaluation_mode", "heuristic")
                    model_name = p.get("model_name", "rules")
                    scores = p.get("scores", {})
                    hook_q = p.get("hook_quote", "")
                    cliff_q = p.get("cliffhanger_quote", "")
                    reasoning = p.get("reasoning", "")
                    risk = p.get("risk", "")
                    segments = p.get("segments", [])
                    characters = p.get("characters", [])
                    quotes_grounded = p.get("quotes_grounded", True)

                    # Badges
                    if is_cross:
                        badge_fmt = '<span class="badge-cross">Cross-Episode Stitched</span>'
                    elif gen == "stitched":
                        badge_fmt = '<span class="badge-stitched">Intra-Episode Stitched</span>'
                    else:
                        badge_fmt = '<span class="badge-continuous">Continuous Clip</span>'

                    exp_label = f"#{idx} {title}"
                    with st.expander(exp_label, expanded=False):
                        # -------------------------------------------------
                        # 2️⃣  Title + badges (rendered together, styled)
                        # -------------------------------------------------
                        # Build a single markdown line that contains the title,
                        # the badge HTML, and the character list.
                        badges = (
                            f"{badge_fmt}<br>"
                            f"👥 {', '.join(characters) if characters else 'Unspecified'}"
                        )
                        st.markdown(badges, unsafe_allow_html=True)
                        # Full details
                        s_cols = st.columns(5)
                        s_cols[0].caption("Hook (25%)")
                        s_cols[0].progress(scores.get("hook", 0) / 10.0, text=f"**{scores.get('hook',0)}/10**")
                        s_cols[1].caption("Cliffhanger (25%)")
                        s_cols[1].progress(scores.get("cliffhanger", 0) / 10.0, text=f"**{scores.get('cliffhanger',0)}/10**")
                        s_cols[2].caption("Clarity (20%)")
                        s_cols[2].progress(scores.get("standalone_clarity", 0) / 10.0, text=f"**{scores.get('standalone_clarity',0)}/10**")
                        s_cols[3].caption("Coherence (20%)")
                        s_cols[3].progress(scores.get("narrative_coherence", 0) / 10.0, text=f"**{scores.get('narrative_coherence',0)}/10**")
                        s_cols[4].caption("Emotion (10%)")
                        s_cols[4].progress(scores.get("emotional_impact", 0) / 10.0, text=f"**{scores.get('emotional_impact',0)}/10**")

                        q1, q2 = st.columns(2)
                        q1.markdown(f"""<div class='quote-box-hook'>
                            <small><b>🎯 OPENING HOOK QUOTE</b></small><br>
                            <em>\"{hook_q}\"</em>
                            </div>""", unsafe_allow_html=True)
                        q2.markdown(f"""<div class='quote-box-cliff'>
                            <small><b>⚡ CLOSING CLIFFHANGER QUOTE</b></small><br>
                            <em>\"{cliff_q}\"</em>
                            </div>""", unsafe_allow_html=True)
                        st.markdown(f"**Editorial Rationale:** {reasoning}")
                        if risk:
                            st.markdown(f"**⚠️ Editorial Risk/Context:** {risk}")
                        if not quotes_grounded:
                            st.warning("Warning: Cited quotes were not verified 100% verbatim against transcript text.")
                        st.subheader(f"📋 Segments ({len(segments)}) – {dur:.1f}s total")
                        if segments:
                            seg_data = []
                            for s_idx, s in enumerate(segments, 1):
                                seg_data.append({
                                    "Seg #": s_idx,
                                    "Episode": s.get("episode"),
                                    "Start Time": f"{format_time(s.get('start', 0.0))} ({s.get('start', 0.0):.1f}s)",
                                    "End Time": f"{format_time(s.get('end', 0.0))} ({s.get('end', 0.0):.1f}s)",
                                    "Duration": f"{s.get('duration', 0.0):.1f}s",
                                    "Lines": s.get("dialogue_count", 0),
                                    "Characters": ", ".join(s.get("characters", [])),
                                })
                            st.table(seg_data)
                        else:
                            st.info("No segments available.")

                        st.subheader("📖 Assembled Promo Transcript")
                        st.text_area(
                            label="Transcript",
                            value=p.get("transcript", ""),
                            height=140,
                            disabled=True,
                            key=f"trans_{p.get('candidate_id', idx)}_{idx}",
                        )
                # -------------------------------------------------
                #  🗂️  Bottom‑of‑page download section (no nesting)
                # -------------------------------------------------
                # A thin separator line to visually separate the main UI from the download area
                st.markdown("---")

                # Create five columns:
                #   col0 – left spacer
                #   col1 – JSON download button
                #   col2 – small spacer (just for a bit of gap)
                #   col3 – CSV download button
                #   col4 – right spacer
                col0, col1, col2, col3, col4 = st.columns([1, 2, 0.5, 2, 1])

                # JSON button (only shown if the file exists)
                with col1:
                    if RANKED_JSON.exists():
                        st.download_button(
                            label="📥 Download JSON",
                            data=RANKED_JSON.read_text(encoding="utf-8"),
                            file_name="ranked_promos.json",
                            mime="application/json",
                        )

                # CSV button (only shown if the file exists)
                with col3:
                    if RANKED_CSV.exists():
                        st.download_button(
                            label="📊 Download CSV",
                            data=RANKED_CSV.read_text(encoding="utf-8-sig"),
                            file_name="ranked_promos.csv",
                            mime="text/csv",
                        )

        # ------------------- Tab 2: Screenplay Corpus -------------------
        with tab2:
            st.subheader("📜 Screenplay Corpus & Validation Report")
            if not val_data or "summary" not in val_data:
                st.info("Validation report not found. Run `python main.py --parse-only` to generate it.")
            else:
                summ = val_data["summary"]
                vc1, vc2, vc3, vc4 = st.columns(4)
                vc1.metric("Total Episodes", summ.get("total_episodes", 0))
                vc2.metric("Total Scenes", summ.get("total_scenes", 0))
                vc3.metric("Total Events", summ.get("total_events", 0))
                vc4.metric("Spoken Dialogue Lines", summ.get("events_by_kind", {}).get("dialogue", 0))
                st.markdown("---")
                st.markdown("#### Episodes Manifest & Status")
                ep_rows = []
                for ep in val_data.get("episodes", []):
                    ep_rows.append({
                        "Episode ID": ep.get("episode_id"),
                        "Number": ep.get("episode_number"),
                        "Duration": f"{ep.get('duration_seconds',0.0):.1f}s ({format_time(ep.get('duration_seconds',0.0))})",
                        "Scenes": ep.get("scenes_count", 0),
                        "Dialogue Lines": ep.get("dialogue_count", 0),
                        "Actions": ep.get("action_count", 0),
                        "Status": ep.get("status", "VALID"),
                        "Anomalies": ep.get("anomalies_count", 0),
                    })
                st.dataframe(ep_rows, use_container_width=True)

        # ------------------- Tab 3: Run Pipeline -------------------
        with tab3:
            st.subheader("⚡ Execute Promo Finder Pipeline")
            st.caption("Re‑run candidate generation, heuristic screening, and LLM evaluation with custom parameters.")
            # ---- Upload screenplays ----

            # ---- Upload & manage screenplays ----

            uploaded_files = st.file_uploader(
                "Upload screenplay .txt files (will be saved to `data/episodes/`)",
                type=["txt"],
                accept_multiple_files=True,
                key="screenplay_uploader",
            )

            if uploaded_files:
                # Initialise session state list
                if "uploaded_files" not in st.session_state:
                    st.session_state.uploaded_files = []
                for uf in uploaded_files:
                    # Save file data in session state (avoid duplicates)
                    if not any(f["name"] == uf.name for f in st.session_state.uploaded_files):
                        st.session_state.uploaded_files.append({"name": uf.name, "data": uf.getvalue()})
                st.success(f"Added {len(uploaded_files)} file(s).")
            # Inline, scrollable file list with delete buttons (max 3 visible rows)
            

            # Write all uploaded files to the data folder before running the pipeline
            for f in st.session_state.get("uploaded_files", []):
                save_path = Path("data/episodes") / f["name"]
                with open(save_path, "wb") as out_f:
                    out_f.write(f["data"])

            with st.form("pipeline_run_form"):
                r_c1, r_c2 = st.columns(2)
                with r_c1:
                    target_top = st.number_input("Top Promos to Output", min_value=1, max_value=50, value=10, step=1)
                    min_sec = st.number_input("Minimum Duration (s)", min_value=15.0, max_value=60.0, value=20.0, step=1.0)
                    max_sec = st.number_input("Maximum Duration (s)", min_value=20.0, max_value=90.0, value=60.0, step=1.0)
                with r_c2:
                    model_choice = st.selectbox(
                        "LLM Model for Semantic Evaluation",
                        ["gemini-3.5-flash-lite", "gemini-3.8-flash", "heuristic-only"],
                        index=0,
                    )
                    cont_only = st.checkbox("Continuous Clips Only (Skip Stitching)", value=False)
                    dry_run_opt = st.checkbox(
                        "Dry Run Only (Heuristic without API calls)",
                        value=(model_choice == "heuristic-only"),
                    )
                submitted = st.form_submit_button("🚀 Run Promo Finder Pipeline", use_container_width=True)
            if submitted:
                placeholder = st.empty()
                placeholder.info("🚀 Running the promo‑finder pipeline – please wait…")
                cmd = [
                    sys.executable,
                    "main.py",
                    "--top", str(target_top),
                    "--min-seconds", str(min_sec),
                    "--max-seconds", str(max_sec),
                ]
                if cont_only:
                    cmd.append("--continuous-only")
                if dry_run_opt or model_choice == "heuristic-only":
                    cmd.append("--dry-run")
                elif model_choice:
                    cmd.extend(["--model", model_choice])
                with st.spinner("Processing screenplays, screening candidates, and evaluating with API..."):
                    t_start = time.time()
                    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
                    elapsed = time.time() - t_start
                placeholder.empty()
                if proc.returncode == 0:
                    st.success(f"Pipeline executed successfully in {elapsed:.2f}s!")
                    with st.expander("Console Output Log"):
                        st.code(proc.stdout)
                    load_data.clear()
                    st.rerun()
                else:
                    st.error("Pipeline failed with error:")
                    st.code(proc.stderr or proc.stdout)

if __name__ == "__main__":
    main()
