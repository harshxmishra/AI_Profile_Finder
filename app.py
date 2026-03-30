"""AI Candidate Pipeline Dashboard"""
from __future__ import annotations

import json
import os
import queue
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, List, Optional

import pandas as pd
import streamlit as st

from pipeline import DASHBOARD_REQUIRED_COLUMNS, PipelineArtifacts, run_full_pipeline

st.set_page_config(page_title="AI Candidate Pipeline Dashboard", layout="wide")

OUTPUTS_ROOT = Path("outputs")
OUTPUTS_ROOT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Phase metadata — used for progress rendering
# ---------------------------------------------------------------------------
PIPELINE_PHASES = [
    ("phase1",          "Phase 1",  "Fetch Profiles"),
    ("phase2",          "Phase 2",  "Normalize Profiles"),
    ("phase3",          "Phase 3",  "Build Dossiers"),
    ("phase3_filtered", "Phase 3B", "Relevance Filter"),
    ("phase4",          "Phase 4",  "LLM Scoring"),
    ("phase5",          "Phase 5",  "Ranking & Export"),
]

PHASE_KEY_TOKENS: dict[str, list[str]] = {
    "phase1":          ["phase 1", "apify", "fetch profiles", "dataset"],
    "phase2":          ["phase 2", "normalize"],
    "phase3":          ["phase 3", "dossier"],
    "phase3_filtered": ["phase 3b", "relevance"],
    "phase4":          ["phase 4", "llm scoring", "scoring"],
    "phase5":          ["phase 5", "ranking", "export", "campaign"],
}

PHASE_CSV_KEY_MAP = {
    "phase1":          "phase1_csv_path",
    "phase2":          "phase2_csv_path",
    "phase3":          "phase3_csv_path",
    "phase3_filtered": "phase3_filtered_csv_path",
    "phase4":          "phase4_csv_path",
    "phase5":          "final_csv_path",
}


# ---------------------------------------------------------------------------
# Session state bootstrap
# ---------------------------------------------------------------------------
def init_session_state() -> None:
    defaults: dict[str, Any] = {
        "page": "Run Pipeline",
        # pipeline run tracking
        "pipeline_running": False,
        "log_queue": None,
        "pipeline_error": None,
        "run_id": None,
        "run_completed": False,
        # progress
        "current_phase_key": None,
        "phase_counts": {},
        "run_logs": [],
        # artifacts & DataFrames
        "artifacts": None,
        "final_df": None,
        "intermediate_df": None,
        # CSV paths per phase
        "phase1_csv_path": None,
        "phase2_csv_path": None,
        "phase3_csv_path": None,
        "phase3_filtered_csv_path": None,
        "phase4_csv_path": None,
        "final_csv_path": None,
        "intermediate_csv_path": None,
        # view-results state
        "shortlist_keys": set(),
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def reset_run_state() -> None:
    st.session_state.pipeline_running = False
    st.session_state.log_queue = None
    st.session_state.pipeline_error = None
    st.session_state.run_completed = False
    st.session_state.current_phase_key = None
    st.session_state.phase_counts = {}
    st.session_state.run_logs = []
    st.session_state.artifacts = None
    st.session_state.final_df = None
    st.session_state.intermediate_df = None
    st.session_state.phase1_csv_path = None
    st.session_state.phase2_csv_path = None
    st.session_state.phase3_csv_path = None
    st.session_state.phase3_filtered_csv_path = None
    st.session_state.phase4_csv_path = None
    st.session_state.final_csv_path = None
    st.session_state.intermediate_csv_path = None
    st.session_state.shortlist_keys = set()


def append_log(message: str) -> None:
    st.session_state.run_logs.append(str(message))


def infer_phase_key(message: str) -> Optional[str]:
    msg = str(message).lower()
    for phase_key, tokens in PHASE_KEY_TOKENS.items():
        if any(tok in msg for tok in tokens):
            return phase_key
    return None


def safe_read_csv(path: str | os.PathLike[str] | None) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


def ensure_required_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in DASHBOARD_REQUIRED_COLUMNS if c not in df.columns]


def store_artifacts(artifacts: PipelineArtifacts) -> None:
    st.session_state.artifacts = artifacts
    st.session_state.final_df = artifacts.phase5_df.copy()
    st.session_state.intermediate_df = safe_read_csv(artifacts.intermediate_csv_path)
    st.session_state.final_csv_path = artifacts.final_csv_path
    st.session_state.intermediate_csv_path = artifacts.intermediate_csv_path
    st.session_state.run_completed = True
    for attr, state_key in [
        ("phase1_csv_path",          "phase1_csv_path"),
        ("phase2_csv_path",          "phase2_csv_path"),
        ("phase3_csv_path",          "phase3_csv_path"),
        ("phase3_filtered_csv_path", "phase3_filtered_csv_path"),
        ("phase4_csv_path",          "phase4_csv_path"),
    ]:
        val = getattr(artifacts, attr, None)
        if val:
            st.session_state[state_key] = val


# ---------------------------------------------------------------------------
# Queue drainage — called on every Streamlit rerun while pipeline is running
# ---------------------------------------------------------------------------
def drain_pipeline_queue() -> None:
    """Pull all pending messages from the thread queue and update session state.

    The background thread NEVER touches session_state directly — it only puts
    messages into this queue.  This function is only ever called from the main
    Streamlit thread, so there are no race conditions on session_state.
    """
    q: Optional[queue.Queue] = st.session_state.log_queue
    if q is None:
        return

    while True:
        try:
            item = q.get_nowait()
        except queue.Empty:
            break

        kind = item[0]

        if kind == "log":
            msg = item[1]
            append_log(msg)
            detected = infer_phase_key(msg)
            if detected:
                st.session_state.current_phase_key = detected

        elif kind == "phase_complete":
            _, phase_name, df, csv_path, row_count = item
            state_key = PHASE_CSV_KEY_MAP.get(phase_name)
            if state_key:
                st.session_state[state_key] = csv_path
            st.session_state.phase_counts[phase_name] = row_count
            if phase_name == "phase3_filtered":
                st.session_state.intermediate_csv_path = csv_path
                st.session_state.intermediate_df = df.copy()
            if phase_name == "phase5":
                st.session_state.final_df = df.copy()

        elif kind == "done":
            artifacts = item[1]
            store_artifacts(artifacts)
            st.session_state.pipeline_running = False
            st.session_state.current_phase_key = None
            append_log("✅ Pipeline complete.")

        elif kind == "error":
            _, err_msg, tb = item
            append_log(f"❌ ERROR: {err_msg}")
            append_log(tb)
            st.session_state.pipeline_error = err_msg
            st.session_state.pipeline_running = False
            st.session_state.current_phase_key = None


# ---------------------------------------------------------------------------
# Background thread target
# ---------------------------------------------------------------------------
def _pipeline_thread_target(
    keywords: List[str],
    locations: List[str],
    limit: int,
    output_dir: Path,
    log_q: queue.Queue,
) -> None:
    def logger(msg: str) -> None:
        log_q.put(("log", msg))

    def on_phase_complete(phase_name: str, df: pd.DataFrame, csv_path: str) -> None:
        log_q.put(("phase_complete", phase_name, df.copy(), csv_path, len(df)))

    try:
        artifacts = run_full_pipeline(
            keywords=keywords,
            location=locations,
            limit=limit,
            output_dir=output_dir,
            logger=logger,
            on_phase_complete=on_phase_complete,
        )
        log_q.put(("done", artifacts))
    except Exception as e:
        log_q.put(("error", str(e), traceback.format_exc()))


def start_pipeline_thread(
    keywords: List[str],
    locations: List[str],
    limit: int,
    output_dir: Path,
) -> None:
    log_q: queue.Queue = queue.Queue()
    st.session_state.log_queue = log_q

    thread = threading.Thread(
        target=_pipeline_thread_target,
        args=(keywords, locations, limit, output_dir, log_q),
        daemon=True,
    )
    thread.start()
    st.session_state.pipeline_running = True


# ---------------------------------------------------------------------------
# Progress renderer
# ---------------------------------------------------------------------------
def render_phase_progress() -> None:
    done_phases: set = set(st.session_state.phase_counts.keys())
    current_key: Optional[str] = st.session_state.current_phase_key
    is_running: bool = st.session_state.pipeline_running

    st.markdown("### Pipeline Progress")
    cols = st.columns(len(PIPELINE_PHASES))

    for col, (phase_key, phase_num, phase_label) in zip(cols, PIPELINE_PHASES):
        row_count = st.session_state.phase_counts.get(phase_key)
        is_done = phase_key in done_phases
        is_active = (not is_done) and is_running and (current_key == phase_key)

        with col:
            if is_done:
                st.markdown(
                    f"<div style='text-align:center;padding:0.5rem;border:1px solid #28a745;"
                    f"border-radius:8px;background:#f0fff4;'>"
                    f"<strong style='color:#28a745'>✅ {phase_num}</strong><br>"
                    f"<small>{phase_label}</small><br>"
                    f"<small style='color:#555'>{row_count} rows</small>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            elif is_active:
                st.markdown(
                    f"<div style='text-align:center;padding:0.5rem;border:2px solid #0d6efd;"
                    f"border-radius:8px;background:#e8f0fe;'>"
                    f"<strong style='color:#0d6efd'>🔄 {phase_num}</strong><br>"
                    f"<small>{phase_label}</small><br>"
                    f"<small style='color:#0d6efd'>running…</small>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div style='text-align:center;padding:0.5rem;border:1px solid #ccc;"
                    f"border-radius:8px;color:#aaa;'>"
                    f"<strong>⏳ {phase_num}</strong><br>"
                    f"<small>{phase_label}</small><br>"
                    f"<small>waiting</small>"
                    f"</div>",
                    unsafe_allow_html=True,
                )


def render_live_logs(expanded: bool = True) -> None:
    logs = st.session_state.run_logs
    with st.expander(f"Live logs ({len(logs)} entries)", expanded=expanded):
        if logs:
            # Show last 50 lines to keep it readable
            st.code("\n".join(logs[-50:]), language="text")
        else:
            st.info("Waiting for logs…")


def render_phase_downloads() -> None:
    phase_map = [
        ("phase1_csv_path",          "Phase 1 CSV"),
        ("phase2_csv_path",          "Phase 2 CSV"),
        ("phase3_csv_path",          "Phase 3 CSV"),
        ("phase3_filtered_csv_path", "Relevance CSV"),
        ("phase4_csv_path",          "Phase 4 CSV"),
        ("intermediate_csv_path",    "Intermediate CSV"),
        ("final_csv_path",           "Final CSV"),
    ]

    available = [
        (k, label) for k, label in phase_map
        if st.session_state.get(k) and Path(st.session_state[k]).exists()
    ]

    if not available:
        st.info("Downloads appear here as each phase completes.")
        return

    st.markdown("### Downloads")
    cols = st.columns(min(len(available), 3))
    for i, (state_key, label) in enumerate(available):
        path = Path(st.session_state[state_key])
        cols[i % 3].download_button(
            label=f"⬇ {label}",
            data=path.read_bytes(),
            file_name=path.name,
            mime="text/csv",
            use_container_width=True,
            key=f"dl_{state_key}",
        )


def render_run_summary() -> None:
    artifacts = st.session_state.artifacts
    if artifacts is None:
        return

    st.markdown("### Run Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Fetched profiles",  len(artifacts.phase1_df))
    c2.metric("Relevant profiles", len(artifacts.phase3_filtered_df))
    c3.metric("Final candidates",  len(artifacts.phase5_df))
    c4.metric("Run ID", st.session_state.run_id or "n/a")

    render_phase_downloads()

    with st.expander("Preview final scored candidates", expanded=True):
        df = st.session_state.final_df
        if isinstance(df, pd.DataFrame) and not df.empty:
            st.dataframe(df.head(50), use_container_width=True)
        else:
            st.info("No final output yet.")

    with st.expander("Preview intermediate dataset", expanded=False):
        idf = st.session_state.intermediate_df
        if isinstance(idf, pd.DataFrame) and not idf.empty:
            st.dataframe(idf.head(50), use_container_width=True)
        else:
            st.info("No intermediate output yet.")

    render_live_logs(expanded=False)


# ---------------------------------------------------------------------------
# Sidebar / navigation
# ---------------------------------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Run Pipeline", "View Results"], key="page")
st.sidebar.markdown("---")
st.sidebar.caption(
    "Set APIFY_API_TOKEN and OPENAI_API_KEY in your environment or HF Space secrets."
)


# ---------------------------------------------------------------------------
# Page: Run Pipeline
# ---------------------------------------------------------------------------
def page_run_pipeline() -> None:
    st.title("AI Candidate Pipeline")
    st.write(
        "Run the LinkedIn scrape → normalization → relevance filter → LLM scoring → export pipeline."
    )

    with st.expander("How this page works", expanded=False):
        st.markdown(
            """
- Enter keywords and location, then click **Run pipeline**.
- The phase tracker and logs update automatically every second.
- Download per-phase CSVs as soon as they're ready — no need to wait for the full run.
- If the pipeline errors mid-run, the logs panel will show exactly where it failed.
            """.strip()
        )

    # ------------------------------------------------------------------
    # Input form — only shown when no run is active
    # ------------------------------------------------------------------
    if not st.session_state.pipeline_running:
        with st.form("pipeline_form"):
            query_text = st.text_area(
                "Search query keywords",
                value="AI speaker, GenAI workshop, LLM trainer",
                help="Comma-separated keywords passed into the Apify actor.",
            )
            location_text = st.text_input(
                "Location filter",
                value="India",
                help="Optional. Comma-separated list for multiple locations.",
            )
            limit = st.slider("Profile limit", min_value=5, max_value=50, value=20, step=5)
            submitted = st.form_submit_button("🚀 Run pipeline", use_container_width=True)

        if submitted:
            keywords = [x.strip() for x in query_text.split(",") if x.strip()]
            locations = [x.strip() for x in location_text.split(",") if x.strip()]

            if not keywords:
                st.error("Please provide at least one keyword.")
                st.stop()
            if not os.getenv("APIFY_API_TOKEN"):
                st.error("APIFY_API_TOKEN is missing.")
                st.stop()
            if not os.getenv("OPENAI_API_KEY"):
                st.error("OPENAI_API_KEY is missing.")
                st.stop()

            reset_run_state()
            st.session_state.run_id = uuid.uuid4().hex[:10]
            output_dir = OUTPUTS_ROOT / f"run_{st.session_state.run_id}"

            start_pipeline_thread(keywords, locations, limit, output_dir)
            st.rerun()  # jump immediately into the running state below

    # ------------------------------------------------------------------
    # Running state — drain queue, render live progress, then rerun
    # ------------------------------------------------------------------
    if st.session_state.pipeline_running:
        # Always drain first so state reflects latest messages
        drain_pipeline_queue()

        st.info("⏳ Pipeline is running — refreshing every second.")
        render_phase_progress()
        st.markdown("---")
        render_live_logs(expanded=True)
        render_phase_downloads()

        # Poll: sleep then rerun so the next script execution picks up new messages.
        # This is the correct Streamlit pattern — st.rerun() raises RerunException
        # which terminates the current script run immediately after this line.
        time.sleep(1.0)
        st.rerun()
        return  # unreachable; here for clarity

    # ------------------------------------------------------------------
    # Completed / idle / error state
    # ------------------------------------------------------------------
    if st.session_state.pipeline_error:
        st.error(f"Pipeline failed: {st.session_state.pipeline_error}")

    if st.session_state.run_completed:
        st.success("✅ Pipeline completed successfully.")
        render_phase_progress()
        st.markdown("---")
        render_run_summary()
    elif st.session_state.run_logs:
        # Partial state from a previous failed run
        render_phase_progress()
        render_live_logs(expanded=True)
        render_phase_downloads()


# ---------------------------------------------------------------------------
# Helper renderers for View Results
# ---------------------------------------------------------------------------
def render_evidence(evidence_str: Any) -> List[str]:
    if pd.isna(evidence_str) or not str(evidence_str).strip():
        return []
    s = str(evidence_str).strip()
    try:
        ev = json.loads(s)
        if isinstance(ev, list):
            return [str(x) for x in ev if str(x).strip()]
    except Exception:
        pass
    return [x.strip() for x in s.split("\n") if x.strip()]


def esc(x: Any) -> str:
    return str(x).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def bullets(items: List[str]) -> str:
    if not items:
        return "<p style='margin:0.25rem 0 0 0; color:#666;'>No evidence provided.</p>"
    lis = "".join(f"<li>{esc(item)}</li>" for item in items)
    return f"<ul style='margin-top:0.25rem; margin-bottom:0;'>{lis}</ul>"


def candidate_key(row: pd.Series) -> str:
    return f"{row.get('full_name', '')}||{row.get('url', '')}"


def save_uploaded_final_csv(uploaded_file: Any) -> pd.DataFrame:
    df = pd.read_csv(uploaded_file)
    missing = ensure_required_columns(df)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    st.session_state.final_df = df
    st.session_state.final_csv_path = None
    return df


def load_current_results_df() -> pd.DataFrame:
    if isinstance(st.session_state.final_df, pd.DataFrame) and not st.session_state.final_df.empty:
        return st.session_state.final_df.copy()
    if st.session_state.final_csv_path:
        df = safe_read_csv(st.session_state.final_csv_path)
        if not df.empty:
            st.session_state.final_df = df
            return df.copy()
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Page: View Results
# ---------------------------------------------------------------------------
def page_view_results() -> None:
    st.title("Candidate Evaluation Dashboard")

    with st.expander("How scoring works", expanded=False):
        st.markdown(
            """
**Tier** — A → Top priority | B → Strong | C → Lower priority

**Best Fit** — The collaboration category with strongest alignment.

**Total Score (0–20)** — Blogs + Courses + Hack Session + PowerTalk.

**Adjusted Score** — Total + small boosts from influence and confidence.

**Confidence (0–5)** — Model certainty in its evaluation.

**Influence Score** — Speaking roles, publications, certifications, awards, projects, orgs.
            """.strip()
        )

    uploaded_file = st.sidebar.file_uploader(
        "Upload final CSV", type="csv", key="results_csv_uploader"
    )
    if uploaded_file is not None:
        try:
            df = save_uploaded_final_csv(uploaded_file)
            st.sidebar.success("Uploaded CSV loaded.")
        except Exception as e:
            st.sidebar.error(str(e))
            df = pd.DataFrame()
    else:
        df = load_current_results_df()

    if df.empty:
        st.warning("Run the pipeline first or upload the final CSV to view results.")
        return

    missing = ensure_required_columns(df)
    if missing:
        st.error(f"Missing required columns: {missing}")
        return

    st.sidebar.subheader("Filters")

    tiers = sorted(df["tier"].dropna().astype(str).unique().tolist())
    tier_filter = st.sidebar.multiselect("Tier", tiers, default=tiers)

    bestfits = df["best_fit"].dropna().astype(str).unique().tolist()
    bestfit_filter = st.sidebar.multiselect("Best Fit", bestfits, default=bestfits)

    conf_series = pd.to_numeric(df["confidence"], errors="coerce").fillna(0)
    min_conf = int(conf_series.min()) if len(df) else 0
    max_conf = int(conf_series.max()) if len(df) else 100
    if min_conf == max_conf:
        conf_range = (min_conf, max_conf)
        st.sidebar.caption(f"Confidence: {min_conf}")
    else:
        conf_range = st.sidebar.slider(
            "Confidence range", min_value=min_conf, max_value=max_conf,
            value=(min_conf, max_conf),
        )

    total_series = pd.to_numeric(df["total_score"], errors="coerce").fillna(0)
    min_total = int(total_series.min()) if len(df) else 0
    max_total = int(total_series.max()) if len(df) else 20
    if min_total == max_total:
        total_range = (min_total, max_total)
        st.sidebar.caption(f"Total score: {min_total}")
    else:
        total_range = st.sidebar.slider(
            "Total score range", min_value=min_total, max_value=max_total,
            value=(min_total, max_total),
        )

    view_df = df.copy()
    view_df = view_df[view_df["tier"].isin(tier_filter)]
    view_df = view_df[view_df["best_fit"].isin(bestfit_filter)]
    view_df = view_df[
        (pd.to_numeric(view_df["confidence"], errors="coerce") >= conf_range[0])
        & (pd.to_numeric(view_df["confidence"], errors="coerce") <= conf_range[1])
    ]
    view_df = view_df[
        (pd.to_numeric(view_df["total_score"], errors="coerce") >= total_range[0])
        & (pd.to_numeric(view_df["total_score"], errors="coerce") <= total_range[1])
    ]

    st.subheader("Sort candidates")
    sort_map = {
        "Adjusted Score":  "adjusted_score",
        "Total Score":     "total_score",
        "Confidence":      "confidence",
        "Influence Score": "influence_score",
        "Blogs Score":     "blogs_score",
        "Courses Score":   "courses_score",
        "Hack Session":    "hack_session_score",
        "PowerTalk Score": "powertalk_score",
    }
    chosen = st.multiselect(
        "Sort by (descending priority order):", list(sort_map.keys()), default=["Adjusted Score"]
    )
    if chosen:
        sort_cols = [sort_map[c] for c in chosen]
        view_df = view_df.sort_values(
            by=sort_cols, ascending=[False] * len(sort_cols)
        ).reset_index(drop=True)

    st.caption(f"Showing {len(view_df)} candidates")
    st.subheader("Tick candidates to shortlist")

    shortlist_rows = []
    for idx, row in view_df.iterrows():
        row_key = candidate_key(row)
        cols = st.columns([0.06, 0.94])

        with cols[0]:
            checked = st.checkbox(
                "Select",
                key=f"tick_{idx}_{row_key}",
                value=row_key in st.session_state.shortlist_keys,
                label_visibility="collapsed",
            )
            if checked:
                st.session_state.shortlist_keys.add(row_key)
            else:
                st.session_state.shortlist_keys.discard(row_key)

        with cols[1]:
            name        = row["full_name"]
            headline    = row.get("headline", "")
            tier        = row.get("tier", "")
            best_fit    = row.get("best_fit", "")
            total_score = row.get("total_score", "")
            adj_score   = row.get("adjusted_score", "")
            confidence  = row.get("confidence", "")
            influence   = row.get("influence_score", "")
            url         = row.get("url", "")

            risk_flags = row.get("risk_flags", "")
            risks: List[str] = []
            if isinstance(risk_flags, str) and risk_flags.strip():
                try:
                    parsed = json.loads(risk_flags)
                    risks = parsed if isinstance(parsed, list) else [risk_flags]
                except Exception:
                    risks = [risk_flags]

            blogs_score   = row.get("blogs_score", 0)
            courses_score = row.get("courses_score", 0)
            hack_score    = row.get("hack_session_score", 0)
            power_score   = row.get("powertalk_score", 0)

            blogs_reason   = row.get("blogs_reasoning", "")
            courses_reason = row.get("courses_reasoning", "")
            hack_reason    = row.get("hack_session_reasoning", "")
            power_reason   = row.get("powertalk_reasoning", "")

            blogs_ev   = render_evidence(row.get("blogs_evidence", ""))
            courses_ev = render_evidence(row.get("courses_evidence", ""))
            hack_ev    = render_evidence(row.get("hack_session_evidence", ""))
            power_ev   = render_evidence(row.get("powertalk_evidence", ""))

            risks_html = ""
            if risks:
                risks_html = (
                    "<p><strong>Risk flags:</strong> "
                    + esc(", ".join(str(r) for r in risks))
                    + "</p>"
                )

            link_html = (
                f"<p style='margin:0;'><a href='{esc(url)}' target='_blank'>🔗 View LinkedIn</a></p>"
                if str(url).startswith("http") else ""
            )

            st.markdown(
                f"""
<div style='border:1px solid #ddd;border-radius:12px;padding:1.25rem;margin-bottom:1rem;'>
  <div style='display:flex;justify-content:space-between;align-items:flex-start;gap:1rem;'>
    <div>
      <h4 style='margin:0;'>{esc(name)}</h4>
      <p style='margin:0.25rem 0 0.5rem 0;color:#555;'>{esc(headline)}</p>
      {link_html}
    </div>
    <div style='text-align:right;'>
      <p style='margin:0;'><strong>Tier:</strong> {esc(tier)}</p>
      <p style='margin:0;'><strong>Best fit:</strong> {esc(best_fit)}</p>
      <p style='margin:0;'><strong>Total:</strong> {esc(total_score)} | <strong>Adj:</strong> {esc(adj_score)}</p>
      <p style='margin:0;'><strong>Conf:</strong> {esc(confidence)} | <strong>Influence:</strong> {esc(influence)}</p>
    </div>
  </div>
  <hr>
  <p style='margin:0;'><strong>Scores</strong> |
    📝 Blogs: {esc(blogs_score)}/5 |
    📚 Courses: {esc(courses_score)}/5 |
    ⚙️ Hack: {esc(hack_score)}/5 |
    🎤 PowerTalk: {esc(power_score)}/5
  </p>
  {risks_html}
  <details style='margin-top:0.75rem;'>
    <summary><strong>Blogs</strong> (reasoning + evidence)</summary>
    <p>{esc(blogs_reason)}</p>
    {bullets(blogs_ev)}
  </details>
  <details style='margin-top:0.5rem;'>
    <summary><strong>Courses</strong> (reasoning + evidence)</summary>
    <p>{esc(courses_reason)}</p>
    {bullets(courses_ev)}
  </details>
  <details style='margin-top:0.5rem;'>
    <summary><strong>Hack Session</strong> (reasoning + evidence)</summary>
    <p>{esc(hack_reason)}</p>
    {bullets(hack_ev)}
  </details>
  <details style='margin-top:0.5rem;'>
    <summary><strong>PowerTalk</strong> (reasoning + evidence)</summary>
    <p>{esc(power_reason)}</p>
    {bullets(power_ev)}
  </details>
</div>
                """,
                unsafe_allow_html=True,
            )

        if row_key in st.session_state.shortlist_keys:
            shortlist_rows.append(row)

    st.sidebar.subheader("Shortlist")
    if shortlist_rows:
        shortlist_df = pd.DataFrame(shortlist_rows).drop_duplicates(subset=["full_name", "url"])
        cols_to_show = [
            c for c in ["full_name", "tier", "best_fit", "total_score", "confidence", "url"]
            if c in shortlist_df.columns
        ]
        st.sidebar.dataframe(shortlist_df[cols_to_show], use_container_width=True)
        st.sidebar.download_button(
            label="⬇ Download shortlist CSV",
            data=shortlist_df.to_csv(index=False).encode("utf-8"),
            file_name="shortlisted_candidates_llm.csv",
            mime="text/csv",
            use_container_width=True,
        )
        if st.sidebar.button("Clear shortlist", use_container_width=True):
            st.session_state.shortlist_keys = set()
            st.rerun()
    else:
        st.sidebar.info("No candidates shortlisted yet.")


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
if page == "Run Pipeline":
    page_run_pipeline()
else:
    page_view_results()
