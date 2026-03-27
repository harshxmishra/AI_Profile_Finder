
"""New Script"""
from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any, List

import pandas as pd
import streamlit as st

from pipeline import DASHBOARD_REQUIRED_COLUMNS, PipelineArtifacts, run_full_pipeline

st.set_page_config(page_title="AI Candidate Pipeline Dashboard", layout="wide")

OUTPUTS_ROOT = Path("outputs")
OUTPUTS_ROOT.mkdir(parents=True, exist_ok=True)


def init_session_state() -> None:
    defaults = {
        "page": "Run Pipeline",
        "artifacts": None,
        "run_id": None,
        "final_df": None,
        "intermediate_df": None,
        "final_csv_path": None,
        "intermediate_csv_path": None,
        "phase1_csv_path": None,
        "phase2_csv_path": None,
        "phase3_csv_path": None,
        "phase3_filtered_csv_path": None,
        "phase4_csv_path": None,
        "shortlist_keys": set(),
        "run_logs": [],
        "current_phase": "Idle",
        "run_completed": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


def reset_run_state() -> None:
    st.session_state.artifacts = None
    st.session_state.final_df = None
    st.session_state.intermediate_df = None
    st.session_state.final_csv_path = None
    st.session_state.intermediate_csv_path = None
    st.session_state.phase1_csv_path = None
    st.session_state.phase2_csv_path = None
    st.session_state.phase3_csv_path = None
    st.session_state.phase3_filtered_csv_path = None
    st.session_state.phase4_csv_path = None
    st.session_state.shortlist_keys = set()
    st.session_state.run_logs = []
    st.session_state.current_phase = "Starting..."
    st.session_state.run_completed = False


def append_log(message: str) -> None:
    st.session_state.run_logs.append(str(message))


def infer_phase(message: str) -> str:
    msg = str(message).lower()
    if "phase 1" in msg or "apify" in msg or "dataset" in msg or "fetch profiles" in msg:
        return "Phase 1: Fetching profiles"
    if "phase 2" in msg or "normalize" in msg:
        return "Phase 2: Normalizing profiles"
    if "phase 3" in msg or "dossier" in msg or "relevance" in msg:
        return "Phase 3: Dossiers and relevance filter"
    if "phase 4" in msg or "llm scoring" in msg or "scoring" in msg:
        return "Phase 4: LLM scoring"
    if "phase 5" in msg or "ranking" in msg or "export" in msg or "campaign" in msg:
        return "Phase 5: Ranking and exports"
    if "pipeline complete" in msg:
        return "Completed"
    return st.session_state.get("current_phase", "Running...")


class StateLogger:
    def __call__(self, message: str) -> None:
        append_log(message)
        st.session_state.current_phase = infer_phase(message)


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


def store_artifacts(artifacts: PipelineArtifacts) -> None:
    st.session_state.artifacts = artifacts
    st.session_state.final_df = artifacts.phase5_df.copy()
    st.session_state.intermediate_df = safe_read_csv(artifacts.intermediate_csv_path)
    st.session_state.final_csv_path = artifacts.final_csv_path
    st.session_state.intermediate_csv_path = artifacts.intermediate_csv_path
    st.session_state.run_completed = True
    if not st.session_state.phase1_csv_path:
        p1 = getattr(artifacts, "phase1_csv_path", None)
        if p1:
            st.session_state.phase1_csv_path = p1
    if not st.session_state.phase2_csv_path:
        p2 = getattr(artifacts, "phase2_csv_path", None)
        if p2:
            st.session_state.phase2_csv_path = p2
    if not st.session_state.phase3_csv_path:
        p3 = getattr(artifacts, "phase3_csv_path", None)
        if p3:
            st.session_state.phase3_csv_path = p3
    if not st.session_state.phase3_filtered_csv_path:
        p3f = getattr(artifacts, "phase3_filtered_csv_path", None)
        if p3f:
            st.session_state.phase3_filtered_csv_path = p3f
    if not st.session_state.phase4_csv_path:
        p4 = getattr(artifacts, "phase4_csv_path", None)
        if p4:
            st.session_state.phase4_csv_path = p4


def render_phase_downloads() -> None:
    st.markdown("### Downloads")
    phase_map = [
        ("phase1_csv_path", "Download Phase 1 CSV"),
        ("phase2_csv_path", "Download Phase 2 CSV"),
        ("phase3_csv_path", "Download Phase 3 CSV"),
        ("phase3_filtered_csv_path", "Download Relevance CSV"),
        ("phase4_csv_path", "Download Phase 4 CSV"),
        ("intermediate_csv_path", "Download Intermediate CSV"),
        ("final_csv_path", "Download Final CSV"),
    ]

    shown = False
    cols = st.columns(2)
    col_idx = 0

    for state_key, label in phase_map:
        path = st.session_state.get(state_key)
        if path and Path(path).exists():
            shown = True
            cols[col_idx % 2].download_button(
                label=label,
                data=Path(path).read_bytes(),
                file_name=Path(path).name,
                mime="text/csv",
                use_container_width=True,
                key=f"download_{state_key}",
            )
            col_idx += 1

    if not shown:
        st.info("Downloads will appear here after the run finishes.")


def render_run_summary() -> None:
    artifacts = st.session_state.artifacts
    if artifacts is None:
        return

    st.markdown("### Latest run summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Fetched profiles", len(artifacts.phase1_df))
    c2.metric("Relevant profiles", len(artifacts.phase3_filtered_df))
    c3.metric("Final candidates", len(artifacts.phase5_df))
    c4.metric("Run ID", st.session_state.run_id or "n/a")

    render_phase_downloads()

    with st.expander("Preview final scored candidates", expanded=True):
        preview_df = st.session_state.final_df if isinstance(st.session_state.final_df, pd.DataFrame) else pd.DataFrame()
        if not preview_df.empty:
            st.dataframe(preview_df.head(50), use_container_width=True)
        else:
            st.info("No final output available yet.")

    with st.expander("Preview intermediate dataset", expanded=False):
        intermediate_df = st.session_state.intermediate_df if isinstance(st.session_state.intermediate_df, pd.DataFrame) else pd.DataFrame()
        if not intermediate_df.empty:
            st.dataframe(intermediate_df.head(50), use_container_width=True)
        else:
            st.info("No intermediate output available yet.")

    with st.expander("Execution logs", expanded=False):
        if st.session_state.run_logs:
            st.code("\n".join(st.session_state.run_logs), language="text")
        else:
            st.info("No logs available.")


st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Run Pipeline", "View Results"], key="page")
st.sidebar.markdown("---")
st.sidebar.caption("Set APIFY_API_TOKEN and OPENAI_API_KEY in your Hugging Face Space secrets.")


def page_run_pipeline() -> None:
    st.title("AI Candidate Pipeline")
    st.write("Run the LinkedIn scrape, normalization, relevance filtering, scoring, and export pipeline.")

    with st.expander("How this page works", expanded=False):
        st.markdown(
            """
This page does the end to end workflow:
- takes your search query and location
- runs the Apify scrape
- builds dossiers and filters AI-relevant profiles
- scores shortlisted profiles with the LLM
- saves intermediate and final CSV files for download
            """.strip()
        )

    with st.form("pipeline_form"):
        query_text = st.text_area(
            "Search query keywords",
            value="AI speaker, GenAI workshop, LLM trainer",
            help="Comma-separated keywords passed into the Apify actor.",
        )
        location_text = st.text_input(
            "Location filter",
            value="India",
            help="Optional. Use a comma-separated list if needed.",
        )
        limit = st.slider("Profile limit", min_value=5, max_value=50, value=20, step=5)
        submitted = st.form_submit_button("Run pipeline", use_container_width=True)

    if submitted:
        keywords = [x.strip() for x in query_text.split(",") if x.strip()]
        locations = [x.strip() for x in location_text.split(",") if x.strip()]

        if not keywords:
            st.error("Please provide at least one keyword.")
            st.stop()

        if not os.getenv("APIFY_API_TOKEN"):
            st.error("APIFY_API_TOKEN is missing. Add it to your environment or Hugging Face Space secrets.")
            st.stop()

        if not os.getenv("OPENAI_API_KEY"):
            st.error("OPENAI_API_KEY is missing. Add it to your environment or Hugging Face Space secrets.")
            st.stop()

        reset_run_state()
        st.session_state.run_id = uuid.uuid4().hex[:10]
        output_dir = OUTPUTS_ROOT / f"run_{st.session_state.run_id}"

        status_box = st.empty()
        status_box.info("Pipeline is running. Please wait for completion.")

        logger = StateLogger()

        def on_phase_complete(phase_name: str, df: pd.DataFrame, csv_path: str) -> None:
            key_map = {
                "phase1": "phase1_csv_path",
                "phase2": "phase2_csv_path",
                "phase3": "phase3_csv_path",
                "phase3_filtered": "phase3_filtered_csv_path",
                "phase4": "phase4_csv_path",
                "phase5": "final_csv_path",
            }
            state_key = key_map.get(phase_name)
            if state_key:
                st.session_state[state_key] = csv_path

            if phase_name == "phase3_filtered":
                st.session_state.intermediate_csv_path = csv_path
                st.session_state.intermediate_df = df.copy()

            if phase_name == "phase5":
                st.session_state.final_df = df.copy()

        try:
            artifacts = run_full_pipeline(
                keywords=keywords,
                location=locations,
                limit=limit,
                output_dir=output_dir,
                logger=logger,
                on_phase_complete=on_phase_complete,
            )
            store_artifacts(artifacts)
            st.session_state.current_phase = "Completed"
            append_log("Pipeline complete.")
            status_box.success("Pipeline completed successfully.")
        except Exception as e:
            append_log(f"ERROR: {e}")
            st.session_state.current_phase = "Failed"
            status_box.error(f"Pipeline failed: {e}")

    st.info(f"Current phase: {st.session_state.current_phase}")

    if st.session_state.artifacts is not None:
        render_run_summary()
    elif st.session_state.run_logs:
        with st.expander("Execution logs", expanded=True):
            st.code("\n".join(st.session_state.run_logs), language="text")
        render_phase_downloads()


def page_view_results() -> None:
    st.title("Candidate Evaluation Dashboard")

    with st.expander("How scoring works", expanded=False):
        st.markdown(
            """
**Tier**
- A → Top priority
- B → Strong candidate
- C → Lower priority
**Best Fit**
- The collaboration category where the model sees strongest alignment.
**Total Score (0–20)**
- Sum of Blogs + Courses + Hack Session + PowerTalk scores.
**Adjusted Score**
- Used for ranking.
- Total score + small boosts from influence and confidence.
**Confidence (0–100)**
- Model certainty in its evaluation.
**Influence Score**
- Deterministic signal based on speaking roles, publications, certifications, awards, projects, and organizations.
            """.strip()
        )

    uploaded_file = st.sidebar.file_uploader("Upload final CSV", type="csv", key="results_csv_uploader")
    if uploaded_file is not None:
        try:
            df = save_uploaded_final_csv(uploaded_file)
            st.sidebar.success("Uploaded final CSV loaded.")
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
        st.sidebar.caption(f"Confidence range: {min_conf}")
    else:
        conf_range = st.sidebar.slider(
            "Confidence range",
            min_value=min_conf,
            max_value=max_conf,
            value=(min_conf, max_conf),
        )
    
    total_series = pd.to_numeric(df["total_score"], errors="coerce").fillna(0)
    min_total = int(total_series.min()) if len(df) else 0
    max_total = int(total_series.max()) if len(df) else 20
    
    if min_total == max_total:
        total_range = (min_total, max_total)
        st.sidebar.caption(f"Total score range: {min_total}")
    else:
        total_range = st.sidebar.slider(
            "Total score range",
            min_value=min_total,
            max_value=max_total,
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
        "Adjusted Score": "adjusted_score",
        "Total Score": "total_score",
        "Confidence": "confidence",
        "Influence Score": "influence_score",
        "Blogs Score": "blogs_score",
        "Courses Score": "courses_score",
        "Hack Session Score": "hack_session_score",
        "PowerTalk Score": "powertalk_score",
    }
    chosen = st.multiselect(
        "Sort by (descending priority order):",
        list(sort_map.keys()),
        default=["Adjusted Score"],
    )
    if chosen:
        sort_cols = [sort_map[c] for c in chosen]
        view_df = view_df.sort_values(by=sort_cols, ascending=[False] * len(sort_cols)).reset_index(drop=True)

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
            name = row["full_name"]
            headline = row.get("headline", "")
            tier = row.get("tier", "")
            best_fit = row.get("best_fit", "")
            total_score = row.get("total_score", "")
            adjusted_score = row.get("adjusted_score", "")
            confidence = row.get("confidence", "")
            influence = row.get("influence_score", "")
            url = row.get("url", "")

            risk_flags = row.get("risk_flags", "")
            risks = []
            if isinstance(risk_flags, str) and risk_flags.strip():
                try:
                    risks = json.loads(risk_flags)
                    if not isinstance(risks, list):
                        risks = [risk_flags]
                except Exception:
                    risks = [risk_flags]

            blogs_score = row.get("blogs_score", 0)
            courses_score = row.get("courses_score", 0)
            hack_score = row.get("hack_session_score", 0)
            power_score = row.get("powertalk_score", 0)

            blogs_reason = row.get("blogs_reasoning", "")
            courses_reason = row.get("courses_reasoning", "")
            hack_reason = row.get("hack_session_reasoning", "")
            power_reason = row.get("powertalk_reasoning", "")

            blogs_ev = render_evidence(row.get("blogs_evidence", ""))
            courses_ev = render_evidence(row.get("courses_evidence", ""))
            hack_ev = render_evidence(row.get("hack_session_evidence", ""))
            power_ev = render_evidence(row.get("powertalk_evidence", ""))

            risks_html = ""
            if risks:
                risks_html = "<p><strong>Risk flags:</strong> " + esc(", ".join([str(r) for r in risks])) + "</p>"

            st.markdown(
                f"""
<div style='border: 1px solid #ddd; border-radius: 12px; padding: 1.25rem; margin-bottom: 1rem;'>
  <div style='display:flex; justify-content:space-between; align-items:flex-start; gap:1rem;'>
    <div>
      <h4 style='margin:0;'>{esc(name)}</h4>
      <p style='margin:0.25rem 0 0.5rem 0; color:#555;'>{esc(headline)}</p>
      {"<p style='margin:0;'><a href='" + esc(url) + "' target='_blank'>🔗 View LinkedIn</a></p>" if str(url).startswith("http") else ""}
    </div>
    <div style='text-align:right;'>
      <p style='margin:0;'><strong>Tier:</strong> {esc(tier)}</p>
      <p style='margin:0;'><strong>Best fit:</strong> {esc(best_fit)}</p>
      <p style='margin:0;'><strong>Total:</strong> {esc(total_score)} | <strong>Adj:</strong> {esc(adjusted_score)}</p>
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
        cols_to_show = [c for c in ["full_name", "tier", "best_fit", "total_score", "confidence", "url"] if c in shortlist_df.columns]
        st.sidebar.dataframe(shortlist_df[cols_to_show], use_container_width=True)
        shortlist_csv = shortlist_df.to_csv(index=False).encode("utf-8")
        st.sidebar.download_button(
            label="Download shortlist CSV",
            data=shortlist_csv,
            file_name="shortlisted_candidates_llm.csv",
            mime="text/csv",
            use_container_width=True,
        )
        if st.sidebar.button("Clear shortlist", use_container_width=True):
            st.session_state.shortlist_keys = set()
            st.rerun()
    else:
        st.sidebar.info("No candidates shortlisted yet.")


if page == "Run Pipeline":
    page_run_pipeline()
else:
    page_view_results()

