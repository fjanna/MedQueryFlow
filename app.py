from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict

import streamlit as st

from core.pipeline import MedQueryFlowPipeline
from utils.logger import setup_logging

setup_logging()


@st.cache_resource(show_spinner=False)
def load_pipeline() -> MedQueryFlowPipeline:
    return MedQueryFlowPipeline(Path("configs/config.yaml"))


def ensure_api_key() -> None:
    """Prompt the user for an LLM provider API key before loading the app."""
    st.session_state.setdefault("api_key_status", "pending")
    st.session_state.setdefault("api_key_skipped", False)

    provider_env_map = {
        "OpenAI": "OPENAI_API_KEY",
        "Anthropic": "ANTHROPIC_API_KEY",
        "Gemini": "GEMINI_API_KEY",
        "DeepSeek": "DEEPSEEK_API_KEY",
    }
    known_envs = list(provider_env_map.values())

    if any(os.environ.get(var) for var in known_envs):
        st.session_state["api_key_status"] = "set"
        st.session_state["api_key_skipped"] = False
        return

    if st.session_state["api_key_status"] == "skipped":
        return

    st.session_state.setdefault("api_key_provider", "OpenAI")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            "<div style='text-align: center;'>"
            "<h2>🔐 Welcome to MedQueryFlow</h2>"
            "<p>Please choose your LLM provider and enter an API key to unlock full features.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        provider = st.selectbox(
            "Select provider",
            ["OpenAI", "Anthropic", "Gemini", "DeepSeek", "Other"],
            key="api_key_provider",
        )
        api_key = st.text_input("API key", type="password", key="api_key_input")
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            save_clicked = st.button("💾 Save and Continue")
        with btn_col2:
            skip_clicked = st.button("⏭️ Skip (limited mode)")

        if save_clicked:
            if not api_key:
                st.error("Please enter your API key before continuing.")
            else:
                env_name = provider_env_map.get(provider, "GENERIC_LLM_API_KEY")
                os.environ[env_name] = api_key
                st.session_state["api_key_status"] = "set"
                st.session_state["api_key_skipped"] = False
                st.success(
                    f"{provider} API key saved for this session. Click the \"Rerun\" button to continue."
                )
                st.stop()

        if skip_clicked:
            st.session_state["api_key_status"] = "skipped"
            st.session_state["api_key_skipped"] = True
            st.experimental_rerun()

    st.stop()


def render_documents(docs):
    if not docs:
        st.warning("No documents retrieved. Populate data/medical_abstracts/abstracts.json.")
        return
    for doc in docs:
        with st.expander(f"{doc['doc_id']} · {doc['title']} (score={doc['score']:.3f})"):
            st.write(doc["text"])


def render_run(run_payload: dict, title: str):
    st.subheader(title)
    st.markdown(f"**Intent:** {run_payload['intent']['intent']} (confidence {run_payload['intent']['confidence']:.2f})")
    st.markdown(f"**Style profile:** {run_payload['intent']['style_profile']}")
    st.markdown(f"**Routing target:** {run_payload['intent']['routing_target']}")
    st.markdown("**Query rewrites:**")
    st.code(json.dumps(run_payload["rewrites"], ensure_ascii=False, indent=2))
    if run_payload["applied_normalizations"]:
        st.markdown("**Applied terminology normalization:**")
        st.code(json.dumps(run_payload["applied_normalizations"], ensure_ascii=False, indent=2))
    render_documents(run_payload["retrieved_documents"])
    safety = run_payload["safety"]
    if safety["triage_level"] == "urgent":
        st.error(
            "Urgent triage triggered. "
            + (safety.get("highlight_message") or "Immediate clinical escalation recommended.")
        )
        if safety.get("matched_keywords"):
            st.caption(f"Matched urgent signals: {', '.join(safety['matched_keywords'])}")
        if safety.get("escalation_message"):
            st.markdown(safety["escalation_message"])
    elif safety["triage_level"] == "caution":
        st.warning(safety.get("highlight_message") or "Monitor symptoms closely and seek care if they worsen.")
        if safety.get("matched_keywords"):
            st.caption(f"Matched caution signals: {', '.join(safety['matched_keywords'])}")
    else:
        st.info(f"Safety triage: {safety['triage_level']}")
    if safety.get("compliance_flags"):
        st.markdown("**Compliance gates triggered:**")
        st.code("\n".join(safety["compliance_flags"]))
    st.markdown("**Answer:**")
    st.write(run_payload["answer"])
    if run_payload.get("export_path"):
        st.success(f"Run exported to {run_payload['export_path']}")


def render_dashboard(pipeline: MedQueryFlowPipeline) -> None:
    st.subheader("Analytics dashboard")
    export_dir = Path(pipeline.config["logging"]["export_dir"])
    if not export_dir.exists():
        st.info("No exports available yet. Enable run export to populate analytics.")
        return

    history = []
    for export_path in sorted(export_dir.glob("*.json")):
        try:
            payload = json.loads(export_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        docs = payload.get("retrieved_documents", [])
        avg_score = sum(doc.get("score", 0.0) for doc in docs) / len(docs) if docs else 0.0
        history.append(
            {
                "timestamp": payload.get("timestamp", export_path.stem),
                "variant": payload.get("variant", "unknown"),
                "retrieved_count": len(docs),
                "avg_score": avg_score,
                "triage_level": payload.get("safety", {}).get("triage_level", "info"),
            }
        )

    if not history:
        st.info("Exports found, but no readable metrics were extracted.")
        return

    triage_priority = {level: idx for idx, level in enumerate(pipeline.config.get("safety", {}).get("triage_levels", []))}
    for entry in history:
        entry["triage_index"] = triage_priority.get(entry["triage_level"], 0)

    variants = sorted({item["variant"] for item in history})
    selected_variant = st.selectbox("Filter by variant", ["All"] + variants)
    filtered = [item for item in history if selected_variant == "All" or item["variant"] == selected_variant]

    st.markdown("**Retrieval metrics over time**")
    if filtered:
        retrieval_chart = {
            "retrieved_count": [item["retrieved_count"] for item in filtered],
            "avg_score": [item["avg_score"] for item in filtered],
        }
        st.line_chart(retrieval_chart)
        st.dataframe(filtered)
    else:
        st.info("No runs match the current filter.")

    st.markdown("**Prompt A/B outcomes**")
    variant_summary: Dict[str, Dict[str, float]] = {}
    for item in history:
        summary = variant_summary.setdefault(item["variant"], {"runs": 0, "avg_retrieved": 0.0, "avg_score": 0.0, "avg_triage": 0.0})
        summary["runs"] += 1
        summary["avg_retrieved"] += item["retrieved_count"]
        summary["avg_score"] += item["avg_score"]
        summary["avg_triage"] += item["triage_index"]
    for summary in variant_summary.values():
        if summary["runs"]:
            summary["avg_retrieved"] /= summary["runs"]
            summary["avg_score"] /= summary["runs"]
            summary["avg_triage"] /= summary["runs"]
    triage_levels = pipeline.config.get("safety", {}).get("triage_levels", [])

    def avg_triage_label(value: float) -> str:
        if not triage_levels:
            return f"{value:.2f}"
        idx = int(round(value))
        idx = max(0, min(idx, len(triage_levels) - 1))
        return triage_levels[idx]

    summary_rows = []
    for variant, metrics in variant_summary.items():
        summary_rows.append(
            {
                "variant": variant,
                "runs": metrics["runs"],
                "avg_retrieved": round(metrics["avg_retrieved"], 2),
                "avg_score": round(metrics["avg_score"], 3),
                "avg_triage_level": avg_triage_label(metrics["avg_triage"]),
            }
        )

    st.table(summary_rows)


ensure_api_key()

if st.session_state.get("api_key_skipped"):
    st.warning(
        "⚠️ Running without an API key — LLM-based features will be disabled or use local fallback."
    )

st.title("MedQueryFlow")
st.caption("Prompt-orchestrated medical retrieval and QA pipeline demo")

pipeline = load_pipeline()
variants = [variant.name for variant in pipeline.variants]

run_tab, analytics_tab = st.tabs(["Run pipeline", "Analytics"])

with run_tab:
    left, right = st.columns(2)
    with left:
        variant_name = st.selectbox("Primary variant", variants)
    with right:
        ab_mode = st.toggle("A/B compare two variants")

    single_playback = st.toggle("Single sample playback")
    question = ""
    if single_playback:
        samples = pipeline.load_samples()
        if not samples:
            st.warning("Add sample runs to data/sample_runs.json to enable playback.")
        else:
            selected = st.selectbox(
                "Choose saved run",
                range(len(samples)),
                format_func=lambda idx: samples[idx]["question"],
            )
            question = samples[selected]["question"]
            st.info("Loaded question from saved sample. You can still edit below if needed.")

    question = st.text_area("User question", value=question, height=120, placeholder="胸闷很闷是吃坏了吗？")
    enable_rewrites = st.checkbox("Enable query rewrites", value=True)
    export_run = st.checkbox("Export run to JSON", value=False)

    if ab_mode:
        variant_b = st.selectbox("Comparison variant", variants, index=min(1, len(variants) - 1))
    else:
        variant_b = None

    if st.button("Run pipeline", type="primary"):
        if not question.strip():
            st.warning("Please provide a question")
        else:
            with st.spinner("Processing..."):
                primary_run = pipeline.run(
                    question=question,
                    variant_name=variant_name,
                    enable_rewrites=enable_rewrites,
                    export=export_run,
                )
                render_run(primary_run, f"Variant: {variant_name}")
                if ab_mode and variant_b:
                    comparison_run = pipeline.run(
                        question=question,
                        variant_name=variant_b,
                        enable_rewrites=enable_rewrites,
                        export=export_run,
                    )
                    render_run(comparison_run, f"Variant: {variant_b}")
                    diffs = {
                        "rewrites": len(comparison_run["rewrites"]) - len(primary_run["rewrites"]),
                        "retrieved_docs": len(comparison_run["retrieved_documents"]) - len(primary_run["retrieved_documents"]),
                    }
                    st.markdown("**A/B delta summary**")
                    st.code(json.dumps(diffs, indent=2))

with analytics_tab:
    render_dashboard(pipeline)

st.sidebar.header("Implementation notes")
st.sidebar.write(
    "Populate the following assets manually for production-quality runs:"\
)
st.sidebar.write(
    "1. data/term_normalization.csv — add columns `informal,canonical` with your symptom lexicon."\
)
st.sidebar.write(
    "2. data/medical_abstracts/abstracts.json — drop your curated medical corpus here."\
)
st.sidebar.write(
    "3. data/sample_runs.json — optional saved sessions for playback demos."\
)
st.sidebar.write(
    "4. Configure real LLM credentials by replacing modules/llm_client.py backend implementation."\
)
