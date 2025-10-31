from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from core.pipeline import MedQueryFlowPipeline
from utils.logger import setup_logging

setup_logging()


@st.cache_resource(show_spinner=False)
def load_pipeline() -> MedQueryFlowPipeline:
    return MedQueryFlowPipeline(Path("configs/config.yaml"))


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
            f"Urgent triage triggered. Keywords: {', '.join(safety['matched_keywords'])}\n{run_payload['safety']['escalation_message']}"
        )
    else:
        st.info(f"Safety triage: {safety['triage_level']}")
    st.markdown("**Answer:**")
    st.write(run_payload["answer"])
    if run_payload.get("export_path"):
        st.success(f"Run exported to {run_payload['export_path']}")


st.title("MedQueryFlow")
st.caption("Prompt-orchestrated medical retrieval & QA pipeline demo")

pipeline = load_pipeline()
variants = [variant.name for variant in pipeline.variants]

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
        selected = st.selectbox("Choose saved run", range(len(samples)), format_func=lambda idx: samples[idx]["question"])
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
