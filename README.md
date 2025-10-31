# MedQueryFlow

MedQueryFlow is a prompt-orchestrated medical retrieval and question-answering pipeline
showcasing how prompt engineering, retrieval-augmented generation (RAG), and
workflow instrumentation come together to deliver safe, contextualized responses.

## System Goals
- Route user intent across medical fact lookup, diagnostic exploration, actionable advice, and emotional support.
- Amplify recall through prompt-driven query rewriting and terminology normalization.
- Retrieve evidence from curated medical corpora and inject context into controlled prompts.
- Enforce safety triage for urgent scenarios while tailoring tone per intent.
- Support A/B experimentation, run exports, and prompt versioning to mirror production PromptOps.

## Architecture Overview
```
User Question
   │
   ▼
[Intent Classifier] ──► Intent routing + style control
   │
   ▼
[Query Rewriter] ──► Prompted multi-query generation + normalization table
   │
   ▼
[RAG Retriever] ──► TF-IDF demo index (swap with FAISS/LlamaIndex in prod)
   │
   ▼
[Answer Generator] ──► Safety-aware prompt with tone modulation
   │
   ▼
[Streamlit Viewer] ──► Single run + A/B comparison + exports
```

## Repository Layout
```
├── app.py                    # Streamlit front-end with playback + A/B toggles
├── core/
│   └── pipeline.py           # Intent → rewrite → retrieve → answer orchestration
├── modules/
│   ├── llm_client.py         # Swappable client (stubbed by default)
│   ├── intent_classifier.py  # Prompt-based classifier with heuristics fallback
│   ├── query_rewriter.py     # Multi-query rewrite + terminology normalization
│   ├── rag_retriever.py      # TF-IDF retriever (drop-in replacement ready)
│   └── answer_generator.py   # Style control + safety triage + prompt versioning
├── configs/
│   └── config.yaml           # Routing rules, AB variants, safety settings
├── data/
│   ├── medical_abstracts/    # Demo medical abstracts (replace with your corpus)
│   ├── term_normalization.csv# Symptom lexicon for rewrite normalization
│   └── sample_runs.json      # Optional playback runs for the UI toggle
├── prompts/                  # Versionable prompt templates
├── utils/                    # Logging, metrics, exporter helpers
└── requirements.txt
```

## Running the Demo
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Launch Streamlit:
   ```bash
   streamlit run app.py
   ```
3. Explore the UI:
   - Toggle **Single sample playback** to load canned queries.
   - Toggle **A/B compare** to run the same question across variants.
   - Inspect rewrites, normalization hits, retrieved documents, and safety triage outcomes.
   - Optionally export runs to timestamped JSON with prompt/index/safety version metadata.

## Bring Your Own LLM + PromptOps
- Replace `modules/llm_client.RuleBasedLLM` with your API-backed client (OpenAI, Anthropic, etc.).
- Connect your prompt management platform by wiring prompt IDs/versions into `configs/config.yaml`.
- Use the exported JSON (`data/exports/`) to compare prompt variants, retrieval strategies, or safety rulesets.

## Evaluation Hooks
- `utils/metrics.py` ships recall@k and nDCG@k helpers for offline experiments.
- The Streamlit A/B toggle exposes a quick qualitative diff; plug the same metrics into notebooks for reporting.

## Manual Assets to Supply
To move beyond the demo corpus, populate:
1. `data/term_normalization.csv` with your UMLS/MeSH mappings.
2. `data/medical_abstracts/abstracts.json` with curated medical passages.
3. `data/sample_runs.json` for scripted demo playback.
4. `.env` or environment variables with API keys, then swap in a real `LLMClient` backend.

## Extensibility Ideas
- Add LLM-based rerankers or chain-of-thought verification steps.
- Introduce safety classifiers for additional compliance gates (e.g., medication dosage warnings).
- Integrate dashboards to visualize prompt A/B outcomes and retrieval metrics over time.

