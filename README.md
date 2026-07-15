# Mastering Generative Engine Optimization (GEO) in B2B Enterprise IT

## Abstract

This repository contains the codebase, data pipeline, and evaluation framework for a Master's thesis investigating Generative Engine Optimization (GEO) in the B2B Enterprise IT-context. The research empirically evaluates the efficacy of various textual optimization strategies in enhancing the visibility, citation rates, and subjective quality of B2B content when processed by Large Language Models (LLMs) and Generative Search Engines (GSEs).

## Repository Architecture

The repository is organized to ensure the separation of data processing, execution logic, and evaluation metrics in accordance with scientific reproducibility standards.

- **`Masterthesis/`**: The primary workspace directory.
  - **`01_Datapreparation/`**: Contains the systematically generated B2B dataset (600 queries segmented by Focus Topic, Customer Journey Phase, and Persona) alongside the raw scraped HTML baseline data.
  - **`03_Visibility Metrics and Results/`**: Stores the processed empirical results (`absolute_visibility_results.csv`, `perplexity_results_server.csv`) and the GEval prompt templates utilized by the evaluator LLM.
  - **`logs and cache/`**: Maintains JSON cache files (e.g., LLM responses, optimization caches) and execution logs to prevent redundant API calls during replication.
  - **`src/`**: The primary source code directory containing the sequential execution pipeline.

## Execution Pipeline

The experimental methodology is implemented via a sequence of Python scripts located in the `Masterthesis/src/` directory. To replicate the empirical findings, the scripts must be executed in the following numerical order:

1. `01_dataset_user_query_generation.py`: Synthesizes realistic search queries for the defined B2B dataset matrix.
2. `02_scrape_full.py`: Retrieves and processes organic search results to establish an unoptimized baseline corpus.
3. `03_build_global_cache.py`: Formats the raw web-scraped data into a standardized JSON structure for LLM context injection.
4. `04_generate_answers.py`: Queries the target Generative Search Engine (local LLM) to generate the baseline answers without source modification.
5. `05_run_geo.py`: Applies distinct Generative Engine Optimization (GEO) strategies (e.g., Inverted Pyramid, Fluency, Statistical Enrichment) to the source texts and regenerates the LLM responses.
6. `06_run_geval.py`: Executes the core evaluation protocol, employing a judge-LLM approach to calculate subjective visibility metrics (GEval) via multi-threaded concurrency.
7. `07_calculate_absolute_scores.py`: Computes deterministic, absolute visibility metrics (e.g., Citation Count, Position-Adjusted Word Count) across all generated responses.
8. `08_server_eval_perplexity.py`: Evaluates the linguistic perplexity of the generated responses utilizing a causal language model on dedicated GPU hardware.
9. `09_quality_review.py`: Conducts an automated quality review, identifying and mitigating LLM hallucinations or anomalies within the generated responses.
10. `10_calculate_heavy_metrics_absolute_remote.py`: Computes computationally intensive absolute metrics on remote hardware to enhance processing efficiency.
11. `11_run_baseline_v2_remote.py`: Executes an updated iteration of the baseline generation protocol (V2) utilizing distributed computation.
12. `12_run_geval_baseline_v2_remote.py`: Evaluates the updated baseline (V2) responses utilizing the standardized GEval subjective evaluation methodology.

## Evaluated Optimization Strategies

The research investigates the efficacy of 10 distinct Generative Engine Optimization (GEO) methodologies compared against an unoptimized baseline (`identity`). The strategies are designed to manipulate specific stylistic, structural, and semantic elements of the source text:

1. **Fluency Optimization (`fluent_gpt`)**: Enhances readability and grammatical flow without altering the core semantic meaning.
2. **Authoritative Tone (`authoritative`)**: Modifies the text to adopt a highly professional, expert, and definitive voice.
3. **Quotation Addition (`more_quotes`)**: Integrates relevant citations, expert quotes, and verifiable claims.
4. **Credible Source Citation (`citing_credible`)**: Embeds references to established, high-authority domain entities.
5. **Language Simplification (`simple_language`)**: Reduces linguistic complexity to improve accessibility for non-expert personas.
6. **Technical Terminology (`technical_terms`)**: Enriches the text with domain-specific jargon and technical depth.
7. **Statistical Enrichment (`stats_optimization`)**: Incorporates quantitative data, metrics, and empirical evidence to substantiate claims.
8. **LLMs.txt Formatting (`llms_txt`)**: Structures the content according to the machine-readable `llms.txt` specification to facilitate optimized parsing by LLM crawlers.
9. **Inverted Pyramid Structure (`inverted_pyramid_mine`)**: Reorganizes the document to present the most critical information first, followed by supporting details.
10. **AutoGEO API Extraction (`autogeo_api_mine`)**: Employs dynamically extracted, engine-specific preference rules to automatically rewrite the content for maximum visibility.

## Evaluation Metrics

The repository employs a multi-faceted evaluation framework to quantify GEO effectiveness across deterministic, subjective, and linguistic dimensions:

- **Absolute Visibility (Fast Metrics)**: Objective metrics including ChatCite, SentenceCite, Position-Adjusted Word Count (PAWC), and Absolute Word Count.
- **Absolute Visibility (Heavy Metrics)**: Computationally intensive textual metrics including ROUGE, BLEU, Length Ratio, and Relative Visibility Score (RVS).
- **Subjective Quality (GEval)**: LLM-as-a-judge methodologies evaluating nuanced criteria including Subjective Position, Subjective Citation Count, Relevance, Diversity, Uniqueness, Follow-up Likelihood, and Influence based on standardized evaluation prompts.
- **Linguistic Quality**: Perplexity computations (via causal language models) to ensure optimizations do not degrade the natural fluency of the generated text.
- **Unified Performance (Meta-Z-Score)**: A composite standardizing metric that aggregates multiple visibility and quality scores, facilitating a holistic evaluation of optimization effectiveness across disparate scales.

## Interactive Data Visualization (Dashboard)

To facilitate the exploratory analysis of the generated results, an interactive visualization dashboard is provided within the `Masterthesis/Dashboard/` directory. The `app.py` script initializes a Gradio-based web application that parses the empirical output files and renders dynamic visualizations (e.g., heatmaps, bar charts, and statistical clusters).

The analytical framework of the dashboard is structured into 8 dedicated modules:

1. **Baseline Aggregation**: Defines and visualizes the unoptimized baseline average across all target metrics.
2. **Visibility Shift Overview**: High-level comparison of absolute and relative visibility gains achieved by each GEO strategy.
3. **Performance Variance Analysis**: Statistical evaluation of performance differences between strategies, including significance clustering (Tukey Post-Hoc).
4. **Textual Modification Assessment**: Granular inspection of document modifications, highlighting representative examples, top-performers, and worst-performers.
5. **Segment-Specific Influence**: Dynamic filtering and analysis based on B2B metadata (Focus Topic, Persona, and Customer Journey Phase).
6. **Detailed Metric Analysis**: Isolated examination of specific absolute and subjective visibility metrics across all strategies.
7. **Source Displacement Analysis**: Evaluation of cannibalization effects and visibility shifts between the target domain, specific competitors, and unrelated sources.

To launch the dashboard locally, execute:
```bash
gradio Dashboard/app.py
```

## Setup and Reproducibility

1. Clone the repository to the local environment or server.
2. Initialize a virtual environment and install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure a `.env` file in the root directory containing necessary environment variables (e.g., `OPENAI_API_KEY`).
4. Ensure a compatible LLM (e.g., `qwen2.5:7b` via Ollama) is running locally to facilitate the generation and GEval pipeline.

## Acknowledgments and Citation

This research builds upon the foundational Generative Engine Optimization (GEO) framework. If this repository or the incorporated GEO optimization methods are utilized, please acknowledge the original authors:

```bibtex
@misc{aggarwal2023geo,
      title={GEO: Generative Engine Optimization}, 
      author={Pranjal Aggarwal and Vishvak Murahari and Tanmay Rajpurohit and Ashwin Kalyan and Karthik R Narasimhan and Ameet Deshpande},
      year={2023},
      eprint={2311.09735},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

Further details are available at the [official GEO GitHub Repository](https://github.com/GEO-optim/GEO) and the corresponding [project website](https://generative-engines.com/GEO/).

Additionally, if the AutoGEO methodology is utilized, please acknowledge the following work:

```bibtex
@inproceedings{wu2026generative,
  title={What Generative Search Engines Like and How to Optimize Web Content Cooperatively},
  author={Wu, Yujiang and Zhong, Shanshan and Kim, Yubin and Xiong, Chenyan},
  booktitle={The Fourteenth International Conference on Learning Representations (ICLR)},
  year={2026},
  url={https://openreview.net/forum?id=K8EinVWtUB}
}
```

Further details regarding AutoGEO are available at their [official GitHub Repository](https://github.com/cxcscmu/AutoGEO).

---
*Developed for academic research purposes.*
