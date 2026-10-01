# Enhancing Catalogue Management in Manufacturing with Hybrid Retrieval and LLM-Based Insertion

<!--[![Journal](https://img.shields.io/badge/Journal-International_Journal_of_Intelligent_Manufacturing-success.svg)]()-->
[![Topic](https://img.shields.io/badge/Topic-Hybrid_Retrieval_&_LLMs-blue.svg)]()
[![Dataset](https://img.shields.io/badge/Dataset-Available-orange.svg)](https://github.com/DIOL-UniTN/manufacturing-catalogue-management-dataset)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

This repository contains the code for the paper **"Enhancing Catalogue Management in Manufacturing with Hybrid Retrieval and LLM-Based Insertion"**<!--, to be published in the **International Journal of Intelligent Manufacturing**-->.

## 📖 Overview

Industrial companies maintain **materials catalogues** containing hundreds of thousands of items, each identified by a code and described by short and long free-text descriptions, often in **multiple languages**. Keeping such a catalogue consistent is a costly, largely manual activity: before inserting a new item, an operator must verify that an equivalent item does not already exist, and, if it does not, write a new description that matches the conventions of the existing entries. Failing at either step produces **duplicate codes** and **inconsistent descriptions**, which in turn propagate into procurement, inventory, and production planning.

This work addresses both steps with a single pipeline, evaluated on the multilingual (Italian/English) materials catalogue of **an industrial manufacturing company**:

- **Selection** — given a free-text query written by an operator, retrieve the catalogue items that most likely correspond to it. We compare lexical, fuzzy, dense, and **hybrid retrieve-and-rerank** approaches.
- **Insertion** — if no existing item matches, prompt a **Large Language Model (LLM)** with the items retrieved in the previous step, used as in-context examples, to generate candidate Italian and English short descriptions that follow the catalogue's style and length conventions.

The two steps are connected: the retriever is not only the component that prevents duplicates, it also supplies the few-shot context that conditions the LLM, so retrieval quality directly affects the quality of the generated descriptions.

A **Flask web application** exposes the full pipeline to catalogue operators, and an **evaluation suite** reproduces the retrieval experiments reported in the paper.

## 🗂️ Repository Structure

```
.
├── app/                      # Flask web application and insertion pipeline
│   ├── app.py                # Web server: selection and insertion endpoints
│   ├── config.ini            # Retriever, LLM, catalogue path, language settings
│   ├── insertion.py          # LLM-based generation of candidate descriptions
│   ├── augmentation.py       # LLM-based augmentation of short/long descriptions
│   ├── APIs.py               # Unified wrapper over the LLM providers
│   ├── vec_db_functions.py   # Qdrant collection creation and vector search
│   ├── rel_db_functions.py   # Relational (SQLAlchemy) catalogue utilities
│   ├── static/               # Front-end assets
│   └── templates/            # Selection and insertion pages
├── retrievers/               # First-stage retrievers (lexical, fuzzy, dense, hybrid)
├── rankers/                  # Second-stage rankers and cross-encoder re-rankers
├── evaluator/                # MAP@k evaluation and rank-position analysis
├── requirements.txt
└── README.md
```

## 📦 Data

The catalogues and the evaluation datasets are released, in **pseudo-anonymised** form, in a companion repository:

```bash
git clone https://github.com/DIOL-UniTN/manufacturing-catalogue-management-dataset.git
```

Two kinds of files are used.

**Catalogue** — comma-separated, one row per item:

```
id,short_ita,short_eng,long_ita,long_eng
```

**Evaluation dataset** — comma-separated, one row per query, where `positive` is the identifier of the correct item, and the two negatives are distractors of increasing difficulty:

```
query,positive,hard_negative,soft_negative
```

Place the files under `catalogues/` and `datasets/` (both are git-ignored), or point to them explicitly through the command-line arguments.

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- [Qdrant](https://qdrant.tech/) (vector database, used by all dense retrievers)
- PyTorch (CUDA or MPS strongly recommended for the dense models)
- [Ollama](https://ollama.com/) (optional, only for locally served LLMs)

### Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Start Qdrant

```bash
docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
```

All retrievers connect to `localhost:6333`. On first use, each dense retriever creates and populates its own collection, named after the catalogue file and the embedding model; subsequent runs reuse it. To drop a collection and force re-indexing:

```bash
curl -X DELETE 'http://localhost:6333/collections/<collection-name>'
```

### API credentials

The LLM calls read their credentials from a `.env` file in the project root:

```
GITHUB_TOKEN=<your-token>
OLLAMA_MODEL=<local-model-name>
```

## ▶️ Running the Code

### Web application

```bash
flask --app app/app.py run --debug
```

The retriever, the LLM, the catalogue path, the number of returned items, and the catalogue language are configured in `app/config.ini`.

### Selection (retrieval) from the command line

```bash
cd retrievers
python Retriever.py \
    --catalogue ../catalogues/catalogue_01.csv \
    --query "vite sicurezza TCBEI A2 M6x16" \
    --model sbert_1024 \
    --output_length 10
```

### Insertion (description generation) from the command line

```bash
cd app
python insertion.py \
    --query "vite sicurezza TCBEI A2 M6x16 80" \
    --catalogue ../catalogues/catalogue_01.csv \
    --retriever sbert_1024 \
    --llm gpt-4.1
```

Passing `--llm all` queries every supported model in turn, which is how the insertion comparison in the paper was produced.

### Evaluation

```bash
python evaluator/evaluator.py \
    --dataset datasets/evaluation_dataset.csv \
    --dataset_name catalogue_01 \
    --catalogue catalogues/catalogue_01.csv \
    --model hybrid_sbert_1024_tfidf \
    --language ita \
    --output_length 10 \
    --output_folder evaluator/output
```

This reports **MAP@1**, **MAP@10**, and the average retrieval time per query, and writes to a timestamped folder:

- `results_<model>_k_1.csv`, `results_<model>_k_10.csv` — aggregate scores;
- `positions_<model>.csv` — the rank of the positive, hard-negative, and soft-negative item for each query;
- `retriever_error_<model>.csv` — the queries for which the positive item was not ranked first.

> **Note.** `evaluate_model_on_datasets` truncates the evaluation set to its first 100 rows. Remove the `dataset.head(100)` call in `evaluator/evaluator.py` to evaluate on the full dataset.

The distribution of the ranks written to `positions_<model>.csv` can then be plotted with:

```bash
python evaluator/position_analysis.py \
    --input_csv evaluator/output/<timestamp>/positions_hybrid_sbert_1024_tfidf.csv \
    --method hybrid_sbert_1024_tfidf \
    --dataset catalogue_01
```

## 📊 Methods

### Retrievers (`retrievers/`)

| Identifier | Method |
| --- | --- |
| `bm25` | BM25 lexical retrieval (`bm25s`) |
| `tfidf` | TF-IDF vector space model |
| `fuzzy_ratio`, `fuzzy_sort_ratio`, `fuzzy_set_ratio` | Fuzzy string matching (`thefuzz`) |
| `sbert_512` | `distiluse-base-multilingual-cased-v2` |
| `sbert_768` | `sentence-transformers/all-mpnet-base-v2` |
| `sbert_1024` | `intfloat/multilingual-e5-large` |
| `qwen_1024`, `qwen_2560`, `qwen_4096` | `Qwen/Qwen3-Embedding-{0.6B, 4B, 8B}` |
| `bge_dense`, `bge_sparse` | `BAAI/bge-m3`, dense and sparse variants |
| `gte` | `Alibaba-NLP/gte-multilingual-base` |
| `nomic` | `nomic-ai/nomic-embed-text-v2-moe` |
| `random` | Random baseline (seeded) |

### Rankers (`rankers/`)

Second-stage components that re-score a candidate list: `TfidfRanker`, `SbertRanker`, `Bm25Ranker`, `FuzzyRanker`, `CrossEncoderModel` (`cross-encoder/ms-marco-MiniLM-L-6-v2`, `ms-marco-TinyBERT-L-2-v2`), and `BGEReranker` (`bge_m3`, `bge_gemma`).

### Hybrid pipelines

`HybridRetriever` chains a first-stage retriever with a second-stage ranker. The two configurations used in the paper are:

- `hybrid_sbert_1024_tfidf` — dense retrieval of 100 candidates, re-ranked by TF-IDF;
- `hybrid_tfidf_sbert_1024` — TF-IDF retrieval of 100 candidates, re-ranked by dense similarity.

### LLMs for insertion

`gpt-4o`, `gpt-4o-mini`, `gpt-4.1`, `gpt-4.1-mini`, `Meta-Llama-3.1-405B-Instruct`, `Mistral-Large-2411`, `DeepSeek-V3-0324`, `grok-3`, and any model served locally through Ollama.

<!--## 📝 Citation

```bibtex
@article{genetti2026catalogue,
  title   = {Enhancing Catalogue Management in Manufacturing with Hybrid Retrieval and LLM-Based Insertion},
  author  = {Genetti, Stefano and Cecchin, Nicol\`o and Iacca, Giovanni},
  journal = {International Journal of Intelligent Manufacturing},
  year    = {2026}
}
```

## 🙏 Acknowledgements

We thank the industrial manufacturing company that provided the materials catalogue and the domain expertise used to build and validate the evaluation datasets.-->

## 👥 Authors

Stefano Genetti  
Nicolò Cecchin  
Giovanni Iacca

## 📄 License

Released under the [MIT License](LICENSE).
