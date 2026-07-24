# Advanced RAG

A hands-on, notebook-driven course in building **Retrieval-Augmented Generation (RAG)** systems from a naive semantic-search prototype to production-grade techniques like contextual retrieval, hybrid search, reranking, and multimodal (image-based) retrieval.

Each notebook is self-contained and builds intuition around one core idea, using small real datasets (wine reviews, arXiv AI papers, a product PDF user guide) so you can see *why* each technique improves retrieval quality.

## Notebooks

| # | Notebook | What it covers |
|---|----------|----------------|
| 01 | `01_simple_rag.ipynb` | A minimal RAG pipeline over a wine-ratings CSV: embed a column, retrieve rows by semantic similarity, and generate an answer. |
| 02 | `02_embedding_model.ipynb` | How text is encoded into vectors with `sentence_transformers`. Compares OpenAI embeddings, open-source input/output embeddings, and bi-encoders tuned for queries vs. documents. |
| 03 | `03_semantic_chunking.ipynb` | Semantic chunking with a rolling window of sentence embeddings  split documents where topic similarity drops instead of at fixed sizes. Uses arXiv AI papers. |
| 04 | `04_contextual_retrieval.ipynb` | Contextual retrieval: use an LLM to generate a context sentence for each chunk, enriching its embedding for better retrieval and hybrid search. |
| 05 | `05_reverse_hyde.ipynb` | Reverse HyDE: generate hypothetical queries from each document and embed those, closing the format/length gap between short queries and long documents. |
| 06 | `06_hybrid_search.ipynb` | Combine sparse keyword search (BM25) with dense semantic search for more robust retrieval on complex queries. |
| 07 | `07_reranking.ipynb` | Re-score initial retrieval results with a cross-encoder to surface the most relevant documents before generation. |
| 08 | `08_multimodal.ipynb` | Index and search image-based documents (a PDF user guide) with ColPali and Qdrant retrieving on visual layout rather than extracted text. |

Work through them in order; later notebooks assume the concepts introduced earlier.

## Data

The `data/` directory holds the datasets used across the notebooks:

- `top_rated_wines.csv`: structured wine reviews (name, region, variety, rating, notes) for the simple RAG demo.
- `corpus.json`: a corpus of arXiv AI paper text used for chunking, contextual retrieval, and search.
- `sparse_results.json` / `dense_results.json`: cached retrieval results used in the hybrid-search notebook.
- `shokz/openrun_pro_user_guide.pdf`: a product PDF used for multimodal image retrieval.

The `themes/` directory and the `*.svg` files at the repo root are visualization assets (rich console themes and embedding heatmaps / t-SNE plots) produced by the notebooks.

## Setup

Requires **Python 3.10+**.

```bash
# 1. Clone
git clone https://github.com/PallaviBhimte/advanced-rag.git
cd advanced-rag

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure your API key
echo "OPENAI_API_KEY=your-key-here" > .env

# 5. Launch Jupyter
jupyter notebook
```

### Environment variables

Notebooks that call OpenAI (embeddings and generation) read from a `.env` file:

```
OPENAI_API_KEY=your-key-here
```



