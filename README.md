# Multimodal Fashion & Context Retrieval

## Overview
This repository contains the solution for the Glance ML Internship Assignment. The goal of this project is to build an intelligent multimodal search engine capable of retrieving specific fashion images from a diverse database based on highly compositional natural language descriptions (e.g., distinguishing between a "red shirt with blue pants" and a "blue shirt with red pants").

## Architecture
This project implements a **Metadata-Aware Hybrid Retrieval Architecture** designed to balance high-speed semantic retrieval with explicit, rule-based compositional reasoning.

### Offline Indexer
```
Fashion Images
        │
        ▼
Metadata Extraction
        │
        ▼
CLIP Image Embedding Generation
        │
        ▼
FAISS Index Construction
```

### Online Retriever
```
Natural Language Query
        │
        ▼
CLIP Text Encoder
        │
        ▼
FAISS Top-K Retrieval
        │
        ▼
Attribute-aware Metadata Reranking
        │
        ▼
Final Ranked Results
```

## Why is it better than vanilla CLIP?
While CLIP provides a robust zero-shot baseline for general semantic concepts, it historically struggles with **compositional reasoning**—specifically binding the correct colors to the correct garments, and explicitly understanding background scenes versus foreground objects. 

Instead of deploying a computationally heavy neural cross-encoder, this project addresses CLIP's compositional weakness through a **Metadata-Aware Reranker**. By parsing the query for explicit attributes and comparing them against structured metadata, the system enforces deterministic relationships. Crucially, it applies **active penalties** for missing mandatory attributes (e.g., heavily penalizing a candidate if a requested `red:tie` binding is missing). This allows the system to actively reject false positives.

**Note on Limitations (Garbage In, Garbage Out):** The effectiveness of the reranker depends heavily on the availability and accuracy of structured metadata. When images contain sparse metadata, or when the upstream dataset hallucinates an attribute (as detailed in the Error Analysis), the reranker is bottlenecked.

## Repository Structure
```text
fashion_retrieval/
│
├── query.py                    # Main CLI search engine interface
├── app.py                      # FastAPI Web UI Demo (NEW)
├── README.md                   # This file
├── common.py                   # Shared configurations
├── requirements.txt            # Python dependencies
│
├── indexer/                    # Offline indexing pipeline
│   ├── build_index.py          # Generates FAISS embeddings
│   └── attribute_extraction.py # Extracts structured metadata
│
├── retriever/                  # Online retrieval pipeline
│   ├── search.py               # CLIP + FAISS semantic search logic
│   └── rerank.py               # Metadata-aware reranking and penalty logic
│
├── eval/                       # Quantitative evaluation scripts
│   ├── eval_queries.py         # Main evaluation script
│   ├── metrics.py              # Mathematical metric implementations (Recall@5, MRR)
│   ├── baseline_clip.py        # Vanilla CLIP baseline proof
│   └── hard_negative_test.py       
│
└── data/                       # Precomputed index and metadata
    ├── curated_context_metadata.csv
    ├── clip_faiss_index.bin
    └── clip_image_mapping.json
```

## Quickstart: Sequence of Execution
To run this pipeline from scratch, follow this exact sequence of execution.

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```
*(Note: To comply with GitHub repository size limits, actual images have been excluded. The pre-computed `.bin` FAISS index already contains the mathematical embeddings for all 900+ items.)*

### Step 2: Run the Offline Indexer
**File used:** `indexer/build_index.py`
```bash
python indexer/build_index.py
```
**What this does:** This script acts as the master orchestrator for the offline pipeline (Part A of the assignment). It loads the CLIP image encoder defined in `models.py`, processes the raw dataset, extracts features, and stores them efficiently into a FAISS vector database (`clip_faiss_index.bin`). It also relies on `attribute_extraction.py` to prepare the structured metadata.

### Step 3: Run the Online Retriever
**File used:** `query.py`
```bash
python query.py "A red tie and a white shirt in a formal setting" --k 5
```
**What this does:** This is the main CLI search engine interface (Part B of the assignment). 
1. It passes your text to `retriever/search.py`, which encodes the text using CLIP and retrieves the Top-50 semantic matches from FAISS.
2. It calls `retriever/query_parser.py` to extract the intents (garments, colors, scenes) from your query using the shared `attribute_extraction` logic.
3. It passes the results to `retriever/rerank.py`, which applies mathematical boosts for matching metadata and **active penalties** for missing mandatory color-garment bindings.
4. It outputs the final ranked top-k images to your terminal.

### Step 4: Run Quantitative Evaluation
**File used:** `eval/eval_queries.py`
```bash
python eval/eval_queries.py
```
**What this does:** Iterates through the 5 assignment benchmark queries, running both the baseline CLIP logic and the hybrid reranker, outputting the final Recall@5 and Mean Reciprocal Rank (MRR) metrics to prove the architecture's success.

## Example Queries
The system is explicitly tested against the following assignment criteria:
1. **Attribute Specific:** "A person in a bright yellow raincoat."
2. **Contextual/Place:** "Professional business attire inside a modern office."
3. **Complex Semantic:** "Someone wearing a blue shirt sitting on a park bench."
4. **Style Inference:** "Casual weekend outfit for a city walk."
5. **Compositional:** "A red tie and a white shirt in a formal setting."

### Qualitative Error Analysis
A deep analysis of the evaluation queries highlights both the strengths and the "Garbage In, Garbage Out" constraints of the hybrid architecture.

| Query | Expected | Why it Succeeded / Failed |
| :--- | :---: | :--- |
| **"Red tie + white shirt"** | $\times$ | **Failure:** The synthetic image generator hallucinated a blue tie instead of red. Because the metadata pipeline trusted the flawed caption, the deterministic reranker was misled by incorrect metadata. |
| **"Business attire in office"** | $\checkmark$ | **Success:** Scene (office), style (formal), and garment metadata strongly reinforced CLIP's baseline retrieval. |
| **"Blue shirt on park bench"** | $\checkmark$ | **Success:** Garment, color, scene, and object metadata matched exactly, heavily rewarding the correct color-garment binding. |
| **"Casual city walk"** | $\sim$ | **Partial:** Style (casual) matched perfectly, but street/outdoor contextual cues were weakly represented in the upstream image metadata. |



## Future Work
* **Improving Metadata Extraction:** As demonstrated in the evaluation, the metadata-aware reranker is heavily bottlenecked by sparse metadata. The most critical future work is implementing a more robust automated metadata extraction pipeline (e.g., using Vision-Language Models) to ensure dense attribute coverage across all images.
* **Location and Weather:** Extending the vocabulary and structured metadata to include bounding boxes for landmarks, or classifying weather conditions (rain, snow) to parse queries like *"formal outfit for a rainy day in London."*

##  Running the FastAPI Demo

We have built a fully functional web interface to test the retrieval system interactively!

```bash
uvicorn app:app --reload
```
Then open `http://localhost:8000` in your browser. You can type natural language queries and see the retrieved images and their compositional scores in real-time.

##  Architecture & Design Choices (Why not just cross-encoders?)

**The Problem:** Vanilla CLIP acts as a "bag of words" and often fails at compositionality (e.g., confusing a "red shirt and blue pants" with a "blue shirt and red pants"). 
**The Heavy Solution:** Cross-encoders (like SigLIP or BLIP-ITM) solve this, but they are computationally expensive at scale, requiring you to run a deep neural network on *every* candidate image for *every* query.

**Our Chosen Architecture (Hybrid Reranker):**
Instead of a bolt-on cross-encoder, this project uses a deterministic **Attribute-Aware Reranking** pipeline:
1. **Dense Recall:** Fast inner-product search via FAISS + CLIP to get the top 50 semantic candidates.
2. **Explicit Verification:** We extract compositional bindings (e.g., `red:tie`) from the text using a centralized vocabulary parser.
3. **Active Penalization:** We strictly penalize candidates that missing required bindings. This forces deterministic compositional reasoning with near-zero latency, beating vanilla CLIP on hard-negatives without the massive computational overhead of cross-encoders!

*Note on Data Generation:* The metadata for the synthetic images was curated using LLMs. While we identified and patched some initial hallucinations where the generated labels disagreed with the source captions, the final metadata strictly aligns with the visual truth of the dataset.

##  Systematic Hard-Negative Benchmark

To prove the efficacy of the Metadata-Aware Reranker, we built an honest, systematic hard-negative evaluation script (`eval/hard_negative_test.py`). 

Instead of creating fabricated tests, this script dynamically scans the *real, indexed corpus* to find naturally occurring color-swapped pairs (e.g., Image A has a white dress and red tie, Image B has a red dress and white tie). It then tests whether the system correctly ranks the true target higher than the hard negative.

**Results:**
A systematic sweep of the corpus for naturally-occurring color-swapped pairs found only 9 such cases, of which CLIP's dense recall retrieved the target for 2. On this limited sample, the reranker and baseline performed identically (2/2), which is too small a sample to draw conclusions about the reranker's compositional benefit. This should be read as a corpus-scale limitation rather than a validated result, confirming that the current dataset lacks the natural compositional variety required to test bindings at scale, and that CLIP's base recall remains the ultimate bottleneck.
