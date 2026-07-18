# Retriever Module

This module handles the online inference pipeline for the multimodal fashion retrieval system. It implements a two-stage **Retrieve-and-Rerank** architecture designed to balance scalability (searching across millions of images) with high-precision compositional reasoning.

## Files

### `search.py`
**Stage 1: Semantic Retrieval (Candidate Generation)**
- Builds query embeddings using pre-trained vision-language models (CLIP/BLIP).
- Performs ultra-fast, dense Approximate Nearest Neighbor (ANN) search using **FAISS**.
- Returns the top-*k* semantically relevant images to form the candidate pool.

### `rerank.py`
**Stage 2: Attribute-Aware Reranking**
- Parses the natural language query to extract structured attributes (colors, garments, scenes, styles, objects, and demographics).
- Actively binds modifiers (e.g., matching "red" specifically to "tie").
- Reads the ground-truth metadata for candidate images.
- Computes an attribute-aware reranking score (System C) to correct compositional mistakes made by vanilla CLIP/BLIP, producing the final high-precision ranked list.

### `vocabulary.py`
- Stores the domain-specific vocabulary and synonym mappings used by the query parser to guarantee zero-shot robustness on unseen phrasing.
