# MultiModal-Fashion-and-Context-Retreival-
Building an intelligent search engine that can retrieve specific  images from a diverse database based on natural language descriptions. The system understands what someone is wearing at which place and what's the vibe of attire.

# Architecture
Indexer

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
Retriever

```Natural Language Query
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
Repository structure
``` fashion_retrieval/
│
├── query.py                    # Main CLI search engine interface
├── README.md                   # This file
├── common.py                   # Shared configurations
│
├── indexer/                    # Offline indexing pipeline
│   ├── build_index.py          # Generates FAISS embeddings
│   ├── attribute_extraction.py # Extracts structured metadata
│   ├── models.py               # CLIP model definitions
│   └── config.py               # Indexer configurations
│
├── retriever/                  # Online retrieval pipeline
│   ├── search.py               # CLIP + FAISS semantic search logic
│   ├── rerank.py               # Metadata-aware reranking and penalty logic
│   ├── query_parser.py         # Parses text queries into attributes
│   └── vocabulary.py           # Domain-specific vocabulary/synonyms
│
├── eval/                       # Quantitative evaluation scripts
│   ├── eval_queries.py         # Main evaluation script
│   ├── metrics.py              # Mathematical metric implementations (Recall@5, MRR)
│   └── baseline_clip.py        # Vanilla CLIP baseline proof
│
└── data/                       # Precomputed index and metadata
    ├── curated_context_metadata.csv
    ├── clip_faiss_index.bin
    └── clip_image_mapping.json
```
