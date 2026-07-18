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
├── requirements.txt 
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

Execution Sequence:
Step 1: 
```
pip install -r requirements.txt
```
Step 2: Run indexer 
```
python indexer/build_index.py
```
It loads the CLIP image encoder defined in models.py, processes the raw dataset, extracts features, and stores them efficiently into a FAISS vector database (clip_faiss_index.bin). It also relies on attribute_extraction.py to prepare the structured metadata.

Step 3: Run the Retriever 
1) It passes your text to retriever/search.py, which encodes the text using CLIP and retrieves the Top-50 semantic matches from FAISS.
2) It calls retriever/query_parser.py to extract the intents (garments, colors, scenes) from your query using vocabulary.py.
3) It passes the results to retriever/rerank.py, which applies mathematical boosts for matching metadata and active penalties for missing mandatory color-garment bindings.
4) It outputs the final ranked top-k images to your terminal.

Step 5: Run evaluation
```
python eval/eval_queries.py
```



