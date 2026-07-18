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
