# MultiModal-Fashion-and-Context-Retreival-
Building an intelligent search engine that can retrieve specific  images from a diverse database based on natural language descriptions. The system understands what someone is wearing at which place and what's the vibe of attire.

# Architecture
                Query
                  │
                  ▼
          CLIP Text Encoder
                  │
                  ▼
               FAISS
                  │
             Top-50 Images
                  │
                  ▼
        BLIP ITM Cross-Attention
                  │
                  ▼
      Metadata-Aware Fashion Rules
                  │
                  ▼
             Final Ranking
