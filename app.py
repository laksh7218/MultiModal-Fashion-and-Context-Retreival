import streamlit as st
import os
import sys
from pathlib import Path
from PIL import Image

# Setup paths
BASE_DIR = Path(__file__).parent
sys.path.append(str(BASE_DIR))

try:
    from retriever.search import MultimodalRetriever
    from retriever.rerank import rerank_top_k
except ImportError as e:
    st.error(f"Import Error: {e}")
    st.stop()

st.set_page_config(page_title="Glance ML - Fashion Retrieval", layout="wide")

st.title("👗 Multimodal Fashion & Context Retrieval")
st.markdown("Interactive demo for the **Three-Stage Neural Hybrid Retrieval System**.")

@st.cache_resource
def get_retriever():
    return MultimodalRetriever()

with st.spinner("Loading models and FAISS indices..."):
    try:
        retriever = get_retriever()
    except Exception as e:
        st.error(f"Error loading retriever: {e}")
        st.stop()

# Helper to find image
def get_image_path(img_name):
    # Try different image directories
    possible_dirs = [
        BASE_DIR / "data" / "curated_images",
        BASE_DIR / "data" / "images",
        BASE_DIR / "data" / "fashionpedia"
    ]
    for d in possible_dirs:
        p = d / img_name
        if p.exists():
            return p
        # if the img_name is already a path or includes extension
        p2 = d / f"{img_name}.jpg"
        if p2.exists():
            return p2
        # sometimes idx_to_image holds absolute paths
        if Path(img_name).exists():
            return Path(img_name)
    return None

query = st.text_input("Enter a fashion query:", "Red tie and white shirt in formal setting")
top_k = st.slider("Top K results to display:", min_value=1, max_value=50, value=5)

if st.button("Search"):
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        st.markdown(f"### Results for: *{query}*")
        
        col1, col2 = st.columns(2)
        
        with st.spinner("Searching..."):
            # System A: CLIP
            clip_results = retriever.search_clip(query, k=top_k)
            
            # System C: Hybrid Architecture (CLIP Candidates + Attribute Reranking)
            candidate_pool_size = max(50, top_k * 2)
            clip_candidates_for_rerank = retriever.search_clip(query, k=candidate_pool_size) 
            hybrid_results = rerank_top_k(clip_candidates_for_rerank, query)[:top_k]
            
            with col1:
                st.subheader("System A: CLIP Baseline")
                if not clip_results:
                    st.write("No results found.")
                for i, res in enumerate(clip_results):
                    img_id = res['image_id']
                    score = res['score']
                    img_path = get_image_path(img_id)
                    st.write(f"**Rank {i+1}** (Score: {score:.4f}) | `{img_id}`")
                    if img_path:
                        st.image(Image.open(img_path), use_container_width=True)
                    else:
                        st.write("*(Image file not found)*")
                        
            with col2:
                st.subheader("System C: Hybrid Architecture")
                if not hybrid_results:
                    st.write("No results found.")
                for i, res in enumerate(hybrid_results):
                    img_id = res['image_id']
                    score = res['score']
                    img_path = get_image_path(img_id)
                    st.write(f"**Rank {i+1}** (Score: {score:.4f}) | `{img_id}`")
                    if img_path:
                        st.image(Image.open(img_path), use_container_width=True)
                    else:
                        st.write("*(Image file not found)*")
