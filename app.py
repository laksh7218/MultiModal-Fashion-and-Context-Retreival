from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent
sys.path.append(str(BASE_DIR))
sys.path.append(str(BASE_DIR.parent))

from retriever.search import MultimodalRetriever
from retriever.rerank import rerank_top_k

app = FastAPI(title="Glance ML - Fashion Retrieval")

# Setup templates and static files if needed
templates_dir = BASE_DIR / "templates"
templates_dir.mkdir(exist_ok=True)

# Generate a simple HTML template
html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Glance ML - Fashion Retrieval</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; background: #f5f5f7; }
        .header { text-align: center; margin-bottom: 30px; }
        .search-box { display: flex; justify-content: center; margin-bottom: 40px; }
        input[type="text"] { width: 60%; padding: 15px; font-size: 18px; border: 1px solid #ccc; border-radius: 8px 0 0 8px; outline: none; }
        button { padding: 15px 30px; font-size: 18px; background: #0066cc; color: white; border: none; border-radius: 0 8px 8px 0; cursor: pointer; }
        button:hover { background: #0055aa; }
        .results { display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 20px; }
        .card { background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 12px rgba(0,0,0,0.1); padding: 15px; text-align: center; }
        .card img { width: 100%; height: 200px; object-fit: cover; border-radius: 8px; margin-bottom: 15px; }
        .score { font-weight: bold; color: #333; font-size: 18px; }
        .rank { color: #888; font-size: 14px; margin-bottom: 5px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Glance ML: Multimodal Fashion Retrieval</h1>
        <p>Enter a natural language query to find matching fashion items with compositional understanding.</p>
    </div>
    
    <div class="search-box">
        <form method="post" action="/search" style="width: 100%; display: flex; justify-content: center;">
            <input type="text" name="query" placeholder="e.g., A red tie and a white shirt in a formal setting" value="{{ query }}" required>
            <button type="submit">Search</button>
        </form>
    </div>

    {% if results %}
    <div class="results">
        {% for res in results %}
        <div class="card">
            <div class="rank">Rank {{ loop.index }}</div>
            <!-- In a real app we would serve images, but for demo we just show the ID -->
            <div style="width: 100%; height: 200px; background: #eee; border-radius: 8px; display: flex; align-items: center; justify-content: center; margin-bottom: 15px; font-family: monospace; font-size: 12px; color: #666;">
                {{ res.image_id }}<br>(Image Preview)
            </div>
            <div class="score">Score: {{ "%.4f"|format(res.score) }}</div>
        </div>
        {% endfor %}
    </div>
    {% elif query %}
    <p style="text-align: center; color: #666;">No results found.</p>
    {% endif %}
</body>
</html>
"""

with open(templates_dir / "index.html", "w", encoding="utf-8") as f:
    f.write(html_template)

templates = Jinja2Templates(directory=str(templates_dir))

# Initialize the retriever globally
print("Initializing MultimodalRetriever...")
retriever = MultimodalRetriever()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "query": "", "results": None})

@app.post("/search", response_class=HTMLResponse)
async def search(request: Request, query: str = Form(...)):
    print(f"Searching for: {query}")
    
    # Candidate Generation (Dense Retrieval)
    clip_candidates = retriever.search_clip(query, k=50)
    
    # Attribute-Aware Metadata Reranking
    final_results = rerank_top_k(clip_candidates, query)[:10]
    
    return templates.TemplateResponse(
        "index.html", 
        {"request": request, "query": query, "results": final_results}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
