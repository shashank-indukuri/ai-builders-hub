import logging
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from mem0 import Memory
import google.generativeai as genai


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load environment variables
load_dotenv()



POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "postgres")
POSTGRES_PORT = os.environ.get("POSTGRES_PORT", "5432")
POSTGRES_DB = os.environ.get("POSTGRES_DB", "postgres")
POSTGRES_USER = os.environ.get("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "postgres")
POSTGRES_COLLECTION_NAME = os.environ.get("POSTGRES_COLLECTION_NAME", "memories")

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://neo4j:7687")
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "mem0graph")

MEMGRAPH_URI = os.environ.get("MEMGRAPH_URI", "bolt://localhost:7687")
MEMGRAPH_USERNAME = os.environ.get("MEMGRAPH_USERNAME", "memgraph")
MEMGRAPH_PASSWORD = os.environ.get("MEMGRAPH_PASSWORD", "mem0graph")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
HISTORY_DB_PATH = os.environ.get("HISTORY_DB_PATH", "/app/history/history.db")

# Configure FastAPI app
app = FastAPI(title="Mem0 API", description="Memory management API for AI agents")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# No static mount needed; UI is served inline at /ui



DEFAULT_CONFIG = {
    "version": "v1.1",
    # "vector_store": {
    #     "provider": "pgvector",
    #     "config": {
    #         "host": POSTGRES_HOST,
    #         "port": int(POSTGRES_PORT),
    #         "dbname": POSTGRES_DB,
    #         "user": POSTGRES_USER,
    #         "password": POSTGRES_PASSWORD,
    #         "collection_name": POSTGRES_COLLECTION_NAME,
    #         "embedding_model_dims": 1024
    #     },
    # },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "url": os.environ.get("QDRANT_URL", "http://qdrant:6333"),
            # "api_key": os.environ.get("QDRANT_API_KEY"),  # if secured
            "collection_name": os.environ.get("QDRANT_COLLECTION", "memories"),
            "embedding_model_dims": 1024  # must match embedder dims
        },
    },
    "graph_store": {
        "provider": "neo4j",
        "config": {"url": NEO4J_URI, "username": NEO4J_USERNAME, "password": NEO4J_PASSWORD},
    },
    # "llm": {"provider": "openai", "config": {"api_key": OPENAI_API_KEY, "temperature": 0.2, "model": "gpt-4o"}},
    # "embedder": {"provider": "openai", "config": {"api_key": OPENAI_API_KEY, "model": "text-embedding-3-small"}},
    "llm": {
        "provider": "gemini",
        "config": {
            "api_key": os.environ.get("GOOGLE_API_KEY"),
            "model": "models/gemini-2.0-flash",
            "temperature": 0.1,
            "max_tokens": 6000
        }
    },
    "embedder": {
        "provider": "gemini",
        "config": {
            "api_key": os.environ.get("GOOGLE_API_KEY"),
            "embedding_dims": 1024,
            # "embedding_dims": 1536,
            "model": "gemini-embedding-001"
            # "model": "gemini-embedding-exp-03-07"
            # "model": "models/text-embedding-004"
            
        }
    },
    "history_db_path": HISTORY_DB_PATH,
}


MEMORY_INSTANCE = Memory.from_config(DEFAULT_CONFIG)

# Configure Gemini once
try:
    if os.environ.get("GOOGLE_API_KEY"):
        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
except Exception:
    # Safe to ignore configuration errors here; endpoint will report if used without key
    pass


class Message(BaseModel):
    role: str = Field(..., description="Role of the message (user or assistant).")
    content: str = Field(..., description="Message content.")


class MemoryCreate(BaseModel):
    messages: List[Message] = Field(..., description="List of messages to store.")
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    run_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query.")
    user_id: Optional[str] = None
    run_id: Optional[str] = None
    agent_id: Optional[str] = None
    filters: Optional[Dict[str, Any]] = None


class ChatRequest(BaseModel):
    message: str = Field(..., description="User message to chat with.")
    user_id: str = Field(..., description="User identifier for retrieving/storing memories.")
    limit: int = Field(3, description="Number of memories to retrieve.")


@app.post("/configure", summary="Configure Mem0")
def set_config(config: Dict[str, Any]):
    """Set memory configuration."""
    global MEMORY_INSTANCE
    MEMORY_INSTANCE = Memory.from_config(config)
    return {"message": "Configuration set successfully"}


@app.post("/memories", summary="Create memories")
def add_memory(memory_create: MemoryCreate):
    """Store new memories."""
    if not any([memory_create.user_id, memory_create.agent_id, memory_create.run_id]):
        raise HTTPException(status_code=400, detail="At least one identifier (user_id, agent_id, run_id) is required.")

    params = {k: v for k, v in memory_create.model_dump().items() if v is not None and k != "messages"}
    try:
        response = MEMORY_INSTANCE.add(messages=[m.model_dump() for m in memory_create.messages], **params)
        return JSONResponse(content=response)
    except Exception as e:
        logging.exception("Error in add_memory:")  # This will log the full traceback
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", summary="Chat with memories")
def chat_with_memories(req: ChatRequest):
    """Generate an assistant reply using Gemini, grounded by retrieved memories.

    Steps:
    - Retrieve relevant memories via MEMORY_INSTANCE.search
    - Build a system prompt including memories (if any)
    - Call Gemini to generate the assistant reply
    - Store the user and assistant messages via MEMORY_INSTANCE.add
    - Return the assistant reply and the retrieved memories
    """
    try:
        # Validate API key early
        if not os.environ.get("GOOGLE_API_KEY"):
            raise HTTPException(status_code=400, detail="GOOGLE_API_KEY is not set.")
        # Retrieve memories
        mem_results = MEMORY_INSTANCE.search(query=req.message, user_id=req.user_id, limit=req.limit)
        # Normalize memories list
        if isinstance(mem_results, dict) and "results" in mem_results:
            memories = mem_results.get("results", []) or []
        else:
            memories = mem_results or []

        # Build context string from text memories
        try:
            memories_str = "\n".join(
                f"- {m.get('memory') or m.get('text') or str(m)}" for m in memories
            ) if memories else ""
        except Exception:
            memories_str = ""

        if memories_str:
            system_prompt = (
                "You are a helpful AI assistant. Use the user's stored memories to answer clearly.\n"
                "User Memories:\n" + memories_str
            )
        else:
            system_prompt = "You are a helpful AI assistant. Answer the user's question clearly."

        # Generate reply with Gemini
        model_name = DEFAULT_CONFIG.get("llm", {}).get("config", {}).get("model", "models/gemini-2.5-flash")
        model = genai.GenerativeModel(model_name)
        prompt = f"{system_prompt}\n\nUser: {req.message}"
        gen_resp = model.generate_content(prompt)
        assistant_text = getattr(gen_resp, "text", None) or ""
        if not assistant_text:
            # try to glean from candidates
            try:
                candidate = (gen_resp.candidates or [])[0]
                assistant_text = getattr(candidate, "content", {}).get("parts", [{}])[0].get("text", "")
            except Exception:
                pass
        if not assistant_text:
            assistant_text = "(No response generated.)"

        # Store conversation in memory
        conversation = [
            {"role": "user", "content": req.message},
            {"role": "assistant", "content": assistant_text},
        ]
        try:
            MEMORY_INSTANCE.add(messages=conversation, user_id=req.user_id)
        except Exception as e:
            logging.warning("Failed to store conversation: %s", e)

        return {
            "assistant": assistant_text,
            "memories_used": memories,
        }
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("Error in chat_with_memories:")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memories", summary="Get memories")
def get_all_memories(
    user_id: Optional[str] = None,
    run_id: Optional[str] = None,
    agent_id: Optional[str] = None,
):
    """Retrieve stored memories."""
    if not any([user_id, run_id, agent_id]):
        raise HTTPException(status_code=400, detail="At least one identifier is required.")
    try:
        params = {
            k: v for k, v in {"user_id": user_id, "run_id": run_id, "agent_id": agent_id}.items() if v is not None
        }
        return MEMORY_INSTANCE.get_all(**params)
    except Exception as e:
        logging.exception("Error in get_all_memories:")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memories/{memory_id}", summary="Get a memory")
def get_memory(memory_id: str):
    """Retrieve a specific memory by ID."""
    try:
        return MEMORY_INSTANCE.get(memory_id)
    except Exception as e:
        logging.exception("Error in get_memory:")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", summary="Search memories")
def search_memories(search_req: SearchRequest):
    """Search for memories based on a query."""
    try:
        params = {k: v for k, v in search_req.model_dump().items() if v is not None and k != "query"}
        return MEMORY_INSTANCE.search(query=search_req.query, **params)
    except Exception as e:
        logging.exception("Error in search_memories:")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/memories/{memory_id}", summary="Update a memory")
def update_memory(memory_id: str, updated_memory: Dict[str, Any]):
    """Update an existing memory with new content.
    
    Args:
        memory_id (str): ID of the memory to update
        updated_memory (str): New content to update the memory with
        
    Returns:
        dict: Success message indicating the memory was updated
    """
    try:
        return MEMORY_INSTANCE.update(memory_id=memory_id, data=updated_memory)
    except Exception as e:
        logging.exception("Error in update_memory:")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memories/{memory_id}/history", summary="Get memory history")
def memory_history(memory_id: str):
    """Retrieve memory history."""
    try:
        return MEMORY_INSTANCE.history(memory_id=memory_id)
    except Exception as e:
        logging.exception("Error in memory_history:")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/memories/{memory_id}", summary="Delete a memory")
def delete_memory(memory_id: str):
    """Delete a specific memory by ID."""
    try:
        MEMORY_INSTANCE.delete(memory_id=memory_id)
        return {"message": "Memory deleted successfully"}
    except Exception as e:
        logging.exception("Error in delete_memory:")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/memories", summary="Delete all memories")
def delete_all_memories(
    user_id: Optional[str] = None,
    run_id: Optional[str] = None,
    agent_id: Optional[str] = None,
):
    """Delete all memories for a given identifier."""
    if not any([user_id, run_id, agent_id]):
        raise HTTPException(status_code=400, detail="At least one identifier is required.")
    try:
        params = {
            k: v for k, v in {"user_id": user_id, "run_id": run_id, "agent_id": agent_id}.items() if v is not None
        }
        MEMORY_INSTANCE.delete_all(**params)
        return {"message": "All relevant memories deleted"}
    except Exception as e:
        logging.exception("Error in delete_all_memories:")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reset", summary="Reset all memories")
def reset_memory():
    """Completely reset stored memories."""
    try:
        MEMORY_INSTANCE.reset()
        return {"message": "All memories reset"}
    except Exception as e:
        logging.exception("Error in reset_memory:")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", summary="Redirect to the OpenAPI documentation", include_in_schema=False)
def home():
    """Redirect to the OpenAPI documentation."""
    return RedirectResponse(url="/docs")


@app.get("/ui", response_class=HTMLResponse, summary="Simple UI for Mem0")
def ui():
    neo4j_http = os.environ.get("NEO4J_HTTP", "http://localhost:8474")
    return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Mem0 Demo UI</title>
  <style>
    body { font-family: system-ui, Arial, sans-serif; margin: 20px; }
    .row { display: flex; gap: 16px; }
    .col { flex: 1; min-width: 280px; border: 1px solid #ddd; padding: 12px; border-radius: 8px; }
    textarea, input { width: 100%; box-sizing: border-box; padding: 8px; margin: 6px 0; }
    button { padding: 8px 12px; margin-right: 8px; }
    pre { background: #f7f7f7; padding: 8px; border-radius: 6px; overflow: auto; }
    .badge { background:#eef; padding:2px 6px; border-radius:4px; }
  </style>
  <script>
    async function addMemory(){
      const user_id = document.getElementById('user_id').value.trim();
      const agent_id = document.getElementById('agent_id').value.trim() || null;
      const run_id = document.getElementById('run_id').value.trim() || null;
      const userMsg = document.getElementById('user_msg').value.trim();
      const assistantMsg = document.getElementById('assistant_msg').value.trim();
      if(!user_id || !userMsg){ alert('user_id and user message are required'); return; }
      const payload = {
        messages: [
          { role: 'user', content: userMsg },
          ...(assistantMsg ? [{ role: 'assistant', content: assistantMsg }] : [])
        ],
        user_id, agent_id, run_id
      };
      try {
        const res = await fetch('/memories', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload)});
        const text = await res.text();
        if (!res.ok) throw new Error(res.status + ' ' + res.statusText + ': ' + text);
        document.getElementById('add_result').textContent = JSON.stringify(JSON.parse(text), null, 2);
      } catch (err) {
        document.getElementById('add_result').textContent = 'Error: ' + err.message;
        console.error('Add Memory failed:', err);
      }
    }

    async function chat(){
      const user_id = document.getElementById('user_id').value.trim();
      const userMsg = document.getElementById('user_msg').value.trim();
      if(!user_id || !userMsg){ alert('user_id and user message are required'); return; }
      const payload = { message: userMsg, user_id, limit: 3 };
      try {
        const res = await fetch('/chat', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload)});
        const text = await res.text();
        if (!res.ok) throw new Error(res.status + ' ' + res.statusText + ': ' + text);
        document.getElementById('chat_result').textContent = JSON.stringify(JSON.parse(text), null, 2);
      } catch (err) {
        document.getElementById('chat_result').textContent = 'Error: ' + err.message;
        console.error('Chat failed:', err);
      }
    }

    async function search(){
      const user_id = document.getElementById('user_id').value.trim() || null;
      const agent_id = document.getElementById('agent_id').value.trim() || null;
      const run_id = document.getElementById('run_id').value.trim() || null;
      const query = document.getElementById('query').value.trim();
      if(!query){ alert('query is required'); return; }
      const payload = { query, user_id, agent_id, run_id };
      try {
        const res = await fetch('/search', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload)});
        const text = await res.text();
        if (!res.ok) throw new Error(res.status + ' ' + res.statusText + ': ' + text);
        document.getElementById('search_result').textContent = JSON.stringify(JSON.parse(text), null, 2);
      } catch (err) {
        document.getElementById('search_result').textContent = 'Error: ' + err.message;
        console.error('Search failed:', err);
      }
    }

    async function listMemories(){
      const params = new URLSearchParams();
      const user_id = document.getElementById('user_id').value.trim();
      const agent_id = document.getElementById('agent_id').value.trim();
      const run_id = document.getElementById('run_id').value.trim();
      if(user_id) params.set('user_id', user_id);
      if(agent_id) params.set('agent_id', agent_id);
      if(run_id) params.set('run_id', run_id);
      if([...params.keys()].length === 0){ alert('Provide at least one identifier to list memories'); return; }
      try {
        const res = await fetch('/memories?' + params.toString());
        const text = await res.text();
        if (!res.ok) throw new Error(res.status + ' ' + res.statusText + ': ' + text);
        document.getElementById('list_result').textContent = JSON.stringify(JSON.parse(text), null, 2);
      } catch (err) {
        document.getElementById('list_result').textContent = 'Error: ' + err.message;
        console.error('List Memories failed:', err);
      }
    }
  </script>
</head>
<body>
  <h1>Mem0 Demo <span class="badge">FastAPI</span></h1>
  <p>This UI calls your existing REST endpoints to show how Mem0 adds, searches, and lists memories. Graph data is stored in Neo4j (<code>""" + NEO4J_URI + """</code>). Use the Neo4j Browser to view the graph: <a href=\"""" + neo4j_http + "\" target=\"_blank\">""" + neo4j_http + """</a>.</p>

  <div class="row">
    <div class="col">
      <h3>Identifiers</h3>
      <input id="user_id" placeholder="user_id (required for add/list)" />
      <input id="agent_id" placeholder="agent_id (optional)" />
      <input id="run_id" placeholder="run_id (optional)" />
    </div>
    <div class="col">
      <h3>Chat (store)</h3>
      <textarea id="user_msg" rows="3" placeholder="User message..."></textarea>
      <textarea id="assistant_msg" rows="3" placeholder="Assistant message (optional)..."></textarea>
      <div>
        <button onclick="addMemory()">Add Memory</button>
        <button onclick="chat()">Chat</button>
      </div>
      <h4>Result</h4>
      <pre id="add_result"></pre>
      <h4>Chat Reply</h4>
      <pre id="chat_result"></pre>
    </div>
    <div class="col">
      <h3>Search</h3>
      <input id="query" placeholder="Search query..." />
      <div>
        <button onclick="search()">Search</button>
      </div>
      <h4>Results</h4>
      <pre id="search_result"></pre>
    </div>
  </div>
  <div class="row" style="margin-top:16px;">
    <div class="col">
      <h3>List Memories</h3>
      <p>Provide at least one identifier above, then list stored memories.</p>
      <button onclick="listMemories()">List Memories</button>
      <h4>Results</h4>
      <pre id="list_result"></pre>
    </div>
    <div class="col">
      <h3>Graph</h3>
      <p>Neo4j URI: <code>""" + NEO4J_URI + """</code></p>
      <p>Open Neo4j Browser: <a href=\""" + neo4j_http + "\" target=\"_blank\">""" + neo4j_http + """</a></p>
      <p>Sample Cypher:</p>
      <pre>MATCH (n)-[r]->(m) RETURN n,r,m LIMIT 50;</pre>
    </div>
  </div>
</body>
</html>
"""