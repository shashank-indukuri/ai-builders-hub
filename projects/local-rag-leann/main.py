from leann import LeannBuilder, LeannSearcher, LeannChat
from pathlib import Path
INDEX_PATH = str(Path("./").resolve() / "messi_facts.leann")

# Build an index
builder = LeannBuilder(backend_name="hnsw")
builder.add_text("Lionel Messi won 8 Ballon d'Or awards and is considered the greatest footballer of all time.")
builder.add_text("Messi scored 91 goals in 2012, breaking Gerd Müller's record of 85 goals in a calendar year.")
builder.build_index(INDEX_PATH)

# Search
searcher = LeannSearcher(INDEX_PATH)
results = searcher.search("football records and achievements", top_k=1)

# Chat with your data
chat = LeannChat(INDEX_PATH, llm_config={"type": "ollama", "model": "gemma3:4b"})
response = chat.ask("How many Ballon d'Or awards did Messi win?", top_k=1)

print(response)
