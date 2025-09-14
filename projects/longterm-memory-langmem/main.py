from langgraph.prebuilt import create_react_agent
from langgraph.store.memory import InMemoryStore
from langmem import create_manage_memory_tool, create_search_memory_tool
from dotenv import load_dotenv
load_dotenv()

# Set up storage 
store = InMemoryStore(
    index={
        "dims": 1536,
        "embed": "ollama:nomic-embed-text",
    }
) 

# Create an agent with memory capabilities 
agent = create_react_agent(
    "ollama:qwen3:8b",
    tools=[
        # Memory tools use LangGraph's BaseStore for persistence
        create_manage_memory_tool(namespace=("memories",)),
        create_search_memory_tool(namespace=("memories",)),
    ],
    store=store,
)

# Store a new memory 
agent.invoke(
    {"messages": [{"role": "user", "content": "Remember that my favorite programming language is Python."}]}
)

# Retrieve the stored memory 
response = agent.invoke(
    {"messages": [{"role": "user", "content": "What do you know about my professional background?"}]}
)
print(response["messages"][-1].content)
# Output: "You've told me that your favorite programming language is Python."