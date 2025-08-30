import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document
from pyvis.network import Network
import networkx as nx
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import time, io

# --- 1. Set up the environment and models ---
print("--- Step 1: Setting up the environment and initializing models ---")
load_dotenv()
try:
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("GOOGLE_API_KEY not found in environment variables.")
    # Using gemini-1.5-pro for better graph extraction
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    print("✓ Models initialized successfully.")
except ValueError as e:
    print(f"Error: {e}")
    exit()

# --- 2. Ingest and transform documents ---
print("\n--- Step 2: Ingesting documents and transforming into a knowledge graph ---")
documents = [
    Document(page_content="Lily is a young girl who loves animals. She has a fluffy cat named Leo."),
    Document(page_content="Leo the cat is very playful. He enjoys chasing red laser dots and napping in the sun."),
    Document(page_content="Lily also has a friendly dog named Max. Max is a golden retriever and is very loyal."),
    Document(page_content="Max the dog and Leo the cat are best friends. They often play together in the garden."),
    Document(page_content="Lily takes Max for a walk every day in the park. Sometimes, Leo watches them from the window.")
]
print("✓ Documents ingested.")
print("\n[Raw Documents]:")
for i, doc in enumerate(documents):
    print(f"  Document {i+1}: '{doc.page_content}'")

print("\n> Extracting entities and relationships from documents using LLM...")
llm_transformer = LLMGraphTransformer(llm=llm)
graph_documents = llm_transformer.convert_to_graph_documents(documents)
print(f"✓ Knowledge graph extracted with {len(graph_documents)} relationships.")

print("\n[Extracted Entities and Relationships]:")
for i, gd in enumerate(graph_documents):
    print(f"  - From Document {i+1}:")
    for rel in gd.relationships:
        source_id = rel.source.id
        target_id = rel.target.id
        rel_type = rel.type
        print(f"    - Relationship: {source_id} --({rel_type})--> {target_id}")

# --- 3. Visualize the full knowledge graph ---
def visualize_graph(graph_documents, filename="knowledge_graph.html"):
    """Creates and displays an interactive knowledge graph."""
    print("\n--- Generating full knowledge graph visualization ---")
    net = Network(notebook=True, height="750px", width="100%", bgcolor="#222222", font_color="white", cdn_resources='in_line')
    
    nx_graph = nx.DiGraph()
    for gd in graph_documents:
        for relationship in gd.relationships:
            source_node = relationship.source
            target_node = relationship.target
            source_title = f"ID: {source_node.id}\nType: {source_node.type}\nAttributes: {source_node.lc_attributes}"
            target_title = f"ID: {target_node.id}\nType: {target_node.type}\nAttributes: {target_node.lc_attributes}"
            nx_graph.add_node(source_node.id, title=source_title)
            nx_graph.add_node(target_node.id, title=target_title)
            nx_graph.add_edge(source_node.id, target_node.id, title=relationship.type, label=relationship.type)
    
    net.from_nx(nx_graph)
    html_content = net.generate_html()
    with io.open(filename, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"✓ Full knowledge graph saved as {filename}")

# --- 4. Build the RAG system ---
print("\n--- Step 4: Building the Retrieval-Augmented Generation (RAG) system ---")
vector_store = InMemoryVectorStore.from_documents(documents, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 2})
print("✓ In-memory vector store created and RAG retriever configured.")

template = """
You are a helpful assistant that answers questions based on the provided context.
If the context doesn't contain the answer, say you don't know.

Context: {context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
print("✓ RAG chain built.")

def visualize_query_subgraph(retrieved_docs, all_graph_docs, filename="query_subgraph.html"):
    """Visualizes the subgraph based on the retrieved documents."""
    print(f"\n--- Generating query-specific subgraph visualization ({filename}) ---")
    retrieved_content = [doc.page_content for doc in retrieved_docs]
    
    query_relationships = []
    for gd in all_graph_docs:
        for relationship in gd.relationships:
            source_id = relationship.source.id
            target_id = relationship.target.id
            if any(source_id in content or target_id in content for content in retrieved_content):
                query_relationships.append(relationship)

    net = Network(notebook=True, height="750px", width="100%", bgcolor="#222222", font_color="white", cdn_resources='in_line')
    
    nodes_to_add = {}
    for relationship in query_relationships:
        nodes_to_add[relationship.source.id] = relationship.source
        nodes_to_add[relationship.target.id] = relationship.target
    
    for node_id, node in nodes_to_add.items():
        node_title = f"ID: {node_id}\nType: {node.type}\nAttributes: {node.lc_attributes}"
        net.add_node(node_id, title=node_title)
        
    for relationship in query_relationships:
        net.add_edge(relationship.source.id, relationship.target.id, title=relationship.type, label=relationship.type)
    
    html_content = net.generate_html()
    with io.open(filename, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"✓ Query-specific subgraph saved as {filename}")

# --- Main Loop for User Queries ---
print("\n--- Program is ready. Type 'exit' to quit. ---")
visualize_graph(graph_documents)

while True:
    try:
        user_query = input("\nEnter your query: ")
        if user_query.lower() == 'exit':
            print("Exiting program.")
            break
        
        print("\n--- Step 5: Executing RAG chain for your query ---")
        start_time = time.time()
        
        print(f"> Retrieving relevant documents for query: '{user_query}'...")
        retrieved_docs_for_viz = retriever.invoke(user_query)
        print("✓ Documents retrieved.")
        print("[Retrieved Document Chunks]:")
        for i, doc in enumerate(retrieved_docs_for_viz):
            print(f"  - Chunk {i+1}: '{doc.page_content}'")
            
        print("\n> Invoking LLM with retrieved context...")
        response = rag_chain.invoke(user_query)
        
        end_time = time.time()
        print(f"✓ LLM response received.")
        print(f"\n[Final RAG Response] ({round(end_time - start_time, 2)}s):\n{response}\n")
        
        print("\n--- Step 6: Visualizing the relevant subgraph for the query ---")
        visualize_query_subgraph(retrieved_docs_for_viz, graph_documents, filename="query_subgraph.html")
    
    except Exception as e:
        print(f"An error occurred during query processing: {e}")