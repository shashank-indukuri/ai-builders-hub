# LangChain Graph RAG Architecture

This document details the architectural components and data flow of the LangChain-based Graph RAG implementation.

## System Architecture

```mermaid
graph TD
    subgraph Input
        A[Raw Documents]
        B[User Queries]
    end
    
    subgraph Processing
        C[LangChain Transformer]
        D[Vector Store]
        E[Knowledge Graph]
    end
    
    subgraph Output
        F[Interactive Visualizations]
        G[LLM Responses]
    end
    
    A --> C
    B --> D
    C --> D
    C --> E
    D --> G
    E --> F
    
    style E fill:#bbf,stroke:#333,stroke-width:2px
    style G fill:#bfb,stroke:#333,stroke-width:2px
```

## Component Details

### 1. Document Processing Layer
```mermaid
graph LR
    A[Documents] --> B[LangChain Documents]
    B --> C[Entity Extraction]
    C --> D[Relationship Detection]
    D --> E[Graph Documents]
    
    style E fill:#bbf,stroke:#333,stroke-width:2px
```

#### Key Components:
- **Document Class**: LangChain's Document structure
- **LLMGraphTransformer**: Handles entity and relationship extraction
- **GoogleGenerativeAI**: Powers the extraction process

### 2. Knowledge Graph Construction
```mermaid
graph TD
    A[Graph Documents] --> B[NetworkX Graph]
    B --> C[Node Creation]
    B --> D[Edge Creation]
    C --> E[Pyvis Network]
    D --> E
    E --> F[Interactive HTML]
    
    style B fill:#bbf,stroke:#333,stroke-width:2px
    style F fill:#bfb,stroke:#333,stroke-width:2px
```

#### Visualization System:
- Uses Pyvis for interactive graphs
- Supports node/edge attributes
- Provides hover information
- Enables dynamic interaction

### 3. RAG Implementation
```mermaid
graph LR
    A[Documents] --> B[Vector Embeddings]
    B --> C[InMemoryVectorStore]
    D[User Query] --> E[Retriever]
    C --> E
    E --> F[Context]
    F --> G[LLM Response]
    
    style C fill:#bbf,stroke:#333,stroke-width:2px
    style G fill:#bfb,stroke:#333,stroke-width:2px
```

#### Components:
- **Embeddings**: GoogleGenerativeAIEmbeddings
- **Vector Store**: InMemoryVectorStore
- **Retriever**: Contextual document retrieval
- **RAG Chain**: Orchestrates the query process

### 4. Query Processing Flow
```mermaid
sequenceDiagram
    participant U as User
    participant R as Retriever
    participant G as Graph
    participant L as LLM
    participant V as Visualizer
    
    U->>R: Submit Query
    R->>G: Get Relevant Documents
    G->>V: Generate Subgraph
    R->>L: Provide Context
    L->>U: Return Response
    V->>U: Show Visualization
```

## Implementation Details

### 1. Model Configuration
```python
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
```

### 2. Graph Structure
```mermaid
classDiagram
    class Node {
        +String id
        +String type
        +Dict attributes
    }
    
    class Edge {
        +Node source
        +Node target
        +String type
    }
    
    class GraphDocument {
        +List[Node] nodes
        +List[Edge] relationships
    }
    
    GraphDocument "1" --> "*" Node
    GraphDocument "1" --> "*" Edge
```

### 3. Visualization Architecture
```mermaid
graph TD
    A[NetworkX DiGraph] --> B[Pyvis Network]
    B --> C[HTML Generation]
    C --> D[Interactive Display]
    
    style B fill:#bbf,stroke:#333,stroke-width:2px
```

## Data Flow

### Document Processing
1. Raw document ingestion
2. Conversion to LangChain format
3. Entity and relationship extraction
4. Knowledge graph construction
5. Vector store population

### Query Processing
1. Query reception
2. Document retrieval
3. Subgraph generation
4. Context assembly
5. LLM response generation

## Performance Considerations

### Memory Management
- InMemoryVectorStore for small to medium datasets
- Efficient graph traversal with NetworkX
- Optimized visualization rendering

### Response Time
- Batch processing for document ingestion
- Efficient retrieval mechanisms
- Parallel processing where possible

### Scalability
- Modular design for easy extension
- Support for different LLM models
- Flexible visualization options

## Security Considerations

1. **API Key Management**
   - Environment variable usage
   - Secure key storage
   - Error handling for missing keys

2. **Data Processing**
   - Input validation
   - Error handling
   - Safe visualization generation

## Future Enhancements

1. **Scalability**
   - Persistent vector storage
   - Distributed processing
   - Batch query handling

2. **Visualization**
   - Custom styling options
   - Advanced filtering
   - Real-time updates

3. **Integration**
   - Additional LLM providers
   - Different embedding models
   - Alternative graph databases
