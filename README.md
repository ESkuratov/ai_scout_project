# AI Scout Project

AI Scout is an intelligent system for ingesting, indexing, and querying case studies from a PostgreSQL database using vector embeddings and RAG (Retrieval-Augmented Generation) technology. The system features an AI agent interface for natural language interaction with the data.

## Architecture Overview

The project consists of three main components:

### 1. Embedding Pipeline
Processes case data from PostgreSQL and creates vector embeddings for semantic search.

**Flow:**
- Load cases from PostgreSQL database
- Clean and deduplicate data
- Split text into chunks
- Generate embeddings for each chunk
- Store vectors in Qdrant vector database

![Embedding Pipeline](diagrams/sq_embedding_pipeline.mmd)

### 2. RAG Pipeline
Enables semantic search and AI-powered response generation.

**Flow:**
- User submits search query
- Generate query embedding
- Search Qdrant for relevant chunks
- Format context with retrieved information
- Generate response using LLM
- Return final answer to user

![RAG Pipeline](diagrams/sq_rag_pipeline.mmd)

### 3. Agent System
ReAct (Reasoning and Action) agent that can use tools to answer complex queries.

**Flow:**
- Agent receives user query
- Determines if tools are needed
- Executes tools (RAG search, database queries)
- Iterates until answer is complete
- Returns final response

![Agent Flow](diagrams/agent_flow.mmd)

## System Components

![Components Diagram](diagrams/components.mmd)

### CLI Layer
- **ingest.py** - Data ingestion entry point
- **search.py** - Search interface
- **Agent Interface** - Interactive AI agent

### Data Processing
- **Postgres Loader** - Loads raw case data
- **Case Cleaner** - Deduplication and cleaning
- **Text Splitter** - Chunks text for embedding
- **Embedding Model** - Generates vector representations

### Retrieval & Generation
- **Retriever** - Semantic search component
- **Context Formatter** - Prepares prompts
- **Response Generator** - LLM interaction

### Storage
- **PostgreSQL** - Structured case data
- **Qdrant** - Vector embeddings

## Getting Started

### Prerequisites
```bash
# Required services
- PostgreSQL (with case data)
- Qdrant vector database
- Python 3.9+
```

### Installation

```bash
# Clone repository
git clone <repository-url>
cd ai-scout-project

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your credentials
```

### Environment Variables

```env
# Database
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=cases_db
POSTGRES_USER=your_user
POSTGRES_PASSWORD=your_password

# Vector Database
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your_api_key

# LLM
OPENAI_API_KEY=your_openai_key
# or
ANTHROPIC_API_KEY=your_anthropic_key

# Langfuse (Observability)
LANGFUSE_PUBLIC_KEY=your_public_key
LANGFUSE_SECRET_KEY=your_secret_key
LANGFUSE_HOST=http://localhost:3000
```

## Usage

### 1. Data Ingestion

```bash
# Ingest cases from PostgreSQL to Qdrant
python src/cli/ingest.py
```

This will:
- Load cases from PostgreSQL
- Clean and deduplicate data
- Split into chunks
- Generate embeddings
- Store in Qdrant

### 2. Search Cases

```bash
# Semantic search
python src/cli/search.py "AI applications in healthcare"
```

### 3. Agent Mode

```bash
# Interactive agent
python src/cli/agent.py

# Or use the graph directly
python -m src.agent.graph
```

The agent can:
- Answer questions about cases
- Search with filters
- Access database directly
- Provide detailed analysis

## Project Structure

```
ai-scout-project/
├── src/
│   ├── agent/
│   │   ├── graph.py          # Agent graph definition
│   │   ├── state.py          # State management
│   │   ├── tools.py          # Agent tools
│   │   └── llm_utils.py      # LLM configuration
│   ├── embedding/
│   │   ├── postgres_loader.py
│   │   ├── case_cleaner.py
│   │   ├── text_splitter.py
│   │   ├── embedding_model.py
│   │   └── qdrant_client.py
│   ├── rag/
│   │   ├── retriever.py
│   │   ├── formatter.py
│   │   └── generator.py
│   └── cli/
│       ├── ingest.py
│       ├── search.py
│       └── agent.py
├── diagrams/
│   ├── sq_embedding_pipeline.mmd
│   ├── sq_rag_pipeline.mmd
│   ├── components.mmd
│   ├── agent_flow.mmd
│   └── schema_er.mmd
├── .env.example
├── requirements.txt
└── README.md
```

## Database Schema

The system works with the following main entities:
- **CASES** - Core case studies
- **REGIONS** - Geographical regions
- **SECTORS** - Industry sectors
- **COMPANIES** - Organizations
- **TECHNOLOGY_DRIVERS** - Technology categories
- **ECONOMIC_EFFECTS** - Financial impacts
- **SOURCES** - Data sources

See [schema_er.mmd](diagrams/schema_er.mmd) for complete schema.

## Features

### Embedding Pipeline
- ✅ Automated data loading from PostgreSQL
- ✅ Data cleaning and deduplication
- ✅ Intelligent text chunking
- ✅ Vector embedding generation
- ✅ Qdrant indexing

### RAG Pipeline
- ✅ Semantic search with filters
- ✅ Context-aware response generation
- ✅ LLM integration (OpenAI, Anthropic)
- ✅ Formatted output

### Agent System
- ✅ ReAct architecture
- ✅ Tool calling (search, database access)
- ✅ Multi-step reasoning
- ✅ Langfuse observability
- ✅ Configurable models

## Observability

The project uses Langfuse for LLM observability:
- Track all LLM calls
- Monitor token usage
- Debug agent behavior
- Analyze performance

Access dashboard at: `http://localhost:3000`

## Development

### Adding New Tools

```python
# src/agent/tools.py
from langchain_core.tools import tool

@tool
def my_new_tool(query: str) -> str:
    """Tool description for the agent."""
    # Implementation
    return result

TOOLS = [..., my_new_tool]
```

### Customizing Agent Behavior

Edit the system prompt in `src/agent/context.py`:

```python
system_prompt = """
You are an AI assistant specialized in...
Current time: {system_time}
"""
```

## Troubleshooting

### Qdrant Connection Issues
```bash
# Check Qdrant is running
curl http://localhost:6333/collections

# Restart Qdrant
docker restart qdrant
```

### PostgreSQL Connection
```bash
# Test connection
psql -h localhost -U your_user -d cases_db
```

### LLM API Issues
- Verify API keys in `.env`
- Check rate limits
- Review Langfuse traces

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## License

[Your License Here]

## Contact

[Your Contact Information]
