# AI Task Manager with FastAPI-MCP 🤖

Manage AI tasks through FastAPI endpoints exposed as Model Context Protocol (MCP) tools - create, update, and track tasks that AI assistants can interact with directly.

## Why This Matters

- 🚀 **FastAPI Integration**: Powered by FastAPI-MCP for seamless endpoint exposure
- 🤖 **AI Assistant Ready**: Direct integration with Claude Desktop, Cursor, and other MCP clients
- 🔒 **Authentication Built-in**: Uses FastAPI's native dependency system for secure access
- ⚡ **ASGI Transport**: Efficient communication using FastAPI's ASGI interface
- 💰 **Zero Configuration**: Minimal setup required - just point and it works

## Key Features

- **Complete Task Management**: Create, read, update, delete AI tasks with full CRUD operations
- **Speaker Detection**: Track task status (pending, in_progress, completed) and priority levels
- **MCP Tool Exposure**: All endpoints automatically available as MCP tools
- **Real-time Processing**: Instant task updates with proper validation
- **Clean API Design**: RESTful endpoints with comprehensive documentation

## Installation and Setup

### Prerequisites
- Python 3.12 or later
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- FastAPI-MCP package

### 1. Install Dependencies

**Using uv (recommended):**
```bash
uv sync
```

**Using pip:**
```bash
pip install fastapi fastapi-mcp pydantic
```

### 2. Run the App

```bash
uv run uvicorn main:app --reload
# or with pip: uvicorn main:app --reload
```

The API will be available at `http://localhost:8000` and MCP endpoint at `http://localhost:8000/mcp`.

## Usage

### Standalone API
1. Start the FastAPI server
2. Visit `http://localhost:8000/docs` for interactive API documentation
3. Use the REST endpoints to manage tasks programmatically

### MCP Integration
1. Configure your MCP client (Claude Desktop, Cursor, etc.)
2. Add the server configuration to connect to `http://localhost:8000/mcp`
3. AI assistants can now use the following tools:
   - `create_task` - Create new AI tasks
   - `get_tasks` - List all tasks
   - `get_task` - Get specific task by ID
   - `update_task` - Update task details
   - `delete_task` - Remove tasks
   - `get_tasks_by_status` - Filter tasks by status

## MCP Client Configuration
### Cursor
Add to your MCP configuration:
```json
{
  "mcpServers": {
    "mcp-with-fastapi": {
        "url": "http://localhost:8000/mcp"
    }
  }
}
```

### Other MCP Clients
This server is compatible with any MCP client that supports HTTP transport. Simply point your client to `http://localhost:8000/mcp` and the tools will be automatically available.

## Available MCP Tools

- **create_task**: Create new AI tasks with title, description, and priority
- **get_tasks**: Retrieve all tasks in the system
- **get_task**: Get details of a specific task by ID
- **update_task**: Modify existing task properties
- **delete_task**: Remove tasks from the system
- **get_tasks_by_status**: Filter tasks by status (pending, in_progress, completed)

---

## Contribution

Contributions are welcome! Please fork the repository and submit a pull request with your improvements.