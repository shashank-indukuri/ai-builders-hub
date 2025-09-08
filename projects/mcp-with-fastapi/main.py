import uvicorn
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi_mcp import FastApiMCP

# Pydantic models
class Task(BaseModel):
    id: Optional[int] = None
    title: str
    description: str
    status: str = "pending"
    priority: str = "medium"
    created_at: Optional[str] = None

class TaskUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    status: Optional[str] = None
    priority: Optional[str] = None

# In-memory storage
tasks_db = []
task_counter = 1

# FastAPI app
app = FastAPI(
    title="AI Task Manager",
    description="Manage AI tasks with MCP integration",
    version="1.0.0"
)

@app.get("/")
async def root():
    return {"message": "AI Task Manager with MCP", "mcp_endpoint": "/mcp"}

@app.post("/tasks", response_model=Task)
async def create_task(task: Task):
    """Create a new AI task"""
    global task_counter
    task.id = task_counter
    task.created_at = datetime.now().isoformat()
    tasks_db.append(task.dict())
    task_counter += 1
    return task

@app.get("/tasks", response_model=List[Task])
async def get_tasks():
    """Get all AI tasks"""
    return tasks_db

@app.get("/tasks/{task_id}", response_model=Task)
async def get_task(task_id: int):
    """Get a specific task by ID"""
    task = next((t for t in tasks_db if t["id"] == task_id), None)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task

@app.put("/tasks/{task_id}", response_model=Task)
async def update_task(task_id: int, task_update: TaskUpdate):
    """Update an existing task"""
    task = next((t for t in tasks_db if t["id"] == task_id), None)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    for field, value in task_update.dict(exclude_unset=True).items():
        task[field] = value
    
    return task

@app.delete("/tasks/{task_id}")
async def delete_task(task_id: int):
    """Delete a task"""
    global tasks_db
    tasks_db = [t for t in tasks_db if t["id"] != task_id]
    return {"message": "Task deleted successfully"}

mcp = FastApiMCP(app)
mcp.mount()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)