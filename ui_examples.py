#!/usr/bin/env python3
"""
Example usage of the new UI block structure for todos and proposed tasks
"""

from datetime import datetime, timedelta
from noki_ai_engine.app.models.ui_blocks import BlockFactory
from noki_ai_engine.app.services.llm import LLMService

def example_explanation_block():
    """Example of explanation block with header and content"""
    return BlockFactory.create_explanation_block(
        text="Here is an explanation of your project",
        explanation_content="This project involves creating a web application using React and Node.js. The main components include user authentication, data management, and real-time updates. You'll need to focus on the frontend components first, then implement the backend API endpoints."
    )

def example_grouped_todos():
    """Example of todos grouped by date"""
    todos = [
        {
            "title": "Complete React component setup",
            "description": "Set up the main React components for the project",
            "project_id": "proj_123",
            "task_id": "task_456",
            "due_datetime": datetime.now() + timedelta(days=1),
            "source": "ai_created"
        },
        {
            "title": "Review project requirements",
            "description": "Go through the project specification document",
            "project_id": "proj_123",
            "task_id": "task_789",
            "due_datetime": datetime.now() + timedelta(days=1),
            "source": "input_received"
        },
        {
            "title": "Set up database schema",
            "description": "Create the database tables and relationships",
            "project_id": "proj_123",
            "task_id": "task_101",
            "due_datetime": datetime.now() + timedelta(days=2),
            "source": "ai_created"
        },
        {
            "title": "Write API documentation",
            "description": "Document all API endpoints",
            "project_id": "proj_123",
            "task_id": "task_102",
            "due_datetime": datetime.now() + timedelta(days=2),
            "source": "input_received"
        }
    ]
    
    return BlockFactory.create_todo_list(
        list_title="Your Project Todos",
        items=todos,
        grouped_by_date=True,
        footer="Todos grouped by due date"
    )

def example_grouped_proposed_tasks():
    """Example of proposed tasks grouped by date"""
    tasks = [
        {
            "title": "Implement user authentication",
            "description": "Add login and registration functionality",
            "due_datetime": datetime.now() + timedelta(days=3),
            "priority": "high",
            "estimated_duration": "4 hours"
        },
        {
            "title": "Create dashboard layout",
            "description": "Design and implement the main dashboard",
            "due_datetime": datetime.now() + timedelta(days=3),
            "priority": "medium",
            "estimated_duration": "3 hours"
        },
        {
            "title": "Add data visualization",
            "description": "Implement charts and graphs for data display",
            "due_datetime": datetime.now() + timedelta(days=4),
            "priority": "medium",
            "estimated_duration": "2 hours"
        },
        {
            "title": "Write unit tests",
            "description": "Create comprehensive test coverage",
            "due_datetime": datetime.now() + timedelta(days=5),
            "priority": "low",
            "estimated_duration": "6 hours"
        }
    ]
    
    return BlockFactory.create_proposed_task_list(
        list_title="Proposed Development Tasks",
        items=tasks,
        grouped_by_date=True,
        footer="Tasks grouped by suggested due date"
    )

def example_llm_service_usage():
    """Example of using LLM service helper methods"""
    # This would be used within the LLM service
    llm_service = LLMService(None)  # VectorService would be injected
    
    # Create explanation with header
    explanation = llm_service.create_explanation_with_header(
        "Here is an explanation of your project",
        "This project involves creating a web application using React and Node.js..."
    )
    
    # Create grouped todo list
    todos = [
        {
            "title": "Complete React component setup",
            "due_datetime": datetime.now() + timedelta(days=1),
            "source": "ai_created"
        }
    ]
    todo_list = llm_service.create_grouped_todo_list(todos, "Your Todos")
    
    # Create grouped proposed task list
    tasks = [
        {
            "title": "Implement user authentication",
            "due_datetime": datetime.now() + timedelta(days=3),
            "priority": "high"
        }
    ]
    task_list = llm_service.create_grouped_proposed_task_list(tasks, "Proposed Tasks")
    
    return [explanation, todo_list, task_list]

if __name__ == "__main__":
    print("=== Explanation Block Example ===")
    print(example_explanation_block())
    print("\n=== Grouped Todos Example ===")
    print(example_grouped_todos())
    print("\n=== Grouped Proposed Tasks Example ===")
    print(example_grouped_proposed_tasks())
    print("\n=== LLM Service Usage Example ===")
    print(example_llm_service_usage())
