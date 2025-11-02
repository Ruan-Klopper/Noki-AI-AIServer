"""
UI Block models for structured responses
"""
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from pydantic import BaseModel, Field


class ResourceItem(BaseModel):
    """Resource item block"""
    type: str = "resource_item"
    item_type: str  # "PDF" | "Website" | "YouTube"
    title: str
    link: str
    footer: Optional[str] = None


class TodoItem(BaseModel):
    """Todo item within a todo list"""
    title: str
    description: Optional[str] = None
    project_id: Optional[str] = None
    task_id: Optional[str] = None
    due_datetime: Optional[datetime] = None
    source: Optional[str] = None  # "ai_created" or "input_received"


class TodoList(BaseModel):
    """Todo list block"""
    type: str = "todo_list"
    list_title: str
    items: List[TodoItem]
    footer: Optional[str] = None
    accept_decline: Optional[bool] = None
    grouped_by_date: Optional[bool] = None  # Enable date grouping


class ExplanationSection(BaseModel):
    """Section within an explanation block"""
    title: str
    description: Optional[str] = None
    list: Optional[List[str]] = None


class ExplanationBlock(BaseModel):
    """Explanation block"""
    type: str = "explanation_block"
    text: str  # Header text like "Here is an explanation of your project"
    explanation_content: str  # The actual explanation content
    footer: Optional[str] = None


class ProposedTask(BaseModel):
    """Proposed task item"""
    title: str
    description: Optional[str] = None
    due_datetime: Optional[datetime] = None
    priority: Optional[str] = None  # "high", "medium", "low"
    estimated_duration: Optional[str] = None


class ProposedTaskList(BaseModel):
    """Proposed task list block"""
    type: str = "proposed_task_list"
    list_title: str
    items: List[ProposedTask]
    footer: Optional[str] = None
    grouped_by_date: Optional[bool] = None  # Enable date grouping


class ProposedListItem(BaseModel):
    """Item within a proposed list"""
    title: str
    due_date: str  # ISO 8601 format datetime string
    is_all_day: bool


class ProposedList(BaseModel):
    """Proposed list block for todo planning"""
    type: str = "proposed_list"
    title: str  # Title of this todo group
    proposed_list_for_task_id: str  # ID of the task or todo this list is for
    items: List[ProposedListItem]  # List of todo items


class Confirmation(BaseModel):
    """Confirmation block"""
    type: str = "confirmation"
    message: str


# Union type for all possible blocks
UIBlock = Union[ResourceItem, TodoList, ExplanationBlock, ProposedTaskList, ProposedList, Confirmation]


class BlockFactory:
    """Factory for creating UI blocks"""
    
    @staticmethod
    def create_resource_item(item_type: str, title: str, link: str, footer: Optional[str] = None) -> Dict[str, Any]:
        """Create a resource item block"""
        return {
            "type": "resource_item",
            "item_type": item_type,
            "title": title,
            "link": link,
            "footer": footer
        }
    
    @staticmethod
    def create_todo_list(list_title: str, items: List[Dict[str, Any]], 
                        footer: Optional[str] = None, accept_decline: Optional[bool] = None,
                        grouped_by_date: Optional[bool] = None) -> Dict[str, Any]:
        """Create a todo list block"""
        return {
            "type": "todo_list",
            "list_title": list_title,
            "items": items,
            "footer": footer,
            "accept_decline": accept_decline,
            "grouped_by_date": grouped_by_date
        }
    
    @staticmethod
    def create_explanation_block(text: str, explanation_content: str, 
                               footer: Optional[str] = None) -> Dict[str, Any]:
        """Create an explanation block"""
        return {
            "type": "explanation_block",
            "text": text,
            "explanation_content": explanation_content,
            "footer": footer
        }
    
    @staticmethod
    def create_proposed_task_list(list_title: str, items: List[Dict[str, Any]], 
                                 footer: Optional[str] = None, grouped_by_date: Optional[bool] = None) -> Dict[str, Any]:
        """Create a proposed task list block"""
        return {
            "type": "proposed_task_list",
            "list_title": list_title,
            "items": items,
            "footer": footer,
            "grouped_by_date": grouped_by_date
        }
    
    @staticmethod
    def create_proposed_list(title: str, proposed_list_for_task_id: str, 
                            items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a proposed list block"""
        return {
            "type": "proposed_list",
            "title": title,
            "proposed_list_for_task_id": proposed_list_for_task_id,
            "items": items
        }
    
    @staticmethod
    def create_confirmation(message: str) -> Dict[str, Any]:
        """Create a confirmation block"""
        return {
            "type": "confirmation",
            "message": message
        }
