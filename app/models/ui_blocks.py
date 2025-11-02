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


class TodoList(BaseModel):
    """Todo list block"""
    type: str = "todo_list"
    list_title: str
    items: List[TodoItem]
    footer: Optional[str] = None
    accept_decline: Optional[bool] = None


class ExplanationSection(BaseModel):
    """Section within an explanation block"""
    title: str
    description: Optional[str] = None
    list: Optional[List[str]] = None


class ExplanationBlock(BaseModel):
    """Explanation block"""
    type: str = "explanation_block"
    title: str
    description: Optional[str] = None
    blocks: List[ExplanationSection]
    footer: Optional[str] = None


class Confirmation(BaseModel):
    """Confirmation block"""
    type: str = "confirmation"
    message: str


# Union type for all possible blocks
UIBlock = Union[ResourceItem, TodoList, ExplanationBlock, Confirmation]


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
                        footer: Optional[str] = None, accept_decline: Optional[bool] = None) -> Dict[str, Any]:
        """Create a todo list block"""
        return {
            "type": "todo_list",
            "list_title": list_title,
            "items": items,
            "footer": footer,
            "accept_decline": accept_decline
        }
    
    @staticmethod
    def create_explanation_block(title: str, blocks: List[Dict[str, Any]], 
                               description: Optional[str] = None, footer: Optional[str] = None) -> Dict[str, Any]:
        """Create an explanation block"""
        return {
            "type": "explanation_block",
            "title": title,
            "description": description,
            "blocks": blocks,
            "footer": footer
        }
    
    @staticmethod
    def create_confirmation(message: str) -> Dict[str, Any]:
        """Create a confirmation block"""
        return {
            "type": "confirmation",
            "message": message
        }
