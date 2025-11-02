"""
Token usage service for tracking and calculating token costs
"""
import logging
from typing import Dict, Any, Optional, List
import tiktoken

from config import settings

logger = logging.getLogger(__name__)


class TokenUsageService:
    """Service for tracking token usage and calculating costs"""
    
    def __init__(self):
        # Initialize tokenizer for the model
        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        except KeyError:
            # Fallback to cl100k_base encoding (used by GPT-4)
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Token costs per 1K tokens (as of 2024)
        self.token_costs = {
            "gpt-4o": {
                "prompt": 0.005,  # $5 per 1M tokens
                "completion": 0.015,  # $15 per 1M tokens
                "embedding": 0.0001  # $0.10 per 1M tokens
            },
            "gpt-4": {
                "prompt": 0.03,  # $30 per 1M tokens
                "completion": 0.06,  # $60 per 1M tokens
                "embedding": 0.0001
            },
            "gpt-3.5-turbo": {
                "prompt": 0.0015,  # $1.5 per 1M tokens
                "completion": 0.002,  # $2 per 1M tokens
                "embedding": 0.0001
            }
        }
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string"""
        try:
            return len(self.tokenizer.encode(text))
        except Exception as e:
            logger.error(f"Error counting tokens: {e}")
            # Fallback: rough estimation (1 token ≈ 4 characters)
            return len(text) // 4
    
    def count_embedding_tokens(self, text: str) -> int:
        """Count tokens for embedding operations"""
        # Embeddings use a different tokenizer (text-embedding-ada-002)
        try:
            # Use cl100k_base for embeddings (same as GPT-4)
            return len(self.tokenizer.encode(text))
        except Exception as e:
            logger.error(f"Error counting embedding tokens: {e}")
            return len(text) // 4
    
    def calculate_cost(self, prompt_tokens: int, completion_tokens: int, 
                      embedding_tokens: int = 0, model: str = None) -> float:
        """Calculate estimated cost in USD"""
        try:
            model = model or settings.openai_model
            costs = self.token_costs.get(model, self.token_costs["gpt-4o"])
            
            prompt_cost = (prompt_tokens / 1000) * costs["prompt"]
            completion_cost = (completion_tokens / 1000) * costs["completion"]
            embedding_cost = (embedding_tokens / 1000) * costs["embedding"]
            
            total_cost = prompt_cost + completion_cost + embedding_cost
            return round(total_cost, 6)  # Round to 6 decimal places
            
        except Exception as e:
            logger.error(f"Error calculating cost: {e}")
            return 0.0
    
    def create_token_usage(self, prompt_tokens: int, completion_tokens: int,
                          embedding_tokens: int = 0, model: str = None) -> Dict[str, Any]:
        """Create token usage object with cost calculation"""
        total_tokens = prompt_tokens + completion_tokens + embedding_tokens
        cost = self.calculate_cost(prompt_tokens, completion_tokens, embedding_tokens, model)
        
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "embedding_tokens": embedding_tokens,
            "cost_estimate_usd": cost
        }
    
    def count_context_tokens(self, context: List[Dict[str, Any]]) -> int:
        """Count tokens in context documents"""
        total_tokens = 0
        for doc in context:
            if hasattr(doc, 'page_content'):
                total_tokens += self.count_tokens(doc.page_content)
            elif isinstance(doc, dict) and 'page_content' in doc:
                total_tokens += self.count_tokens(doc['page_content'])
        return total_tokens
    
    def count_projects_tokens(self, projects: List[Dict[str, Any]]) -> int:
        """Count tokens in project data"""
        total_tokens = 0
        for project in projects:
            total_tokens += self.count_tokens(project.get('title', ''))
            total_tokens += self.count_tokens(project.get('description', ''))
            total_tokens += self.count_tokens(project.get('instructor', ''))
        return total_tokens
    
    def count_tasks_tokens(self, tasks: List[Dict[str, Any]]) -> int:
        """Count tokens in task data"""
        total_tokens = 0
        for task in tasks:
            total_tokens += self.count_tokens(task.get('title', ''))
            total_tokens += self.count_tokens(task.get('description', ''))
        return total_tokens
    
    def estimate_prompt_tokens(self, chat_input: Dict[str, Any], 
                             context: List[Dict[str, Any]] = None) -> int:
        """Estimate total prompt tokens including system prompt and context"""
        try:
            # System prompt tokens (approximate)
            system_prompt_tokens = 200  # Base system prompt
            
            # User prompt tokens
            user_prompt_tokens = self.count_tokens(chat_input.get('prompt', ''))
            
            # Context tokens
            context_tokens = 0
            if context:
                context_tokens = self.count_context_tokens(context)
            
            # Projects tokens
            projects_tokens = 0
            if chat_input.get('projects'):
                projects_tokens = self.count_projects_tokens(chat_input['projects'])
            
            # Tasks tokens
            tasks_tokens = 0
            if chat_input.get('tasks'):
                tasks_tokens = self.count_tasks_tokens(chat_input['tasks'])
            
            total_prompt_tokens = (
                system_prompt_tokens + 
                user_prompt_tokens + 
                context_tokens + 
                projects_tokens + 
                tasks_tokens
            )
            
            return total_prompt_tokens
            
        except Exception as e:
            logger.error(f"Error estimating prompt tokens: {e}")
            return 0
    
    def get_model_info(self, model: str = None) -> Dict[str, Any]:
        """Get model information including token limits and costs"""
        model = model or settings.openai_model
        
        model_info = {
            "model": model,
            "max_tokens": 4096,  # Default max tokens
            "costs_per_1k_tokens": self.token_costs.get(model, self.token_costs["gpt-4o"])
        }
        
        # Update max tokens based on model
        if "gpt-4o" in model:
            model_info["max_tokens"] = 128000
        elif "gpt-4" in model:
            model_info["max_tokens"] = 8192
        elif "gpt-3.5" in model:
            model_info["max_tokens"] = 16384
        
        return model_info
