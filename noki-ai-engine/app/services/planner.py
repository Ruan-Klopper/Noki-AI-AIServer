"""
Planner service to transform LLM output into structured UI blocks
"""
import logging
import re
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from app.models.ui_blocks import BlockFactory
from app.models.schemas import AIIntent, IntentType

logger = logging.getLogger(__name__)


class PlannerService:
    """Service for transforming LLM output into structured UI blocks"""
    
    def __init__(self):
        self.block_factory = BlockFactory()
    
    def create_todo_list_from_assignments(self, assignments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a todo list block from assignment data"""
        try:
            items = []
            for assignment in assignments:
                item = {
                    "title": assignment.get("title", "Untitled Assignment"),
                    "description": assignment.get("description"),
                    "project_id": assignment.get("project_id"),
                    "task_id": assignment.get("task_id"),
                    "due_datetime": assignment.get("due_datetime")
                }
                items.append(item)
            
            return self.block_factory.create_todo_list(
                list_title="Upcoming Assignments",
                items=items,
                footer="Would you like me to create a study schedule for these?"
            )
            
        except Exception as e:
            logger.error(f"Failed to create todo list from assignments: {e}")
            return self.block_factory.create_confirmation("Failed to process assignments.")
    
    def create_study_schedule_proposal(self, assignments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a study schedule proposal"""
        try:
            sessions = []
            current_time = datetime.utcnow()
            
            for i, assignment in enumerate(assignments):
                # Create study sessions (simplified logic)
                session_start = current_time + timedelta(days=i, hours=9)
                session_end = session_start + timedelta(hours=2)
                
                sessions.append({
                    "title": f"Study: {assignment.get('title', 'Assignment')}",
                    "start": session_start.isoformat() + "Z",
                    "end": session_end.isoformat() + "Z",
                    "project_id": assignment.get("project_id"),
                    "task_id": assignment.get("task_id")
                })
            
            return {
                "type": "explanation_block",
                "title": "Proposed Study Schedule",
                "description": "I've created a study schedule based on your assignments.",
                "blocks": [
                    {
                        "title": "Study Sessions",
                        "list": [f"{s['title']} - {s['start']} to {s['end']}" for s in sessions]
                    }
                ],
                "footer": "Accept this schedule?"
            }
            
        except Exception as e:
            logger.error(f"Failed to create study schedule proposal: {e}")
            return self.block_factory.create_confirmation("Failed to create study schedule.")
    
    def create_explanation_from_resources(self, resources: List[Dict[str, Any]], 
                                        query: str) -> Dict[str, Any]:
        """Create an explanation block from resource analysis"""
        try:
            sections = []
            
            for resource in resources:
                section = {
                    "title": resource.get("title", "Resource"),
                    "description": resource.get("summary", "No summary available"),
                    "list": resource.get("key_points", [])
                }
                sections.append(section)
            
            return self.block_factory.create_explanation_block(
                title=f"Analysis: {query}",
                description="Based on your available resources:",
                blocks=sections,
                footer="Would you like me to elaborate on any of these points?"
            )
            
        except Exception as e:
            logger.error(f"Failed to create explanation from resources: {e}")
            return self.block_factory.create_confirmation("Failed to analyze resources.")
    
    def create_task_breakdown(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Create a task breakdown with subtasks"""
        try:
            subtasks = self._generate_subtasks(task)
            
            return self.block_factory.create_todo_list(
                list_title=f"Breakdown: {task.get('title', 'Task')}",
                items=subtasks,
                footer="Would you like me to schedule these subtasks?"
            )
            
        except Exception as e:
            logger.error(f"Failed to create task breakdown: {e}")
            return self.block_factory.create_confirmation("Failed to break down task.")
    
    def create_learning_plan(self, topic: str, resources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a structured learning plan"""
        try:
            sections = [
                {
                    "title": "Learning Objectives",
                    "list": [
                        f"Understand key concepts of {topic}",
                        "Apply knowledge to practical problems",
                        "Complete related assignments"
                    ]
                },
                {
                    "title": "Study Resources",
                    "list": [r.get("title", "Resource") for r in resources]
                },
                {
                    "title": "Study Schedule",
                    "list": [
                        "Week 1: Review materials and take notes",
                        "Week 2: Practice problems and exercises",
                        "Week 3: Complete assignments and review"
                    ]
                }
            ]
            
            return self.block_factory.create_explanation_block(
                title=f"Learning Plan: {topic}",
                description="Here's a structured approach to mastering this topic:",
                blocks=sections,
                footer="Would you like me to create specific study sessions?"
            )
            
        except Exception as e:
            logger.error(f"Failed to create learning plan: {e}")
            return self.block_factory.create_confirmation("Failed to create learning plan.")
    
    def create_progress_summary(self, projects: List[Dict[str, Any]], 
                              tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a progress summary"""
        try:
            sections = []
            
            # Project progress
            if projects:
                project_section = {
                    "title": "Project Progress",
                    "list": [f"{p.get('title', 'Project')} - {p.get('status', 'Unknown')}" 
                            for p in projects]
                }
                sections.append(project_section)
            
            # Task progress
            if tasks:
                task_section = {
                    "title": "Task Progress",
                    "list": [f"{t.get('title', 'Task')} - {t.get('status', 'Unknown')}" 
                            for t in tasks]
                }
                sections.append(task_section)
            
            return self.block_factory.create_explanation_block(
                title="Academic Progress Summary",
                description="Here's an overview of your current progress:",
                blocks=sections,
                footer="Would you like me to suggest next steps?"
            )
            
        except Exception as e:
            logger.error(f"Failed to create progress summary: {e}")
            return self.block_factory.create_confirmation("Failed to create progress summary.")
    
    def create_intent_response(self, intent: AIIntent, context_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create response blocks based on intent and context data"""
        try:
            blocks = []
            
            if intent.type == IntentType.BACKEND_QUERY:
                if "assignments" in (intent.targets or []):
                    assignments = context_data.get("assignments", [])
                    if assignments:
                        blocks.append(self.create_todo_list_from_assignments(assignments))
                
                if "schedule" in (intent.targets or []):
                    schedule = context_data.get("schedule", [])
                    if schedule:
                        blocks.append(self._create_schedule_block(schedule))
            
            elif intent.type == IntentType.PROPOSED_SCHEDULE:
                sessions = intent.payload.get("sessions", []) if intent.payload else []
                if sessions:
                    blocks.append(self._create_schedule_proposal_block(sessions))
            
            elif intent.type == IntentType.PROPOSED_TASKS:
                tasks = intent.payload.get("tasks", []) if intent.payload else []
                if tasks:
                    blocks.append(self._create_task_proposal_block(tasks))
            
            return blocks
            
        except Exception as e:
            logger.error(f"Failed to create intent response: {e}")
            return [self.block_factory.create_confirmation("Failed to process intent response.")]
    
    def _generate_subtasks(self, task: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate subtasks for a given task"""
        # This is a simplified implementation
        # In production, you'd use more sophisticated task decomposition
        subtasks = [
            {
                "title": f"Research {task.get('title', 'task')}",
                "description": "Gather information and resources",
                "project_id": task.get("project_id")
            },
            {
                "title": f"Plan {task.get('title', 'task')}",
                "description": "Create detailed plan and timeline",
                "project_id": task.get("project_id")
            },
            {
                "title": f"Execute {task.get('title', 'task')}",
                "description": "Complete the main work",
                "project_id": task.get("project_id"),
                "due_datetime": task.get("due_datetime")
            }
        ]
        
        return subtasks
    
    def _create_schedule_block(self, schedule: Dict[str, Any]) -> Dict[str, Any]:
        """Create a schedule display block"""
        try:
            items = []
            available_slots = schedule.get("available_slots", [])
            
            for slot in available_slots:
                date = slot.get("date", "Unknown Date")
                time_slots = slot.get("time_slots", [])
                
                for time_slot in time_slots:
                    items.append({
                        "title": f"Available - {time_slot.get('type', 'General')}",
                        "description": f"{date} from {time_slot.get('start', '')} to {time_slot.get('end', '')}",
                        "due_datetime": f"{date}T{time_slot.get('start', '09:00')}:00Z"
                    })
            
            return self.block_factory.create_todo_list(
                list_title="Available Time Slots",
                items=items,
                footer="These are your available time slots for scheduling tasks."
            )
            
        except Exception as e:
            logger.error(f"Failed to create schedule block: {e}")
            return self.block_factory.create_confirmation("Failed to process schedule data.")
    
    def _create_schedule_proposal_block(self, sessions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a schedule proposal block"""
        items = []
        for session in sessions:
            items.append({
                "title": session.get("title", "Study Session"),
                "description": f"{session.get('start', '')} - {session.get('end', '')}",
                "due_datetime": session.get("start")
            })
        
        return self.block_factory.create_todo_list(
            list_title="Proposed Study Sessions",
            items=items,
            accept_decline=True,
            footer="Accept this study schedule?"
        )
    
    def _create_task_proposal_block(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a task proposal block"""
        items = []
        for task in tasks:
            items.append({
                "title": task.get("title", "New Task"),
                "description": task.get("description"),
                "project_id": task.get("project_id"),
                "due_datetime": task.get("due_datetime")
            })
        
        return self.block_factory.create_todo_list(
            list_title="Proposed Tasks",
            items=items,
            accept_decline=True,
            footer="Accept these tasks?"
        )
