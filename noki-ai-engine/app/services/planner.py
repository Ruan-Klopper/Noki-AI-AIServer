"""
Planner service to transform LLM output into structured UI blocks
"""
import logging
import re
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from app.models.ui_blocks import BlockFactory

logger = logging.getLogger(__name__)


class PlannerService:
    """Service for transforming LLM output into structured UI blocks"""
    
    def __init__(self):
        self.block_factory = BlockFactory()
    
    def create_todo_list_from_assignments(self, assignments: List[Dict[str, Any]], 
                                        existing_todos: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a todo list block from assignment data with conflict checking"""
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
            
            # Break down assignments into actionable todos
            items = self._break_down_tasks_to_todos(items)
            
            # Check for conflicts with existing todos if provided
            if existing_todos:
                items = self._check_todo_conflicts(items, existing_todos)
            
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
    
    def _check_todo_conflicts(self, new_todos: List[Dict[str, Any]], 
                             existing_todos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Check for conflicts between new todos and existing todos.
        Returns optimized list of todos without conflicts.
        """
        try:
            logger.info(f"PlannerService: Checking conflicts between {len(new_todos)} new todos and {len(existing_todos)} existing todos")
            
            # Create a set of existing todo identifiers for quick lookup
            existing_todo_ids = set()
            existing_todo_titles = set()
            existing_todo_dates = {}
            
            for existing_todo in existing_todos:
                todo_id = existing_todo.get('todo_id')
                title = existing_todo.get('title', '').lower().strip()
                due_date = existing_todo.get('due_date') or existing_todo.get('due_datetime')
                
                if todo_id:
                    existing_todo_ids.add(todo_id)
                if title:
                    existing_todo_titles.add(title)
                if due_date:
                    existing_todo_dates[title] = due_date
            
            # Filter out conflicting todos
            filtered_todos = []
            conflicts_found = []
            
            for new_todo in new_todos:
                new_title = new_todo.get('title', '').lower().strip()
                new_due_date = new_todo.get('due_date') or new_todo.get('due_datetime')
                new_todo_id = new_todo.get('todo_id')
                
                # Check for exact ID match
                if new_todo_id and new_todo_id in existing_todo_ids:
                    conflicts_found.append({
                        'type': 'duplicate_id',
                        'todo': new_todo,
                        'reason': f"Todo with ID {new_todo_id} already exists"
                    })
                    continue
                
                # Check for title similarity (fuzzy matching)
                is_duplicate_title = False
                for existing_title in existing_todo_titles:
                    if self._titles_similar(new_title, existing_title):
                        is_duplicate_title = True
                        conflicts_found.append({
                            'type': 'duplicate_title',
                            'todo': new_todo,
                            'reason': f"Similar todo '{new_todo.get('title')}' already exists"
                        })
                        break
                
                if is_duplicate_title:
                    continue
                
                # Check for time conflicts (same day, similar time)
                if new_due_date and new_title in existing_todo_dates:
                    existing_date = existing_todo_dates[new_title]
                    if self._dates_conflict(new_due_date, existing_date):
                        conflicts_found.append({
                            'type': 'time_conflict',
                            'todo': new_todo,
                            'reason': f"Time conflict with existing todo on {existing_date}"
                        })
                        # Don't skip this one, but add a note
                        new_todo['conflict_note'] = f"⚠️ Time conflict with existing todo"
                
                filtered_todos.append(new_todo)
            
            # Log conflicts for debugging
            if conflicts_found:
                logger.info(f"PlannerService: Found {len(conflicts_found)} conflicts:")
                for conflict in conflicts_found:
                    logger.info(f"  - {conflict['type']}: {conflict['reason']}")
            
            logger.info(f"PlannerService: Filtered todos from {len(new_todos)} to {len(filtered_todos)} after conflict checking")
            return filtered_todos
            
        except Exception as e:
            logger.error(f"PlannerService: Error checking todo conflicts: {e}")
            # Return original todos if conflict checking fails
            return new_todos
    
    def _titles_similar(self, title1: str, title2: str, threshold: float = 0.8) -> bool:
        """
        Check if two todo titles are similar enough to be considered duplicates.
        Uses simple string similarity checking.
        """
        if not title1 or not title2:
            return False
        
        # Normalize titles
        title1 = title1.lower().strip()
        title2 = title2.lower().strip()
        
        # Exact match
        if title1 == title2:
            return True
        
        # Check if one title contains the other (for partial matches)
        if title1 in title2 or title2 in title1:
            return True
        
        # Simple word-based similarity
        words1 = set(title1.split())
        words2 = set(title2.split())
        
        if not words1 or not words2:
            return False
        
        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        similarity = intersection / union if union > 0 else 0
        
        return similarity >= threshold
    
    def _dates_conflict(self, date1: str, date2: str) -> bool:
        """
        Check if two dates conflict (same day, overlapping times).
        """
        try:
            # Parse dates
            if isinstance(date1, str):
                date1 = datetime.fromisoformat(date1.replace('Z', '+00:00'))
            if isinstance(date2, str):
                date2 = datetime.fromisoformat(date2.replace('Z', '+00:00'))
            
            # Check if same day
            return date1.date() == date2.date()
            
        except Exception as e:
            logger.error(f"PlannerService: Error comparing dates: {e}")
            return False
    
    def _break_down_tasks_to_todos(self, assignments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Break down assignments/tasks into smaller, actionable todos.
        This creates a comprehensive action plan with specific steps.
        """
        try:
            todos = []
            
            for assignment in assignments:
                title = assignment.get("title", "Untitled Task")
                description = assignment.get("description", "")
                project_id = assignment.get("project_id")
                task_id = assignment.get("task_id")
                due_datetime = assignment.get("due_datetime")
                
                # Generate specific todos based on the assignment type and content
                assignment_todos = self._generate_todos_for_assignment(
                    title, description, project_id, task_id, due_datetime
                )
                todos.extend(assignment_todos)
            
            logger.info(f"PlannerService: Broke down {len(assignments)} assignments into {len(todos)} actionable todos")
            return todos
            
        except Exception as e:
            logger.error(f"PlannerService: Error breaking down tasks to todos: {e}")
            # Fallback: return assignments as-is
            return assignments
    
    def _generate_todos_for_assignment(self, title: str, description: str, 
                                     project_id: str, task_id: str, due_datetime: str) -> List[Dict[str, Any]]:
        """
        Generate specific todos for a given assignment based on its content.
        """
        todos = []
        title_lower = title.lower()
        description_lower = description.lower()
        
        # Photography-related tasks
        if "photography" in title_lower or "portfolio" in title_lower or "photo" in title_lower:
            if "equipment" in title_lower or "maintenance" in title_lower:
                todos.extend([
                    {
                        "title": "Clean camera lenses",
                        "description": "Remove dust and fingerprints from all camera lenses",
                        "project_id": project_id,
                        "task_id": task_id,
                        "due_datetime": due_datetime,
                        "priority": "high",
                        "estimated_duration": "30 minutes"
                    },
                    {
                        "title": "Check camera settings",
                        "description": "Verify ISO, aperture, and shutter speed settings are optimal",
                        "project_id": project_id,
                        "task_id": task_id,
                        "due_datetime": due_datetime,
                        "priority": "medium",
                        "estimated_duration": "15 minutes"
                    },
                    {
                        "title": "Test camera functionality",
                        "description": "Take test shots to ensure camera is working properly",
                        "project_id": project_id,
                        "task_id": task_id,
                        "due_datetime": due_datetime,
                        "priority": "medium",
                        "estimated_duration": "20 minutes"
                    }
                ])
            elif "portfolio" in title_lower or "review" in title_lower:
                todos.extend([
                    {
                        "title": "Organize photo collection",
                        "description": "Sort photos by date, location, and subject matter",
                        "project_id": project_id,
                        "task_id": task_id,
                        "due_datetime": due_datetime,
                        "priority": "high",
                        "estimated_duration": "1 hour"
                    },
                    {
                        "title": "Select best photos",
                        "description": "Choose top 20-30 photos for portfolio inclusion",
                        "project_id": project_id,
                        "task_id": task_id,
                        "due_datetime": due_datetime,
                        "priority": "high",
                        "estimated_duration": "2 hours"
                    },
                    {
                        "title": "Edit selected photos",
                        "description": "Apply color correction, cropping, and enhancement",
                        "project_id": project_id,
                        "task_id": task_id,
                        "due_datetime": due_datetime,
                        "priority": "medium",
                        "estimated_duration": "3 hours"
                    },
                    {
                        "title": "Create portfolio layout",
                        "description": "Design and arrange photos in portfolio format",
                        "project_id": project_id,
                        "task_id": task_id,
                        "due_datetime": due_datetime,
                        "priority": "medium",
                        "estimated_duration": "2 hours"
                    }
                ])
        
        # Research-related tasks
        elif "research" in title_lower or "literature" in title_lower or "paper" in title_lower:
            if "notes" in title_lower or "compilation" in title_lower:
                todos.extend([
                    {
                        "title": "Gather research materials",
                        "description": "Collect all relevant articles, books, and sources",
                        "project_id": project_id,
                        "task_id": task_id,
                        "due_datetime": due_datetime,
                        "priority": "high",
                        "estimated_duration": "1 hour"
                    },
                    {
                        "title": "Read and annotate sources",
                        "description": "Read through materials and take detailed notes",
                        "project_id": project_id,
                        "task_id": task_id,
                        "due_datetime": due_datetime,
                        "priority": "high",
                        "estimated_duration": "3 hours"
                    },
                    {
                        "title": "Organize notes by topic",
                        "description": "Categorize notes into logical themes and topics",
                        "project_id": project_id,
                        "task_id": task_id,
                        "due_datetime": due_datetime,
                        "priority": "medium",
                        "estimated_duration": "1 hour"
                    },
                    {
                        "title": "Create reference bibliography",
                        "description": "Format all sources into proper citation style",
                        "project_id": project_id,
                        "task_id": task_id,
                        "due_datetime": due_datetime,
                        "priority": "medium",
                        "estimated_duration": "45 minutes"
                    }
                ])
            elif "literature" in title_lower:
                todos.extend([
                    {
                        "title": "Define research scope",
                        "description": "Identify key themes and research questions",
                        "project_id": project_id,
                        "task_id": task_id,
                        "due_datetime": due_datetime,
                        "priority": "high",
                        "estimated_duration": "1 hour"
                    },
                    {
                        "title": "Search academic databases",
                        "description": "Find relevant peer-reviewed articles and studies",
                        "project_id": project_id,
                        "task_id": task_id,
                        "due_datetime": due_datetime,
                        "priority": "high",
                        "estimated_duration": "2 hours"
                    },
                    {
                        "title": "Analyze and synthesize findings",
                        "description": "Compare and contrast different research findings",
                        "project_id": project_id,
                        "task_id": task_id,
                        "due_datetime": due_datetime,
                        "priority": "high",
                        "estimated_duration": "4 hours"
                    },
                    {
                        "title": "Write literature review draft",
                        "description": "Create comprehensive review of existing research",
                        "project_id": project_id,
                        "task_id": task_id,
                        "due_datetime": due_datetime,
                        "priority": "medium",
                        "estimated_duration": "3 hours"
                    },
                    {
                        "title": "Revise and edit review",
                        "description": "Polish the literature review for clarity and flow",
                        "project_id": project_id,
                        "task_id": task_id,
                        "due_datetime": due_datetime,
                        "priority": "medium",
                        "estimated_duration": "2 hours"
                    }
                ])
        
        # Generic task breakdown if no specific pattern matches
        else:
            todos.extend([
                {
                    "title": f"Plan approach for {title}",
                    "description": f"Break down the task and create a step-by-step plan",
                    "project_id": project_id,
                    "task_id": task_id,
                    "due_datetime": due_datetime,
                    "priority": "high",
                    "estimated_duration": "30 minutes"
                },
                {
                    "title": f"Gather resources for {title}",
                    "description": f"Collect all necessary materials and information",
                    "project_id": project_id,
                    "task_id": task_id,
                    "due_datetime": due_datetime,
                    "priority": "high",
                    "estimated_duration": "1 hour"
                },
                {
                    "title": f"Execute main work for {title}",
                    "description": description or f"Complete the primary work for this task",
                    "project_id": project_id,
                    "task_id": task_id,
                    "due_datetime": due_datetime,
                    "priority": "high",
                    "estimated_duration": "2 hours"
                },
                {
                    "title": f"Review and finalize {title}",
                    "description": f"Check work quality and make final adjustments",
                    "project_id": project_id,
                    "task_id": task_id,
                    "due_datetime": due_datetime,
                    "priority": "medium",
                    "estimated_duration": "45 minutes"
                }
            ])
        
        return todos
