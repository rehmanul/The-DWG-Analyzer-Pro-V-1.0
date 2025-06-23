import json
import sqlite3
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import uuid

@dataclass
class Comment:
    """Collaborative comment on a spatial location"""
    id: str
    author: str
    content: str
    position: Dict[str, float]  # x, y coordinates
    timestamp: datetime
    resolved: bool
    space_id: str
    category: str  # 'suggestion', 'issue', 'approval', 'question'

@dataclass
class ProjectCollaborator:
    """Project team member"""
    id: str
    name: str
    email: str
    role: str  # 'architect', 'designer', 'client', 'contractor', 'engineer'
    permissions: List[str]
    last_active: datetime

class CollaborationManager:
    """Manages collaborative features for architectural projects"""
    
    def __init__(self, db_path: str = "collaboration.db"):
        self.db_path = db_path
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize collaboration database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Comments table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS comments (
            id TEXT PRIMARY KEY,
            author TEXT NOT NULL,
            content TEXT NOT NULL,
            position_x REAL,
            position_y REAL,
            timestamp TEXT,
            resolved BOOLEAN,
            space_id TEXT,
            category TEXT
        )
        ''')
        
        # Collaborators table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS collaborators (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT UNIQUE,
            role TEXT,
            permissions TEXT,
            last_active TEXT
        )
        ''')
        
        # Project activity log
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS activity_log (
            id TEXT PRIMARY KEY,
            user_id TEXT,
            action TEXT,
            details TEXT,
            timestamp TEXT
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_comment(self, author: str, content: str, position: Dict[str, float], 
                   space_id: str, category: str = 'suggestion') -> str:
        """Add a collaborative comment"""
        comment_id = str(uuid.uuid4())
        comment = Comment(
            id=comment_id,
            author=author,
            content=content,
            position=position,
            timestamp=datetime.now(),
            resolved=False,
            space_id=space_id,
            category=category
        )
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO comments VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            comment.id, comment.author, comment.content,
            comment.position['x'], comment.position['y'],
            comment.timestamp.isoformat(), comment.resolved,
            comment.space_id, comment.category
        ))
        
        conn.commit()
        conn.close()
        
        # Log activity
        self._log_activity(author, 'comment_added', f'Added comment in space {space_id}')
        
        return comment_id
    
    def get_comments_for_project(self, project_id: str = None) -> List[Comment]:
        """Get all comments for a project"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM comments ORDER BY timestamp DESC')
        rows = cursor.fetchall()
        conn.close()
        
        comments = []
        for row in rows:
            comment = Comment(
                id=row[0], author=row[1], content=row[2],
                position={'x': row[3], 'y': row[4]},
                timestamp=datetime.fromisoformat(row[5]),
                resolved=row[6], space_id=row[7], category=row[8]
            )
            comments.append(comment)
        
        return comments
    
    def resolve_comment(self, comment_id: str, user: str):
        """Mark a comment as resolved"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            'UPDATE comments SET resolved = ? WHERE id = ?',
            (True, comment_id)
        )
        
        conn.commit()
        conn.close()
        
        self._log_activity(user, 'comment_resolved', f'Resolved comment {comment_id}')
    
    def add_collaborator(self, name: str, email: str, role: str, 
                        permissions: List[str]) -> str:
        """Add a team collaborator"""
        collaborator_id = str(uuid.uuid4())
        collaborator = ProjectCollaborator(
            id=collaborator_id,
            name=name,
            email=email,
            role=role,
            permissions=permissions,
            last_active=datetime.now()
        )
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
            INSERT INTO collaborators VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                collaborator.id, collaborator.name, collaborator.email,
                collaborator.role, json.dumps(collaborator.permissions),
                collaborator.last_active.isoformat()
            ))
            
            conn.commit()
            conn.close()
            
            return collaborator_id
        except sqlite3.IntegrityError:
            conn.close()
            return None  # Email already exists
    
    def get_project_collaborators(self) -> List[ProjectCollaborator]:
        """Get all project collaborators"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM collaborators ORDER BY last_active DESC')
        rows = cursor.fetchall()
        conn.close()
        
        collaborators = []
        for row in rows:
            collaborator = ProjectCollaborator(
                id=row[0], name=row[1], email=row[2], role=row[3],
                permissions=json.loads(row[4]),
                last_active=datetime.fromisoformat(row[5])
            )
            collaborators.append(collaborator)
        
        return collaborators
    
    def _log_activity(self, user_id: str, action: str, details: str):
        """Log user activity"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO activity_log VALUES (?, ?, ?, ?, ?)
        ''', (
            str(uuid.uuid4()), user_id, action, details,
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def get_activity_feed(self, limit: int = 50) -> List[Dict]:
        """Get recent project activity"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT * FROM activity_log 
        ORDER BY timestamp DESC 
        LIMIT ?
        ''', (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        activities = []
        for row in rows:
            activity = {
                'id': row[0],
                'user_id': row[1],
                'action': row[2],
                'details': row[3],
                'timestamp': row[4]
            }
            activities.append(activity)
        
        return activities

class TeamPlanningInterface:
    """Interface for team planning and coordination"""
    
    def __init__(self, collaboration_manager: CollaborationManager):
        self.collaboration = collaboration_manager
        self.role_permissions = {
            'architect': ['edit', 'comment', 'approve', 'export'],
            'designer': ['edit', 'comment', 'export'],
            'client': ['comment', 'approve'],
            'contractor': ['comment', 'view'],
            'engineer': ['edit', 'comment', 'export']
        }
    
    def create_review_session(self, spaces: List[Dict], participants: List[str]) -> Dict:
        """Create a collaborative review session"""
        session_id = str(uuid.uuid4())
        
        review_data = {
            'session_id': session_id,
            'participants': participants,
            'spaces': spaces,
            'created_at': datetime.now().isoformat(),
            'status': 'active',
            'comments': [],
            'decisions': []
        }
        
        return review_data
    
    def check_user_permissions(self, user_role: str, action: str) -> bool:
        """Check if user has permission for action"""
        return action in self.role_permissions.get(user_role, [])
    
    def generate_team_report(self) -> Dict:
        """Generate comprehensive team collaboration report"""
        collaborators = self.collaboration.get_project_collaborators()
        comments = self.collaboration.get_comments_for_project()
        activities = self.collaboration.get_activity_feed()
        
        # Analyze collaboration metrics
        total_comments = len(comments)
        resolved_comments = len([c for c in comments if c.resolved])
        active_collaborators = len([c for c in collaborators 
                                  if (datetime.now() - c.last_active).days < 7])
        
        comment_categories = {}
        for comment in comments:
            category = comment.category
            comment_categories[category] = comment_categories.get(category, 0) + 1
        
        return {
            'team_size': len(collaborators),
            'active_members': active_collaborators,
            'total_comments': total_comments,
            'resolved_comments': resolved_comments,
            'resolution_rate': (resolved_comments / total_comments * 100) if total_comments > 0 else 0,
            'comment_categories': comment_categories,
            'recent_activity_count': len(activities),
            'collaboration_score': self._calculate_collaboration_score(
                collaborators, comments, activities
            )
        }
    
    def _calculate_collaboration_score(self, collaborators: List, 
                                     comments: List, activities: List) -> float:
        """Calculate overall collaboration effectiveness score"""
        if not collaborators:
            return 0.0
        
        # Factors: team engagement, comment resolution, activity level
        engagement_score = len([c for c in collaborators 
                              if (datetime.now() - c.last_active).days < 3]) / len(collaborators)
        
        resolution_score = (len([c for c in comments if c.resolved]) / 
                          max(len(comments), 1))
        
        activity_score = min(len(activities) / 20, 1.0)  # Normalize to 20 activities = 1.0
        
        return (engagement_score * 0.4 + resolution_score * 0.4 + activity_score * 0.2) * 100