import asyncio
import websockets
import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
import threading
import queue
from sqlalchemy import create_engine, Column, String, DateTime, Text, Float, Integer, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import hashlib

Base = declarative_base()

class Project(Base):
    __tablename__ = 'projects'
    
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(Text)
    created_by = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String, default='active')
    data = Column(Text)  # JSON storage for project data

class Comment(Base):
    __tablename__ = 'comments'
    
    id = Column(String, primary_key=True)
    project_id = Column(String, nullable=False)
    user_id = Column(String, nullable=False)
    username = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    x_position = Column(Float)
    y_position = Column(Float)
    zone_id = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    resolved = Column(Boolean, default=False)

class Version(Base):
    __tablename__ = 'versions'
    
    id = Column(String, primary_key=True)
    project_id = Column(String, nullable=False)
    version_number = Column(Integer, nullable=False)
    created_by = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    description = Column(Text)
    data = Column(Text)  # JSON storage for version data
    checksum = Column(String)

@dataclass
class User:
    user_id: str
    username: str
    role: str
    permissions: List[str]
    last_active: datetime
    current_project: Optional[str] = None

@dataclass
class CollaborationMessage:
    message_type: str
    user_id: str
    project_id: str
    data: Dict[str, Any]
    timestamp: datetime

class CollaborationManager:
    """
    Real-time collaboration system for team-based architectural planning
    """
    
    def __init__(self, database_url: str = "sqlite:///collaboration.db"):
        self.engine = create_engine(database_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        
        self.active_sessions = {}  # user_id -> websocket
        self.active_projects = {}  # project_id -> set of user_ids
        self.user_cursors = {}     # user_id -> {x, y, project_id}
        self.project_locks = {}    # zone_id -> user_id
        
        self.message_queue = queue.Queue()
        self.event_handlers = {
            'cursor_move': self._handle_cursor_move,
            'zone_select': self._handle_zone_select,
            'zone_edit': self._handle_zone_edit,
            'comment_add': self._handle_comment_add,
            'comment_resolve': self._handle_comment_resolve,
            'analysis_update': self._handle_analysis_update,
            'user_join': self._handle_user_join,
            'user_leave': self._handle_user_leave
        }
    
    async def start_collaboration_server(self, host: str = "localhost", port: int = 8765):
        """Start the WebSocket collaboration server"""
        async def handle_client(websocket, path):
            user_id = None
            try:
                async for message in websocket:
                    await self._process_message(websocket, message)
            except websockets.exceptions.ConnectionClosed:
                if user_id:
                    await self._handle_user_disconnect(user_id)
            except Exception as e:
                print(f"Error handling client: {e}")
        
        start_server = websockets.serve(handle_client, host, port)
        await start_server
    
    async def _process_message(self, websocket, raw_message: str):
        """Process incoming WebSocket message"""
        try:
            message_data = json.loads(raw_message)
            
            message = CollaborationMessage(
                message_type=message_data['type'],
                user_id=message_data['user_id'],
                project_id=message_data['project_id'],
                data=message_data['data'],
                timestamp=datetime.utcnow()
            )
            
            # Store websocket connection
            self.active_sessions[message.user_id] = websocket
            
            # Handle the message
            if message.message_type in self.event_handlers:
                await self.event_handlers[message.message_type](message)
            
            # Broadcast to relevant users
            await self._broadcast_to_project(message)
            
        except json.JSONDecodeError:
            await websocket.send(json.dumps({
                'type': 'error',
                'message': 'Invalid JSON format'
            }))
        except Exception as e:
            await websocket.send(json.dumps({
                'type': 'error',
                'message': str(e)
            }))
    
    async def _broadcast_to_project(self, message: CollaborationMessage):
        """Broadcast message to all users in the project"""
        project_users = self.active_projects.get(message.project_id, set())
        
        broadcast_data = {
            'type': message.message_type,
            'user_id': message.user_id,
            'project_id': message.project_id,
            'data': message.data,
            'timestamp': message.timestamp.isoformat()
        }
        
        # Send to all active users in the project
        for user_id in project_users:
            if user_id != message.user_id and user_id in self.active_sessions:
                websocket = self.active_sessions[user_id]
                try:
                    await websocket.send(json.dumps(broadcast_data))
                except:
                    # Remove disconnected user
                    self.active_sessions.pop(user_id, None)
                    project_users.discard(user_id)
    
    async def _handle_cursor_move(self, message: CollaborationMessage):
        """Handle real-time cursor movement"""
        self.user_cursors[message.user_id] = {
            'x': message.data['x'],
            'y': message.data['y'],
            'project_id': message.project_id,
            'timestamp': message.timestamp
        }
    
    async def _handle_zone_select(self, message: CollaborationMessage):
        """Handle zone selection and locking"""
        zone_id = message.data['zone_id']
        
        # Check if zone is already locked
        if zone_id in self.project_locks:
            current_locker = self.project_locks[zone_id]
            if current_locker != message.user_id:
                # Zone is locked by another user
                await self._send_to_user(message.user_id, {
                    'type': 'zone_locked',
                    'zone_id': zone_id,
                    'locked_by': current_locker
                })
                return
        
        # Lock the zone
        self.project_locks[zone_id] = message.user_id
        
        # Set lock timeout (auto-release after 5 minutes of inactivity)
        asyncio.create_task(self._auto_release_lock(zone_id, message.user_id, 300))
    
    async def _handle_zone_edit(self, message: CollaborationMessage):
        """Handle zone editing with conflict resolution"""
        zone_id = message.data['zone_id']
        
        # Verify user has lock on the zone
        if self.project_locks.get(zone_id) != message.user_id:
            await self._send_to_user(message.user_id, {
                'type': 'edit_denied',
                'zone_id': zone_id,
                'reason': 'Zone not locked by user'
            })
            return
        
        # Store the edit in database for version control
        await self._save_zone_edit(message)
    
    async def _handle_comment_add(self, message: CollaborationMessage):
        """Handle adding comments to the plan"""
        comment_id = str(uuid.uuid4())
        
        session = self.Session()
        try:
            comment = Comment(
                id=comment_id,
                project_id=message.project_id,
                user_id=message.user_id,
                username=message.data['username'],
                content=message.data['content'],
                x_position=message.data.get('x_position'),
                y_position=message.data.get('y_position'),
                zone_id=message.data.get('zone_id')
            )
            session.add(comment)
            session.commit()
            
            # Add comment_id to the broadcast data
            message.data['comment_id'] = comment_id
            
        finally:
            session.close()
    
    async def _handle_comment_resolve(self, message: CollaborationMessage):
        """Handle resolving comments"""
        comment_id = message.data['comment_id']
        
        session = self.Session()
        try:
            comment = session.query(Comment).filter_by(id=comment_id).first()
            if comment:
                comment.resolved = True
                session.commit()
        finally:
            session.close()
    
    async def _handle_analysis_update(self, message: CollaborationMessage):
        """Handle real-time analysis updates"""
        # Store analysis results and broadcast to team
        await self._save_analysis_results(message)
    
    async def _handle_user_join(self, message: CollaborationMessage):
        """Handle user joining a project"""
        project_id = message.project_id
        user_id = message.user_id
        
        if project_id not in self.active_projects:
            self.active_projects[project_id] = set()
        
        self.active_projects[project_id].add(user_id)
        
        # Send current project state to joining user
        await self._send_project_state(user_id, project_id)
    
    async def _handle_user_leave(self, message: CollaborationMessage):
        """Handle user leaving a project"""
        user_id = message.user_id
        project_id = message.project_id
        
        # Remove user from active projects
        if project_id in self.active_projects:
            self.active_projects[project_id].discard(user_id)
        
        # Release any locks held by the user
        locks_to_release = [zone_id for zone_id, locked_by in self.project_locks.items() 
                           if locked_by == user_id]
        
        for zone_id in locks_to_release:
            del self.project_locks[zone_id]
        
        # Remove cursor
        self.user_cursors.pop(user_id, None)
    
    async def _auto_release_lock(self, zone_id: str, user_id: str, timeout: int):
        """Auto-release zone lock after timeout"""
        await asyncio.sleep(timeout)
        
        if (zone_id in self.project_locks and 
            self.project_locks[zone_id] == user_id):
            del self.project_locks[zone_id]
            
            # Notify all users in the project
            await self._broadcast_lock_release(zone_id, user_id)
    
    async def _send_to_user(self, user_id: str, data: Dict):
        """Send message to specific user"""
        if user_id in self.active_sessions:
            websocket = self.active_sessions[user_id]
            try:
                await websocket.send(json.dumps(data))
            except:
                self.active_sessions.pop(user_id, None)
    
    async def _send_project_state(self, user_id: str, project_id: str):
        """Send current project state to user"""
        session = self.Session()
        try:
            # Get project data
            project = session.query(Project).filter_by(id=project_id).first()
            if not project:
                return
            
            # Get active users
            active_users = list(self.active_projects.get(project_id, set()))
            
            # Get user cursors
            cursors = {uid: pos for uid, pos in self.user_cursors.items() 
                      if pos['project_id'] == project_id}
            
            # Get comments
            comments = session.query(Comment).filter_by(
                project_id=project_id, resolved=False
            ).all()
            
            state_data = {
                'type': 'project_state',
                'project': {
                    'id': project.id,
                    'name': project.name,
                    'data': json.loads(project.data) if project.data else {}
                },
                'active_users': active_users,
                'cursors': cursors,
                'comments': [self._comment_to_dict(c) for c in comments],
                'locks': {zone_id: locked_by for zone_id, locked_by in self.project_locks.items()}
            }
            
            await self._send_to_user(user_id, state_data)
            
        finally:
            session.close()
    
    def _comment_to_dict(self, comment: Comment) -> Dict:
        """Convert comment object to dictionary"""
        return {
            'id': comment.id,
            'user_id': comment.user_id,
            'username': comment.username,
            'content': comment.content,
            'x_position': comment.x_position,
            'y_position': comment.y_position,
            'zone_id': comment.zone_id,
            'created_at': comment.created_at.isoformat(),
            'resolved': comment.resolved
        }
    
    async def _save_zone_edit(self, message: CollaborationMessage):
        """Save zone edit with version control"""
        session = self.Session()
        try:
            # Create new version
            version_id = str(uuid.uuid4())
            
            # Get current version number
            latest_version = session.query(Version).filter_by(
                project_id=message.project_id
            ).order_by(Version.version_number.desc()).first()
            
            version_number = (latest_version.version_number + 1) if latest_version else 1
            
            # Calculate checksum
            data_str = json.dumps(message.data, sort_keys=True)
            checksum = hashlib.md5(data_str.encode()).hexdigest()
            
            version = Version(
                id=version_id,
                project_id=message.project_id,
                version_number=version_number,
                created_by=message.user_id,
                description=f"Zone {message.data['zone_id']} edited",
                data=data_str,
                checksum=checksum
            )
            
            session.add(version)
            session.commit()
            
        finally:
            session.close()
    
    async def _save_analysis_results(self, message: CollaborationMessage):
        """Save analysis results to project"""
        session = self.Session()
        try:
            project = session.query(Project).filter_by(id=message.project_id).first()
            if project:
                # Update project data with new analysis results
                project_data = json.loads(project.data) if project.data else {}
                project_data['analysis_results'] = message.data
                project_data['last_analysis'] = message.timestamp.isoformat()
                
                project.data = json.dumps(project_data)
                project.updated_at = message.timestamp
                session.commit()
                
        finally:
            session.close()


class PermissionManager:
    """
    Role-based permission system for collaborative features
    """
    
    def __init__(self):
        self.roles = {
            'admin': {
                'permissions': [
                    'project.create', 'project.delete', 'project.edit',
                    'zone.edit', 'zone.delete', 'analysis.run',
                    'comment.add', 'comment.resolve', 'comment.delete',
                    'user.invite', 'user.remove', 'settings.edit'
                ]
            },
            'architect': {
                'permissions': [
                    'project.edit', 'zone.edit', 'analysis.run',
                    'comment.add', 'comment.resolve', 'user.invite'
                ]
            },
            'designer': {
                'permissions': [
                    'zone.edit', 'analysis.run', 'comment.add'
                ]
            },
            'viewer': {
                'permissions': [
                    'project.view', 'comment.add'
                ]
            }
        }
    
    def has_permission(self, user_role: str, permission: str) -> bool:
        """Check if user role has specific permission"""
        role_permissions = self.roles.get(user_role, {}).get('permissions', [])
        return permission in role_permissions
    
    def get_user_permissions(self, user_role: str) -> List[str]:
        """Get all permissions for a user role"""
        return self.roles.get(user_role, {}).get('permissions', [])


class TeamPlanningInterface:
    """
    Interface for team-based planning features
    """
    
    def __init__(self, collaboration_manager: CollaborationManager):
        self.collaboration_manager = collaboration_manager
        self.permission_manager = PermissionManager()
    
    def create_project(self, project_name: str, created_by: str, 
                      description: str = None) -> str:
        """Create a new collaborative project"""
        project_id = str(uuid.uuid4())
        
        session = self.collaboration_manager.Session()
        try:
            project = Project(
                id=project_id,
                name=project_name,
                description=description,
                created_by=created_by,
                data=json.dumps({
                    'zones': [],
                    'analysis_results': {},
                    'settings': {
                        'units': 'metric',
                        'precision': 2
                    }
                })
            )
            session.add(project)
            session.commit()
            
            return project_id
            
        finally:
            session.close()
    
    def invite_user_to_project(self, project_id: str, user_email: str, 
                              role: str, invited_by: str) -> bool:
        """Invite user to collaborate on project"""
        # In production, this would send an email invitation
        # For now, we'll just log the invitation
        
        invitation_data = {
            'project_id': project_id,
            'user_email': user_email,
            'role': role,
            'invited_by': invited_by,
            'invited_at': datetime.utcnow().isoformat(),
            'status': 'pending'
        }
        
        # Store invitation (would use database in production)
        print(f"Invitation sent: {invitation_data}")
        return True
    
    def get_project_collaborators(self, project_id: str) -> List[Dict]:
        """Get list of project collaborators"""
        # In production, would query user-project relationships
        # For demo, return sample data
        
        return [
            {'user_id': 'user1', 'username': 'John Architect', 'role': 'architect', 'status': 'active'},
            {'user_id': 'user2', 'username': 'Jane Designer', 'role': 'designer', 'status': 'active'},
            {'user_id': 'user3', 'username': 'Bob Viewer', 'role': 'viewer', 'status': 'active'}
        ]
    
    def get_project_activity(self, project_id: str, limit: int = 50) -> List[Dict]:
        """Get recent project activity"""
        session = self.collaboration_manager.Session()
        try:
            # Get recent versions
            versions = session.query(Version).filter_by(
                project_id=project_id
            ).order_by(Version.created_at.desc()).limit(limit).all()
            
            # Get recent comments
            comments = session.query(Comment).filter_by(
                project_id=project_id
            ).order_by(Comment.created_at.desc()).limit(limit).all()
            
            activity = []
            
            for version in versions:
                activity.append({
                    'type': 'edit',
                    'user_id': version.created_by,
                    'description': version.description,
                    'timestamp': version.created_at.isoformat()
                })
            
            for comment in comments:
                activity.append({
                    'type': 'comment',
                    'user_id': comment.user_id,
                    'username': comment.username,
                    'content': comment.content,
                    'timestamp': comment.created_at.isoformat()
                })
            
            # Sort by timestamp
            activity.sort(key=lambda x: x['timestamp'], reverse=True)
            
            return activity[:limit]
            
        finally:
            session.close()
    
    def export_project_report(self, project_id: str) -> Dict[str, Any]:
        """Export comprehensive project report"""
        session = self.collaboration_manager.Session()
        try:
            project = session.query(Project).filter_by(id=project_id).first()
            if not project:
                return {}
            
            # Get project statistics
            versions = session.query(Version).filter_by(project_id=project_id).all()
            comments = session.query(Comment).filter_by(project_id=project_id).all()
            
            report = {
                'project_info': {
                    'id': project.id,
                    'name': project.name,
                    'description': project.description,
                    'created_by': project.created_by,
                    'created_at': project.created_at.isoformat(),
                    'updated_at': project.updated_at.isoformat()
                },
                'statistics': {
                    'total_versions': len(versions),
                    'total_comments': len(comments),
                    'resolved_comments': len([c for c in comments if c.resolved]),
                    'active_collaborators': len(self.get_project_collaborators(project_id))
                },
                'recent_activity': self.get_project_activity(project_id, 20),
                'analysis_results': json.loads(project.data).get('analysis_results', {}) if project.data else {}
            }
            
            return report
            
        finally:
            session.close()


# Client-side JavaScript interface (would be embedded in frontend)
CLIENT_JS_INTERFACE = """
class CollaborationClient {
    constructor(serverUrl, userId, projectId) {
        this.serverUrl = serverUrl;
        this.userId = userId;
        this.projectId = projectId;
        this.socket = null;
        this.callbacks = {};
    }
    
    connect() {
        this.socket = new WebSocket(this.serverUrl);
        
        this.socket.onopen = () => {
            this.send('user_join', {});
        };
        
        this.socket.onmessage = (event) => {
            const message = JSON.parse(event.data);
            this.handleMessage(message);
        };
        
        this.socket.onclose = () => {
            console.log('Collaboration connection closed');
        };
    }
    
    send(messageType, data) {
        if (this.socket && this.socket.readyState === WebSocket.OPEN) {
            this.socket.send(JSON.stringify({
                type: messageType,
                user_id: this.userId,
                project_id: this.projectId,
                data: data
            }));
        }
    }
    
    handleMessage(message) {
        const callback = this.callbacks[message.type];
        if (callback) {
            callback(message);
        }
    }
    
    on(messageType, callback) {
        this.callbacks[messageType] = callback;
    }
    
    // Collaboration methods
    moveCursor(x, y) {
        this.send('cursor_move', { x, y });
    }
    
    selectZone(zoneId) {
        this.send('zone_select', { zone_id: zoneId });
    }
    
    editZone(zoneId, changes) {
        this.send('zone_edit', { zone_id: zoneId, changes });
    }
    
    addComment(content, x, y, zoneId = null) {
        this.send('comment_add', {
            content,
            x_position: x,
            y_position: y,
            zone_id: zoneId,
            username: this.username
        });
    }
    
    resolveComment(commentId) {
        this.send('comment_resolve', { comment_id: commentId });
    }
}
"""