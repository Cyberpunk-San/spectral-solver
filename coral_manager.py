# coral_manager.py
import asyncio
import logging
import redis.asyncio as redis
import json
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Callable
from enum import Enum

# Coral framework imports
from coral_s import (
    CoralBaseAgent, AgentCapability, MessageType, CoralMessage,
    TaskPriority, AgentStatus, CoralThread
)


class CoralManager:
    """
    Manager for Coral multi-agent system with Redis-based messaging.
    """
    
    def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379, 
                 redis_db: int = 0, channel_prefix: str = "coral"):
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_db = redis_db
        self.channel_prefix = channel_prefix
        self.redis_client: Optional[redis.Redis] = None
        self.pubsub: Optional[redis.client.PubSub] = None
        self.agents: Dict[str, CoralBaseAgent] = {}
        self.threads: Dict[str, CoralThread] = {}
        self.message_handlers: Dict[str, Callable] = {}
        self.is_running = False
        self.logger = logging.getLogger('coral_manager')
        
    async def initialize(self):
        """Initialize Redis connection and start listening."""
        try:
            self.redis_client = redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                db=self.redis_db,
                decode_responses=True
            )
            
            # Test connection
            await self.redis_client.ping()
            self.logger.info("✅ Connected to Redis server successfully")
            
            # Initialize pubsub
            self.pubsub = self.redis_client.pubsub()
            await self.pubsub.subscribe(f"{self.channel_prefix}:messages")
            
            # Register default message handlers
            self._register_default_handlers()
            
            self.is_running = True
            self.logger.info("Coral Manager initialized successfully")
            
            # Start message processing task
            asyncio.create_task(self.process_incoming_messages())
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Coral Manager: {e}")
            raise
    
    def _register_default_handlers(self):
        """Register default message handlers."""
        self.message_handlers = {
            MessageType.AGENT_REGISTRATION.value: self._handle_agent_registration,
            MessageType.AGENT_STATUS_UPDATE.value: self._handle_agent_status_update,
            MessageType.TASK_RESPONSE.value: self._handle_task_response,
            MessageType.ANALYZE_RESULT.value: self._handle_analyze_result,
            MessageType.INVESTIGATE_RESULT.value: self._handle_investigate_result,
            MessageType.REPORT_RESULT.value: self._handle_report_result,
            MessageType.HEARTBEAT.value: self._handle_heartbeat,
            MessageType.ERROR.value: self._handle_error,
        }
    
    async def register_agent(self, agent: CoralBaseAgent):
        """Register an agent with the manager."""
        self.agents[agent.agent_id] = agent
        self.logger.info(f"Registered agent: {agent.agent_id} ({agent.name})")
        
        # Notify other agents about new registration
        registration_msg = CoralMessage(
            message_id=self._generate_message_id(),
            thread_id=str(uuid.uuid4()),
            sender_id="coral_manager",
            recipient_id="broadcast",
            message_type=MessageType.AGENT_REGISTRATION,
            payload={
                "agent_id": agent.agent_id,
                "agent_name": agent.name,
                "agent_type": agent.agent_type,
                "capabilities": [cap.to_dict() for cap in agent.capabilities],
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            timestamp=datetime.now(timezone.utc),
            priority=TaskPriority.NORMAL
        )
        
        await self.send_message(registration_msg)
    
    async def send_message(self, message: CoralMessage):
        """Send a message via Redis pub/sub."""
        if not self.redis_client:
            raise RuntimeError("Redis client not initialized")
        
        try:
            message_dict = message.to_dict()
            
            channel = f"{self.channel_prefix}:messages"
            await self.redis_client.publish(channel, json.dumps(message_dict))
            self.logger.debug(f"Sent message: {message.message_type.value} from {message.sender_id} to {message.recipient_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to send message: {e}")
            raise
    
    async def broadcast_message(self, message: CoralMessage):
        """Broadcast a message to all agents."""
        message.recipient_id = "broadcast"
        await self.send_message(message)
    
    async def process_incoming_messages(self):
        """Process incoming messages from Redis pub/sub."""
        self.logger.info("Starting message processing loop...")
        
        while self.is_running:
            try:
                message = await self.pubsub.get_message(
                    ignore_subscribe_messages=True, 
                    timeout=1.0
                )
                
                if message and message['type'] == 'message':
                    message_data = json.loads(message['data'])
                    await self._route_message(message_data)
                    
            except Exception as e:
                self.logger.error(f"Error processing message: {e}")
                await asyncio.sleep(0.1)  # Prevent tight loop on errors
    
    async def _route_message(self, message_data: Dict[str, Any]):
        """Route incoming message to appropriate handler."""
        try:
            # Convert back to CoralMessage object
            coral_message = CoralMessage.from_dict(message_data)
            
            # Handle system-level messages first
            handler = self.message_handlers.get(coral_message.message_type.value)
            if handler:
                await handler(coral_message)
            
            # Route to specific agent if recipient is specified
            recipient_id = coral_message.recipient_id
            if recipient_id != "broadcast":
                if recipient_id in self.agents:
                    agent = self.agents[recipient_id]
                    response = await agent.receive_message(coral_message)
                    if response:
                        await self.send_message(response)
                elif recipient_id != "coral_manager":
                    self.logger.warning(f"Unknown recipient: {recipient_id}")
            
            # Update thread if message has thread_id
            if coral_message.thread_id and coral_message.thread_id in self.threads:
                thread = self.threads[coral_message.thread_id]
                thread.messages.append(coral_message)
                
        except Exception as e:
            self.logger.error(f"Error routing message: {e}")
    
    async def _handle_agent_registration(self, message: CoralMessage):
        """Handle agent registration messages."""
        agent_id = message.payload.get('agent_id')
        self.logger.info(f"Agent registration processed: {agent_id}")
    
    async def _handle_agent_status_update(self, message: CoralMessage):
        """Handle agent status updates."""
        agent_id = message.payload.get('agent_id')
        status = message.payload.get('status')
        self.logger.debug(f"Agent {agent_id} status: {status}")
    
    async def _handle_task_response(self, message: CoralMessage):
        """Handle task response messages."""
        task_id = message.correlation_id
        result = message.payload
        self.logger.info(f"Task {task_id} completed by {message.sender_id}")
    
    async def _handle_analyze_result(self, message: CoralMessage):
        """Handle analysis result messages."""
        analysis_type = message.payload.get('analysis_type', 'unknown')
        anomalies_count = len(message.payload.get('anomalies', []))
        self.logger.info(f"Analysis completed: {analysis_type}, found {anomalies_count} anomalies")
        
        # Automatically forward to investigator if anomalies found
        if anomalies_count > 0:
            investigator_msg = CoralMessage(
                message_id=self._generate_message_id(),
                thread_id=message.thread_id,
                sender_id="coral_manager",
                recipient_id="investigator_001",  
                message_type=MessageType.INVESTIGATE_REQUEST,
                payload=message.payload,
                timestamp=datetime.now(timezone.utc),
                priority=message.priority,
                correlation_id=message.message_id
            )
            await self.send_message(investigator_msg)
    
    async def _handle_investigate_result(self, message: CoralMessage):
        """Handle investigation result messages."""
        diagnosis = message.payload.get('diagnosis', 'unknown')
        confidence = message.payload.get('confidence', 0)
        self.logger.info(f"Investigation completed: {diagnosis} (confidence: {confidence:.2f})")
        
        # Automatically forward to reporter
        reporter_msg = CoralMessage(
            message_id=self._generate_message_id(),
            thread_id=message.thread_id,
            sender_id="coral_manager",
            recipient_id="reporter_001",  
            message_type=MessageType.REPORT_REQUEST,
            payload=message.payload,
            timestamp=datetime.now(timezone.utc),
            priority=message.priority,
            correlation_id=message.message_id
        )
        await self.send_message(reporter_msg)
    
    async def _handle_report_result(self, message: CoralMessage):
        """Handle report result messages."""
        report_count = len(message.payload.get('reports', {}))
        alert_level = message.payload.get('alert_level', 'normal')
        self.logger.info(f"Report generation completed: {report_count} reports, alert level: {alert_level}")
    
    async def _handle_heartbeat(self, message: CoralMessage):
        """Handle heartbeat messages."""
        self.logger.debug(f"Heartbeat from {message.sender_id}")
    
    async def _handle_error(self, message: CoralMessage):
        """Handle error messages."""
        error = message.payload.get('error', 'Unknown error')
        self.logger.error(f"Error from {message.sender_id}: {error}")
    
    async def create_thread(self, created_by: str, metadata: Dict[str, Any] = None) -> str:
        """Create a new Coral thread."""
        thread_id = str(uuid.uuid4())
        thread = CoralThread(
            thread_id=thread_id,
            created_at=datetime.now(timezone.utc),
            created_by=created_by,
            metadata=metadata or {}
        )
        self.threads[thread_id] = thread
        self.logger.info(f"Created thread: {thread_id}")
        return thread_id
    
    async def submit_task(self, task_data: Dict[str, Any], 
                         target_agent_id: str = None,
                         thread_id: str = None,
                         priority: TaskPriority = TaskPriority.NORMAL) -> str:
        """Submit a task to a specific agent or broadcast to capable agents."""
        if not thread_id:
            thread_id = await self.create_thread("coral_manager")
        
        task_id = str(uuid.uuid4())
        
        if target_agent_id:
            # Send to specific agent
            message = CoralMessage(
                message_id=self._generate_message_id(),
                thread_id=thread_id,
                sender_id="coral_manager",
                recipient_id=target_agent_id,
                message_type=MessageType.TASK_REQUEST,
                payload=task_data,
                timestamp=datetime.now(timezone.utc),
                priority=priority,
                correlation_id=task_id
            )
            await self.send_message(message)
        else:
            # Broadcast to all agents
            message = CoralMessage(
                message_id=self._generate_message_id(),
                thread_id=thread_id,
                sender_id="coral_manager",
                recipient_id="broadcast",
                message_type=MessageType.TASK_REQUEST,
                payload=task_data,
                timestamp=datetime.now(timezone.utc),
                priority=priority,
                correlation_id=task_id
            )
            await self.broadcast_message(message)
        
        self.logger.info(f"Submitted task {task_id} to {target_agent_id or 'broadcast'}")
        return task_id
    
    async def request_analysis(self, file_path: str = None, file_content: str = None, 
                              file_type: str = None, analysis_type: str = "comprehensive",
                              target_agent: str = "analyzer_001") -> str:
        """Convenience method to request file analysis."""
        task_data = {
            "file_path": file_path,
            "file_content": file_content,
            "file_type": file_type,
            "analysis_type": analysis_type,
            "metadata": {
                "requested_by": "coral_manager",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        }
        
        return await self.submit_task(task_data, target_agent, priority=TaskPriority.NORMAL)
    
    def get_agent_status(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific agent."""
        agent = self.agents.get(agent_id)
        if agent:
            return {
                "agent_id": agent.agent_id,
                "agent_name": agent.name,
                "status": agent.status.value,
                "current_load": agent.current_load,
                "capabilities": [cap.name for cap in agent.capabilities]
            }
        return None
    
    def list_agents(self) -> List[Dict[str, Any]]:
        """List all registered agents."""
        return [self.get_agent_status(agent_id) for agent_id in self.agents.keys()]
    
    def get_thread(self, thread_id: str) -> Optional[CoralThread]:
        """Get a thread by ID."""
        return self.threads.get(thread_id)
    
    def list_threads(self) -> List[Dict[str, Any]]:
        """List all active threads."""
        return [
            {
                "thread_id": thread.thread_id,
                "created_at": thread.created_at.isoformat(),
                "created_by": thread.created_by,
                "status": thread.status,
                "message_count": len(thread.messages)
            }
            for thread in self.threads.values()
        ]
    
    def _generate_message_id(self) -> str:
        """Generate a unique message ID."""
        return f"msg_mgr_{uuid.uuid4().hex[:8]}"
    
    async def shutdown(self):
        """Shutdown the Coral Manager gracefully."""
        self.is_running = False
        self.logger.info("Shutting down Coral Manager...")
        
        if self.pubsub:
            await self.pubsub.unsubscribe(f"{self.channel_prefix}:messages")
            await self.pubsub.close()
        
        if self.redis_client:
            await self.redis_client.close()
        
        # Notify agents of shutdown
        shutdown_msg = CoralMessage(
            message_id=self._generate_message_id(),
            thread_id=str(uuid.uuid4()),
            sender_id="coral_manager",
            recipient_id="broadcast",
            message_type=MessageType.SYSTEM_SHUTDOWN,
            payload={"message": "System shutdown initiated"},
            timestamp=datetime.now(timezone.utc),
            priority=TaskPriority.HIGH
        )
        
        try:
            await self.broadcast_message(shutdown_msg)
        except Exception as e:
            self.logger.warning(f"Error during shutdown broadcast: {e}")
        
        self.logger.info("Coral Manager shutdown complete")


# Singleton instance for easy access
_coral_manager_instance: Optional[CoralManager] = None

async def get_coral_manager(redis_host: str = 'localhost', 
                          redis_port: int = 6379, 
                          redis_db: int = 0) -> CoralManager:
    """Get or create the singleton Coral Manager instance."""
    global _coral_manager_instance
    if _coral_manager_instance is None:
        _coral_manager_instance = CoralManager(redis_host, redis_port, redis_db)
        await _coral_manager_instance.initialize()
    return _coral_manager_instance

async def create_coral_manager(redis_host: str = 'localhost', 
                                redis_port: int = 6379, 
                                redis_db: int = 0) -> CoralManager:
    """Create a new Coral Manager instance (non-singleton)."""
    manager = CoralManager(redis_host, redis_port, redis_db)
    await manager.initialize()
    return manager