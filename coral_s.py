# coral_s.py
"""
Coral Protocol Core Implementation for Spectral Solver
Production-grade multi-agent coordination system
"""
import asyncio
import json
import logging
import uuid
import hashlib
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Callable, Union
from enum import Enum
from dataclasses import dataclass, asdict, field
from abc import ABC, abstractmethod

import aiohttp
import redis.asyncio as redis
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
import base64


class MessageType(Enum):
    """Coral Protocol Message Types for Spectral Solver"""
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    STATUS_UPDATE = "status_update"
    CAPABILITY_ANNOUNCEMENT = "capability_announcement"
    THREAD_CREATE = "thread_create"
    THREAD_UPDATE = "thread_update"
    THREAD_CLOSE = "thread_close"
    ANALYZE_REQUEST = "analyze_request"
    ANALYZE_RESULT = "analyze_result"
    INVESTIGATE_REQUEST = "investigate_request"
    INVESTIGATE_RESULT = "investigate_result"
    REPORT_REQUEST = "report_request"
    REPORT_RESULT = "report_result"
    EMERGENCY_ALERT = "emergency_alert"
    HEARTBEAT = "heartbeat"
    FILE_ANALYSIS_REQUEST = "file_analysis_request"
    DATA_ANALYSIS_REQUEST = "data_analysis_request"
    AGENT_REGISTRATION = "agent_registration"
    AGENT_STATUS_UPDATE = "agent_status_update"
    ERROR = "error"
    SYSTEM_SHUTDOWN = "system_shutdown"


class AgentStatus(Enum):
    """Agent operational status"""
    ONLINE = "online"
    OFFLINE = "offline"
    BUSY = "busy"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class TaskPriority(Enum):
    """Task priority levels"""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    NORMAL = 5  # Added for compatibility


@dataclass
class CoralMessage:
    """Coral Protocol Message Structure"""
    message_id: str
    thread_id: str
    sender_id: str
    recipient_id: str
    message_type: MessageType
    payload: Dict[str, Any]
    timestamp: datetime
    priority: TaskPriority = TaskPriority.NORMAL
    correlation_id: Optional[str] = None
    signature: Optional[str] = None
    encrypted: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for serialization"""
        data = asdict(self)
        data['message_type'] = self.message_type.value
        data['priority'] = self.priority.value
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CoralMessage':
        """Create message from dictionary"""
        data['message_type'] = MessageType(data['message_type'])
        data['priority'] = TaskPriority(data['priority'])
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class AgentCapability:
    """Agent capability definition"""
    name: str
    version: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AgentProfile:
    """Agent profile and identity"""
    agent_id: str
    name: str
    agent_type: str
    status: AgentStatus
    capabilities: List[AgentCapability]
    endpoint: str
    public_key: str
    current_load: float = 0.0
    trust_score: float = 1.0
    last_heartbeat: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['status'] = self.status.value
        data['last_heartbeat'] = self.last_heartbeat.isoformat()
        data['capabilities'] = [cap.to_dict() for cap in self.capabilities]
        return data


@dataclass
class CoralThread:
    """Coral Thread for tracking analysis lifecycle"""
    thread_id: str
    created_at: datetime
    created_by: str
    status: str = "active"  # active, completed, failed, cancelled
    metadata: Dict[str, Any] = field(default_factory=dict)
    messages: List[CoralMessage] = field(default_factory=list)
    artifacts: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['messages'] = [msg.to_dict() for msg in self.messages]
        return data


class CoralCrypto:
    """Cryptographic utilities for Coral Protocol"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def generate_keypair(self) -> tuple[str, str]:
        """Generate RSA keypair for agent"""
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        
        public_key = private_key.public_key()
        
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        ).decode('utf-8')
        
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ).decode('utf-8')
        
        return private_pem, public_pem
    
    def sign_message(self, message_data: str, private_key_pem: str) -> str:
        """Sign message data with private key"""
        try:
            private_key = serialization.load_pem_private_key(
                private_key_pem.encode('utf-8'),
                password=None
            )
            
            signature = private_key.sign(
                message_data.encode('utf-8'),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            
            return base64.b64encode(signature).decode('utf-8')
            
        except Exception as e:
            self.logger.error(f"Message signing failed: {str(e)}")
            raise
    
    def verify_signature(self, message_data: str, signature: str, public_key_pem: str) -> bool:
        """Verify message signature"""
        try:
            public_key = serialization.load_pem_public_key(public_key_pem.encode('utf-8'))
            signature_bytes = base64.b64decode(signature.encode('utf-8'))
            
            public_key.verify(
                signature_bytes,
                message_data.encode('utf-8'),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except Exception:
            return False


class CoralBaseAgent(ABC):
    """Base class for all Coral agents in Spectral Solver"""
    
    def __init__(self, agent_id: str, name: str, agent_type: str, capabilities: List[AgentCapability]):
        self.agent_id = agent_id
        self.name = name
        self.agent_type = agent_type
        self.capabilities = capabilities
        self.status = AgentStatus.ONLINE
        self.current_load = 0.0
        self.crypto = CoralCrypto()
        self.private_key, self.public_key = self.crypto.generate_keypair()
        self.endpoint = f"http://localhost:8000/agent/{agent_id}"
        
        self.logger = logging.getLogger(f"coral.agent.{agent_id}")
        self.message_handlers = self._register_message_handlers()
    
    def _register_message_handlers(self) -> Dict[MessageType, Callable]:
        """Register message handlers - override in subclasses"""
        return {
            MessageType.HEARTBEAT: self._handle_heartbeat,
            MessageType.STATUS_UPDATE: self._handle_status_update,
            MessageType.TASK_REQUEST: self._handle_task_request,
        }
    
    def get_profile(self) -> AgentProfile:
        """Get agent profile"""
        return AgentProfile(
            agent_id=self.agent_id,
            name=self.name,
            agent_type=self.agent_type,
            status=self.status,
            capabilities=self.capabilities,
            endpoint=self.endpoint,
            public_key=self.public_key,
            current_load=self.current_load
        )
    
    async def receive_message(self, message: CoralMessage) -> Optional[CoralMessage]:
        """Handle incoming Coral message - must be implemented by subclasses"""
        try:
            handler = self.message_handlers.get(message.message_type)
            if handler:
                return await handler(message)
            else:
                self.logger.warning(f"No handler for message type: {message.message_type}")
                return await self._handle_unknown_message(message)
        except Exception as e:
            self.logger.error(f"Message handling failed: {str(e)}")
            return await self._handle_error(message, str(e))
    
    async def _handle_heartbeat(self, message: CoralMessage) -> Optional[CoralMessage]:
        """Handle heartbeat message"""
        self.logger.debug(f"Heartbeat received from {message.sender_id}")
        return None
    
    async def _handle_status_update(self, message: CoralMessage) -> Optional[CoralMessage]:
        """Handle status update message"""
        status_data = message.payload
        self.logger.info(f"Status update: {status_data}")
        return None
    
    async def _handle_task_request(self, message: CoralMessage) -> Optional[CoralMessage]:
        """Handle task request message"""
        try:
            result = await self.process_task(message.payload)
            response = CoralMessage(
                message_id=self._generate_message_id(),
                thread_id=message.thread_id,
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                message_type=MessageType.TASK_RESPONSE,
                payload=result,
                timestamp=datetime.now(timezone.utc),
                priority=message.priority,
                correlation_id=message.message_id
            )
            return response
        except Exception as e:
            self.logger.error(f"Task processing failed: {str(e)}")
            error_response = CoralMessage(
                message_id=self._generate_message_id(),
                thread_id=message.thread_id,
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                message_type=MessageType.ERROR,
                payload={"error": str(e), "status": "failed"},
                timestamp=datetime.now(timezone.utc),
                priority=TaskPriority.HIGH,
                correlation_id=message.message_id
            )
            return error_response
    
    async def _handle_unknown_message(self, message: CoralMessage) -> Optional[CoralMessage]:
        """Handle unknown message types"""
        self.logger.warning(f"Unknown message type received: {message.message_type}")
        return None
    
    async def _handle_error(self, message: CoralMessage, error: str) -> Optional[CoralMessage]:
        """Handle errors during message processing"""
        error_response = CoralMessage(
            message_id=self._generate_message_id(),
            thread_id=message.thread_id,
            sender_id=self.agent_id,
            recipient_id=message.sender_id,
            message_type=MessageType.ERROR,
            payload={"error": error, "original_message": message.to_dict()},
            timestamp=datetime.now(timezone.utc),
            priority=TaskPriority.HIGH,
            correlation_id=message.message_id
        )
        return error_response
    
    @abstractmethod
    async def process_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process task - to be implemented by specific agents"""
        pass
    
    def _generate_message_id(self) -> str:
        """Generate unique message ID"""
        return f"msg_{self.agent_id}_{uuid.uuid4().hex[:8]}"
    
    async def send_message(self, manager, message: CoralMessage):
        """Send message through manager"""
        if hasattr(manager, 'send_message'):
            await manager.send_message(message)
        else:
            self.logger.error("Manager does not have send_message method")
    
    async def update_status(self, status: AgentStatus, load: float = None):
        """Update agent status"""
        self.status = status
        if load is not None:
            self.current_load = load
        self.logger.info(f"Status updated: {status.value}, Load: {self.current_load}")