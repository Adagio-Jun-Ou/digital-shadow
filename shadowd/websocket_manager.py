"""
WebSocket Manager for Shadow AI - Real-time bidirectional communication.

Implementation based on patterns from:
- websockets library (5.7k★) - async WebSocket handling
- FastAPI WebSocket support - integration with ASGI
- OpenAI Realtime API - event-based communication
"""

import asyncio
import copy
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

import re

_MAX_MESSAGE_SIZE = 1_048_576  # 1MB max WebSocket message
_MAX_CLIENT_ID_LENGTH = 128
_MAX_TOPIC_LENGTH = 128
_TOPIC_RE = re.compile(r'^[a-zA-Z0-9_\-]{1,128}$')

from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


def _validate_client_id(client_id: str) -> str:
    if not client_id or len(client_id) > _MAX_CLIENT_ID_LENGTH:
        raise ValueError("Invalid client_id length")
    if not re.match(r'^[a-zA-Z0-9_\-\.]+$', client_id):
        raise ValueError("Invalid client_id format")
    return client_id


class WSMessageType(str, Enum):
    CONNECT = "connect"
    DISCONNECT = "disconnect"
    PING = "ping"
    PONG = "pong"
    ERROR = "error"
    
    ASK = "ask"
    ASK_STREAM = "ask_stream"
    ASK_RESPONSE = "ask_response"
    ASK_STREAM_DELTA = "ask_stream_delta"
    ASK_STREAM_DONE = "ask_stream_done"
    
    OBSERVE = "observe"
    OBSERVE_ACK = "observe_ack"
    
    AGENT_STATUS = "agent_status"
    AGENT_CREATED = "agent_created"
    AGENT_MESSAGE = "agent_message"
    
    MEMORY_UPDATE = "memory_update"
    MEMORY_QUERY = "memory_query"
    MEMORY_RESPONSE = "memory_response"
    
    INSIGHT = "insight"
    ACTION = "action"
    ACTION_RESULT = "action_result"
    
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    EVENT = "event"


@dataclass
class WSMessage:
    type: WSMessageType
    data: Dict[str, Any] = field(default_factory=dict)
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    client_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "data": copy.deepcopy(self.data),
            "message_id": self.message_id,
            "timestamp": self.timestamp,
            "client_id": self.client_id,
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WSMessage":
        return cls(
            type=WSMessageType(data["type"]),
            data=data.get("data", {}),
            message_id=data.get("message_id", str(uuid.uuid4())),
            timestamp=data.get("timestamp", datetime.now(timezone.utc).isoformat()),
            client_id=data.get("client_id"),
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> "WSMessage":
        return cls.from_dict(json.loads(json_str))


@dataclass
class WSClient:
    websocket: WebSocket
    client_id: str
    subscriptions: Set[str] = field(default_factory=set)
    connected_at: float = field(default_factory=time.time)
    last_ping: float = field(default_factory=time.time)
    last_pong: float = field(default_factory=time.time)
    is_alive: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    async def send(self, message: WSMessage):
        if self.is_alive:
            try:
                await self.websocket.send_text(message.to_json())
            except Exception as e:
                logger.error("Error sending to client %s: %s", self.client_id, type(e).__name__)
                self.is_alive = False
    
    async def receive(self) -> Optional[WSMessage]:
        try:
            data = await self.websocket.receive_text()
            if len(data) > _MAX_MESSAGE_SIZE:
                logger.warning("Message too large from %s: %d bytes", self.client_id, len(data))
                return None
            return WSMessage.from_json(data)
        except WebSocketDisconnect:
            self.is_alive = False
            return None
        except Exception as e:
            logger.error("Error receiving from client %s: %s", self.client_id, type(e).__name__)
            self.is_alive = False
            return None


class ConnectionManager:
    def __init__(self):
        self.clients: Dict[str, WSClient] = {}
        self.client_handlers: Dict[str, Dict[WSMessageType, List[Callable]]] = {}
        self.global_handlers: Dict[WSMessageType, List[Callable]] = {}
        self.broadcast_queues: Dict[str, asyncio.Queue] = {}
        self._lock = asyncio.Lock()
        self._client_last_seen: Dict[str, float] = {}
        
    async def connect(self, websocket: WebSocket, client_id: Optional[str] = None) -> WSClient:
        await websocket.accept()
        client_id = client_id or str(uuid.uuid4())
        try:
            client_id = _validate_client_id(client_id)
        except ValueError:
            client_id = str(uuid.uuid4())
        client = WSClient(websocket=websocket, client_id=client_id)
        
        async with self._lock:
            if client_id in self.clients:
                old_client = self.clients[client_id]
                logger.warning("Client ID takeover detected: %s. Disconnecting old client.", client_id)
                try:
                    await old_client.websocket.close()
                except (RuntimeError, ValueError, KeyError, TypeError):
                    pass
                old_client.is_alive = False
            self.clients[client_id] = client
            self._client_last_seen[client_id] = time.time()
        
        await self._send_to_client(client, WSMessage(
            type=WSMessageType.CONNECT,
            data={"client_id": client_id, "message": "Connected to Shadow AI"}
        ))
        
        logger.info("Client %s connected. Total clients: %d", client_id, len(self.clients))
        
        return client
    
    async def disconnect(self, client_id: str):
        async with self._lock:
            if client_id in self.clients:
                client = self.clients.pop(client_id)
                client.is_alive = False
                
                for topic in list(client.subscriptions):
                    if topic in self.broadcast_queues:
                        try:
                            self.broadcast_queues[topic].task_done()
                        except ValueError:
                            pass
            self._client_last_seen.pop(client_id, None)
        
        logger.info("Client %s disconnected. Total clients: %d", client_id, len(self.clients))
    
    async def _send_to_client(self, client: WSClient, message: WSMessage):
        message.client_id = client.client_id
        await client.send(message)
    
    async def send_to_client(self, client_id: str, message: WSMessage):
        async with self._lock:
            client = self.clients.get(client_id)
        
        if client and client.is_alive:
            await self._send_to_client(client, message)
    
    async def broadcast(self, message: WSMessage, exclude: Optional[Set[str]] = None):
        exclude = exclude or set()
        
        async with self._lock:
            clients_to_broadcast = [
                client for client in self.clients.values()
                if client.client_id not in exclude and client.is_alive
            ]
        
        for client in clients_to_broadcast:
            await self._send_to_client(client, message)
    
    async def broadcast_to_topic(self, topic: str, message: WSMessage, exclude: Optional[Set[str]] = None):
        exclude = exclude or set()
        
        async with self._lock:
            subscribers = [
                client for client in self.clients.values()
                if topic in client.subscriptions and client.client_id not in exclude and client.is_alive
            ]
        
        for client in subscribers:
            await self._send_to_client(client, message)
    
    def on_message(self, message_type: WSMessageType, handler: Callable = None, client_id: Optional[str] = None):
        def decorator(func: Callable):
            if client_id:
                if client_id not in self.client_handlers:
                    self.client_handlers[client_id] = {}
                if message_type not in self.client_handlers[client_id]:
                    self.client_handlers[client_id][message_type] = []
                self.client_handlers[client_id][message_type].append(func)
            else:
                if message_type not in self.global_handlers:
                    self.global_handlers[message_type] = []
                self.global_handlers[message_type].append(func)
            return func
        
        if handler:
            return decorator(handler)
        return decorator
    
    async def handle_message(self, client: WSClient, message: WSMessage):
        handlers = []
        
        if client.client_id in self.client_handlers:
            if message.type in self.client_handlers[client.client_id]:
                handlers.extend(self.client_handlers[client.client_id][message.type])
        
        if message.type in self.global_handlers:
            handlers.extend(self.global_handlers[message.type])
        
        for handler in handlers:
            try:
                result = handler(client, message)
                if asyncio.iscoroutine(result):
                    result = await result
                if isinstance(result, WSMessage):
                    await self._send_to_client(client, result)
            except Exception as e:
                logger.error("Handler error for %s: %s", message.type, type(e).__name__)
                await self._send_to_client(client, WSMessage(
                    type=WSMessageType.ERROR,
                    data={"error": type(e).__name__, "message_id": message.message_id}
                ))
    
    async def handle_client_loop(self, client: WSClient):
        while client.is_alive:
            message = await client.receive()
            if message is None:
                break
            
            self._client_last_seen[client.client_id] = time.time()
            
            if message.type == WSMessageType.PING:
                client.last_ping = time.time()
                await self._send_to_client(client, WSMessage(
                    type=WSMessageType.PONG,
                    data={"timestamp": datetime.now(timezone.utc).isoformat()}
                ))
            elif message.type == WSMessageType.PONG:
                client.last_pong = time.time()
            elif message.type == WSMessageType.SUBSCRIBE:
                topic = message.data.get("topic")
                if topic and _TOPIC_RE.match(topic):
                    client.subscriptions.add(topic)
                    await self._send_to_client(client, WSMessage(
                        type=WSMessageType.SUBSCRIBE,
                        data={"topic": topic, "status": "subscribed"}
                    ))
            elif message.type == WSMessageType.UNSUBSCRIBE:
                topic = message.data.get("topic")
                if topic and topic in client.subscriptions:
                    client.subscriptions.discard(topic)
                    await self._send_to_client(client, WSMessage(
                        type=WSMessageType.UNSUBSCRIBE,
                        data={"topic": topic, "status": "unsubscribed"}
                    ))
            else:
                await self.handle_message(client, message)
        
        await self.disconnect(client.client_id)
    
    def get_client_count(self) -> int:
        return len(self.clients)
    
    def get_connected_clients(self) -> List[str]:
        return list(cid for cid, client in self.clients.items() if client.is_alive)
    
    def get_topic_subscribers(self, topic: str) -> List[str]:
        return list([
            client.client_id for client in self.clients.values()
            if topic in client.subscriptions and client.is_alive
        ])

    async def check_client_health(self) -> Dict[str, bool]:
        """Send a ping to all connected clients and track which ones respond."""
        responses: Dict[str, bool] = {}
        async with self._lock:
            clients_snapshot = dict(self.clients)

        for client_id, client in clients_snapshot.items():
            if not client.is_alive:
                responses[client_id] = False
                continue
            try:
                await self._send_to_client(client, WSMessage(
                    type=WSMessageType.PING,
                    data={"timestamp": datetime.now(timezone.utc).isoformat()}
                ))
                responses[client_id] = True
            except (RuntimeError, ValueError, KeyError, TypeError):
                responses[client_id] = False
        return responses

    def get_client_health(self) -> Dict[str, Dict]:
        """Return health info for each client: connected, last_seen, subscriptions."""
        result = {}
        for client_id, client in self.clients.items():
            result[client_id] = {
                "connected": client.is_alive,
                "last_seen": self._client_last_seen.get(client_id, client.connected_at),
                "subscriptions": list(client.subscriptions),
            }
        return copy.deepcopy(result)


class StreamingResponseHandler:
    _MAX_ACTIVE_STREAMS = 500

    def __init__(self, manager: ConnectionManager):
        self.manager = manager
        self.active_streams: Dict[str, asyncio.Task] = {}
    
    async def stream_response(
        self,
        client_id: str,
        message_id: str,
        response_generator,
    ):
        async def stream_task():
            try:
                full_response = ""
                async for delta in response_generator:
                    full_response += delta
                    
                    await self.manager.send_to_client(client_id, WSMessage(
                        type=WSMessageType.ASK_STREAM_DELTA,
                        data={
                            "message_id": message_id,
                            "delta": delta,
                            "accumulated": full_response,
                        }
                    ))
                
                await self.manager.send_to_client(client_id, WSMessage(
                    type=WSMessageType.ASK_STREAM_DONE,
                    data={
                        "message_id": message_id,
                        "full_response": full_response,
                    }
                ))
            except Exception as e:
                logger.error("Streaming error for %s: %s", client_id, type(e).__name__)
                await self.manager.send_to_client(client_id, WSMessage(
                    type=WSMessageType.ERROR,
                    data={
                        "message_id": message_id,
                        "error": "Stream processing error",
                    }
                ))
            finally:
                if message_id in self.active_streams:
                    del self.active_streams[message_id]
        
        if len(self.active_streams) >= self._MAX_ACTIVE_STREAMS:
            oldest_id = next(iter(self.active_streams))
            oldest_task = self.active_streams.pop(oldest_id)
            oldest_task.cancel()
        
        self.active_streams[message_id] = asyncio.create_task(stream_task())
    
    async def cancel_stream(self, message_id: str):
        if message_id in self.active_streams:
            self.active_streams[message_id].cancel()
            del self.active_streams[message_id]


class HeartbeatManager:
    def __init__(
        self,
        manager: ConnectionManager,
        ping_interval: float = 30.0,
        pong_timeout: float = 10.0,
    ):
        self.manager = manager
        self.ping_interval = ping_interval
        self.pong_timeout = pong_timeout
        self._running = False
        self._task: Optional[asyncio.Task] = None
    
    async def start(self):
        self._running = True
        self._task = asyncio.create_task(self._heartbeat_loop())
        logger.info("Heartbeat manager started")
    
    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Heartbeat manager stopped")
    
    async def _heartbeat_loop(self):
        while self._running:
            await asyncio.sleep(self.ping_interval)
            
            now = time.time()
            dead_clients = []
            
            async with self.manager._lock:
                for client in self.manager.clients.values():
                    if not client.is_alive:
                        continue
                    
                    if now - client.last_ping > self.pong_timeout:
                        if now - client.last_pong > self.pong_timeout:
                            dead_clients.append(client.client_id)
                            continue
                    
                    try:
                        await self.manager._send_to_client(client, WSMessage(
                            type=WSMessageType.PING,
                            data={"timestamp": datetime.now(timezone.utc).isoformat()}
                        ))
                        client.last_ping = now
                    except (RuntimeError, ValueError, KeyError, TypeError):
                        dead_clients.append(client.client_id)
            
            for client_id in dead_clients:
                await self.manager.disconnect(client_id)


class WebSocketServer:
    def __init__(
        self,
        manager: Optional[ConnectionManager] = None,
        enable_heartbeat: bool = True,
        ping_interval: float = 30.0,
        pong_timeout: float = 10.0,
    ):
        self.manager = manager or ConnectionManager()
        self.streaming = StreamingResponseHandler(self.manager)
        self.heartbeat = HeartbeatManager(
            self.manager, ping_interval, pong_timeout
        ) if enable_heartbeat else None
        self._setup_default_handlers()
    
    def _setup_default_handlers(self):
        @self.manager.on_message(WSMessageType.ASK)
        async def handle_ask(client: WSClient, message: WSMessage) -> WSMessage:
            shadow_ai = client.metadata.get("shadow_ai")
            if not shadow_ai:
                return WSMessage(
                    type=WSMessageType.ERROR,
                    data={"error": "Shadow AI not initialized", "message_id": message.message_id}
                )
            
            question = message.data.get("question", "")
            context = message.data.get("context", {})
            
            result = await shadow_ai.ask(question, context)
            
            return WSMessage(
                type=WSMessageType.ASK_RESPONSE,
                data={
                    "message_id": message.message_id,
                    "response": result.get("unified_response", ""),
                    "ability_used": result.get("ability_used"),
                    "insight": result.get("insight"),
                }
            )
        
        @self.manager.on_message(WSMessageType.OBSERVE)
        async def handle_observe(client: WSClient, message: WSMessage) -> WSMessage:
            shadow_ai = client.metadata.get("shadow_ai")
            if not shadow_ai:
                return WSMessage(
                    type=WSMessageType.ERROR,
                    data={"error": "Shadow AI not initialized", "message_id": message.message_id}
                )
            
            observation = message.data.get("observation", {})
            
            if hasattr(shadow_ai, 'observe'):
                await shadow_ai.observe(observation)
            
            return WSMessage(
                type=WSMessageType.OBSERVE_ACK,
                data={"message_id": message.message_id, "status": "observed"}
            )
    
    async def handle_connection(self, websocket: WebSocket, client_id: Optional[str] = None):
        client = await self.manager.connect(websocket, client_id)
        
        try:
            await self.manager.handle_client_loop(client)
        except Exception as e:
            logger.error("Connection error for %s: %s", client.client_id, type(e).__name__)
        finally:
            await self.manager.disconnect(client.client_id)
    
    def get_router(self):
        from fastapi import APIRouter
        
        router = APIRouter()
        
        @router.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await self.handle_connection(websocket)
        
        @router.websocket("/ws/{client_id}")
        async def websocket_endpoint_with_id(websocket: WebSocket, client_id: str):
            await self.handle_connection(websocket, client_id)
        
        return router
    
    async def start_heartbeat(self):
        if self.heartbeat:
            await self.heartbeat.start()
    
    async def stop_heartbeat(self):
        if self.heartbeat:
            await self.heartbeat.stop()
    
    async def cleanup(self):
        """Clean up all resources: cancel streams, stop heartbeat, disconnect clients."""
        for message_id in list(self.streaming.active_streams.keys()):
            await self.streaming.cancel_stream(message_id)
        
        await self.stop_heartbeat()
        
        for client_id in list(self.manager.clients.keys()):
            await self.manager.disconnect(client_id)