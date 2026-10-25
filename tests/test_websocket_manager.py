"""
Tests for WebSocket Manager - Real-time communication.

Hour 11: WebSocket Support Implementation
"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from shadowd.websocket_manager import (
    WSMessageType,
    WSMessage,
    WSClient,
    ConnectionManager,
    StreamingResponseHandler,
    HeartbeatManager,
    WebSocketServer,
)
from fastapi import WebSocketDisconnect


class TestWSMessage:
    def test_message_creation(self):
        msg = WSMessage(type=WSMessageType.ASK, data={"question": "test"})
        assert msg.type == WSMessageType.ASK
        assert msg.data == {"question": "test"}
        assert msg.message_id is not None
        assert msg.timestamp is not None
    
    def test_message_to_dict(self):
        msg = WSMessage(
            type=WSMessageType.PING,
            data={"timestamp": "2024-01-01"},
            message_id="test-id-123",
        )
        d = msg.to_dict()
        assert d["type"] == "ping"
        assert d["data"] == {"timestamp": "2024-01-01"}
        assert d["message_id"] == "test-id-123"
        assert "timestamp" in d
    
    def test_message_to_json(self):
        msg = WSMessage(type=WSMessageType.PONG, data={"status": "ok"})
        json_str = msg.to_json()
        parsed = json.loads(json_str)
        assert parsed["type"] == "pong"
        assert parsed["data"] == {"status": "ok"}
    
    def test_message_from_dict(self):
        d = {
            "type": "ask_response",
            "data": {"response": "test response"},
            "message_id": "msg-456",
            "timestamp": "2024-01-01T00:00:00",
        }
        msg = WSMessage.from_dict(d)
        assert msg.type == WSMessageType.ASK_RESPONSE
        assert msg.data == {"response": "test response"}
        assert msg.message_id == "msg-456"
    
    def test_message_from_json(self):
        json_str = '{"type": "error", "data": {"error": "test error"}, "message_id": "err-789"}'
        msg = WSMessage.from_json(json_str)
        assert msg.type == WSMessageType.ERROR
        assert msg.data == {"error": "test error"}
        assert msg.message_id == "err-789"


class TestWSClient:
    @pytest.fixture
    def mock_websocket(self):
        ws = AsyncMock()
        ws.send_text = AsyncMock()
        return ws
    
    def test_client_creation(self, mock_websocket):
        client = WSClient(websocket=mock_websocket, client_id="test-client-1")
        assert client.client_id == "test-client-1"
        assert client.is_alive
        assert len(client.subscriptions) == 0
    
    @pytest.mark.asyncio
    async def test_client_send(self, mock_websocket):
        client = WSClient(websocket=mock_websocket, client_id="test-client-1")
        msg = WSMessage(type=WSMessageType.PONG, data={"status": "ok"})
        
        await client.send(msg)
        
        mock_websocket.send_text.assert_called_once()
        sent_data = mock_websocket.send_text.call_args[0][0]
        assert json.loads(sent_data)["type"] == "pong"
    
    @pytest.mark.asyncio
    async def test_client_send_when_not_alive(self, mock_websocket):
        client = WSClient(websocket=mock_websocket, client_id="test-client-1")
        client.is_alive = False
        
        await client.send(WSMessage(type=WSMessageType.PONG))
        
        mock_websocket.send_text.assert_not_called()


class TestConnectionManager:
    @pytest.fixture
    def manager(self):
        return ConnectionManager()
    
    @pytest.fixture
    def mock_websocket(self):
        ws = AsyncMock()
        ws.accept = AsyncMock()
        ws.send_text = AsyncMock()
        ws.receive_text = AsyncMock()
        return ws
    
    @pytest.mark.asyncio
    async def test_connect_client(self, manager, mock_websocket):
        client = await manager.connect(mock_websocket, "client-1")
        
        assert client.client_id == "client-1"
        assert "client-1" in manager.clients
        mock_websocket.accept.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_connect_auto_generate_id(self, manager, mock_websocket):
        client = await manager.connect(mock_websocket)
        
        assert client.client_id is not None
        assert client.client_id in manager.clients
    
    @pytest.mark.asyncio
    async def test_disconnect_client(self, manager, mock_websocket):
        await manager.connect(mock_websocket, "client-1")
        await manager.disconnect("client-1")
        
        assert "client-1" not in manager.clients
    
    @pytest.mark.asyncio
    async def test_send_to_client(self, manager, mock_websocket):
        client = await manager.connect(mock_websocket, "client-1")
        msg = WSMessage(type=WSMessageType.ASK_RESPONSE, data={"response": "test"})
        
        await manager.send_to_client("client-1", msg)
        
        assert mock_websocket.send_text.call_count == 2
    
    @pytest.mark.asyncio
    async def test_broadcast(self, manager):
        ws1 = AsyncMock()
        ws1.accept = AsyncMock()
        ws1.send_text = AsyncMock()
        
        ws2 = AsyncMock()
        ws2.accept = AsyncMock()
        ws2.send_text = AsyncMock()
        
        await manager.connect(ws1, "client-1")
        await manager.connect(ws2, "client-2")
        
        msg = WSMessage(type=WSMessageType.INSIGHT, data={"insight": "test"})
        await manager.broadcast(msg)
        
        assert ws1.send_text.call_count == 2
        assert ws2.send_text.call_count == 2
    
    @pytest.mark.asyncio
    async def test_broadcast_with_exclude(self, manager):
        ws1 = AsyncMock()
        ws1.accept = AsyncMock()
        ws1.send_text = AsyncMock()
        
        ws2 = AsyncMock()
        ws2.accept = AsyncMock()
        ws2.send_text = AsyncMock()
        
        await manager.connect(ws1, "client-1")
        await manager.connect(ws2, "client-2")
        
        initial_count_1 = ws1.send_text.call_count
        initial_count_2 = ws2.send_text.call_count
        
        msg = WSMessage(type=WSMessageType.INSIGHT, data={"insight": "test"})
        await manager.broadcast(msg, exclude={"client-2"})
        
        assert ws1.send_text.call_count == initial_count_1 + 1
        assert ws2.send_text.call_count == initial_count_2
    
    @pytest.mark.asyncio
    async def test_subscribe_to_topic(self, manager, mock_websocket):
        client = await manager.connect(mock_websocket, "client-1")
        
        client.subscriptions.add("insights")
        
        msg = WSMessage(type=WSMessageType.EVENT, data={"event": "test"})
        await manager.broadcast_to_topic("insights", msg)
        
        assert mock_websocket.send_text.call_count == 2
    
    def test_on_message_handler(self, manager):
        @manager.on_message(WSMessageType.ASK)
        async def handler(client, msg):
            return "handled"
        
        assert WSMessageType.ASK in manager.global_handlers
        assert len(manager.global_handlers[WSMessageType.ASK]) == 1
    
    def test_on_message_handler_for_client(self, manager):
        @manager.on_message(WSMessageType.ASK, client_id="client-1")
        async def handler(client, msg):
            return "handled"
        
        assert "client-1" in manager.client_handlers
        assert WSMessageType.ASK in manager.client_handlers["client-1"]
    
    @pytest.mark.asyncio
    async def test_handle_message(self, manager, mock_websocket):
        handler_called = False
        
        @manager.on_message(WSMessageType.ASK)
        async def handler(client, msg):
            nonlocal handler_called
            handler_called = True
        
        client = await manager.connect(mock_websocket, "client-1")
        msg = WSMessage(type=WSMessageType.ASK, data={"question": "test"})
        
        await manager.handle_message(client, msg)
        
        assert handler_called


class TestStreamingResponseHandler:
    @pytest.fixture
    def manager(self):
        return ConnectionManager()
    
    @pytest.fixture
    def streaming_handler(self, manager):
        return StreamingResponseHandler(manager)
    
    @pytest.mark.asyncio
    async def test_stream_response(self, streaming_handler):
        async def response_gen():
            yield "Hello"
            yield " World"
            yield "!"
        
        mock_ws = AsyncMock()
        mock_ws.send_text = AsyncMock()
        
        manager = ConnectionManager()
        await manager.connect(mock_ws, "client-1")
        streaming = StreamingResponseHandler(manager)
        
        await streaming.stream_response(
            "client-1", "msg-123", response_gen()
        )
        
        await asyncio.sleep(0.1)
        
        assert mock_ws.send_text.call_count >= 3
    
    @pytest.mark.asyncio
    async def test_cancel_stream(self, streaming_handler, manager):
        async def slow_gen():
            yield "start"
            await asyncio.sleep(1)
            yield "end"
        
        mock_ws = AsyncMock()
        mock_ws.send_text = AsyncMock()
        await manager.connect(mock_ws, "client-1")
        
        streaming = StreamingResponseHandler(manager)
        
        await streaming.stream_response("client-1", "msg-123", slow_gen())
        
        await asyncio.sleep(0.05)
        
        await streaming.cancel_stream("msg-123")
        
        assert "msg-123" not in streaming.active_streams


class TestHeartbeatManager:
    @pytest.fixture
    def manager(self):
        return ConnectionManager()
    
    @pytest.fixture
    def heartbeat(self, manager):
        return HeartbeatManager(manager, ping_interval=0.1, pong_timeout=0.1)
    
    @pytest.mark.asyncio
    async def test_heartbeat_starts_and_stops(self, heartbeat):
        await heartbeat.start()
        assert heartbeat._running
        
        await heartbeat.stop()
        assert not heartbeat._running
    
    @pytest.mark.asyncio
    async def test_heartbeat_sends_ping(self, heartbeat, manager):
        mock_ws = AsyncMock()
        mock_ws.accept = AsyncMock()
        mock_ws.send_text = AsyncMock()
        
        await manager.connect(mock_ws, "client-1")
        
        await heartbeat.start()
        await asyncio.sleep(0.15)
        await heartbeat.stop()
        
        mock_ws.send_text.assert_called()
    
    @pytest.mark.asyncio
    async def test_heartbeat_removes_dead_clients(self, heartbeat, manager):
        mock_ws = AsyncMock()
        mock_ws.accept = AsyncMock()
        mock_ws.send_text = AsyncMock()
        mock_ws.receive_text = AsyncMock()
        
        await manager.connect(mock_ws, "client-1")
        
        client = manager.clients["client-1"]
        client.last_pong = 0
        
        await heartbeat.start()
        await asyncio.sleep(0.25)
        await heartbeat.stop()
        
        assert "client-1" not in manager.clients


class TestWebSocketServer:
    @pytest.fixture
    def server(self):
        return WebSocketServer()
    
    def test_server_creates_manager(self, server):
        assert server.manager is not None
    
    def test_server_creates_streaming_handler(self, server):
        assert server.streaming is not None
    
    def test_server_creates_heartbeat_by_default(self, server):
        assert server.heartbeat is not None
    
    def test_server_without_heartbeat(self):
        server = WebSocketServer(enable_heartbeat=False)
        assert server.heartbeat is None
    
    def test_get_router_returns_api_router(self, server):
        router = server.get_router()
        assert router is not None
        assert hasattr(router, 'routes')
    
    @pytest.mark.asyncio
    async def test_start_and_stop_heartbeat(self, server):
        if server.heartbeat:
            await server.start_heartbeat()
            assert server.heartbeat._running
            
            await server.stop_heartbeat()
            assert not server.heartbeat._running
    
    @pytest.mark.asyncio
    async def test_handle_connection_default_handlers(self, server):
        assert WSMessageType.ASK in server.manager.global_handlers
        assert WSMessageType.OBSERVE in server.manager.global_handlers


class TestMessageTypes:
    def test_all_message_types_are_strings(self):
        for msg_type in WSMessageType:
            assert isinstance(msg_type.value, str)
    
    def test_message_type_values(self):
        assert WSMessageType.CONNECT.value == "connect"
        assert WSMessageType.ASK.value == "ask"
        assert WSMessageType.ASK_STREAM.value == "ask_stream"
        assert WSMessageType.PING.value == "ping"
        assert WSMessageType.PONG.value == "pong"


class TestIntegration:
    @pytest.mark.asyncio
    async def test_full_client_flow(self):
        manager = ConnectionManager()
        
        mock_ws = AsyncMock()
        mock_ws.accept = AsyncMock()
        mock_ws.send_text = AsyncMock()
        mock_ws.receive_text = AsyncMock()
        mock_ws.receive_text.side_effect = [
            '{"type": "subscribe", "data": {"topic": "insights"}}',
            '{"type": "ping", "data": {}}',
            WebSocketDisconnect(),
        ]
        
        client = await manager.connect(mock_ws, "test-client")
        
        assert client.client_id == "test-client"
        assert "test-client" in manager.clients
        
        received_calls = []
        
        def capture_send(*args):
            received_calls.append(args[0])
        
        mock_ws.send_text.side_effect = capture_send
        
        subscribe_msg = WSMessage.from_json('{"type": "subscribe", "data": {"topic": "insights"}}')
        client.subscriptions.add(subscribe_msg.data["topic"])
        
        assert "insights" in client.subscriptions
        
        await manager.disconnect("test-client")
        
        assert "test-client" not in manager.clients
    
    @pytest.mark.asyncio
    async def test_concurrent_connections(self):
        manager = ConnectionManager()
        
        clients = []
        for i in range(5):
            mock_ws = AsyncMock()
            mock_ws.accept = AsyncMock()
            mock_ws.send_text = AsyncMock()
            
            client = await manager.connect(mock_ws, f"client-{i}")
            clients.append(client)
        
        assert manager.get_client_count() == 5
        
        assert len(manager.get_connected_clients()) == 5
        
        msg = WSMessage(type=WSMessageType.INSIGHT, data={"insight": "test"})
        await manager.broadcast(msg)
        
        for client in clients:
            assert client.websocket.send_text.call_count == 2


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_send_to_disconnected_client(self):
        manager = ConnectionManager()
        
        mock_ws = AsyncMock()
        mock_ws.accept = AsyncMock()
        mock_ws.send_text = AsyncMock()
        
        client = await manager.connect(mock_ws, "client-1")
        initial_call_count = mock_ws.send_text.call_count
        client.is_alive = False
        
        msg = WSMessage(type=WSMessageType.PING, data={})
        await manager.send_to_client("client-1", msg)
        
        assert mock_ws.send_text.call_count == initial_call_count
    
    @pytest.mark.asyncio
    async def test_handler_exception_sends_error(self):
        manager = ConnectionManager()
        
        @manager.on_message(WSMessageType.ASK)
        async def bad_handler(client, msg):
            raise ValueError("Test error")
        
        mock_ws = AsyncMock()
        mock_ws.accept = AsyncMock()
        mock_ws.send_text = AsyncMock()
        
        client = await manager.connect(mock_ws, "client-1")
        msg = WSMessage(type=WSMessageType.ASK, data={"question": "test"})
        
        await manager.handle_message(client, msg)
        
        calls = [json.loads(call[0][0]) for call in mock_ws.send_text.call_args_list]
        error_calls = [c for c in calls if c.get("type") == "error"]
        assert len(error_calls) == 1


class TestSubscriptions:
    @pytest.mark.asyncio
    async def test_subscribe_to_multiple_topics(self):
        manager = ConnectionManager()
        
        mock_ws = AsyncMock()
        mock_ws.accept = AsyncMock()
        mock_ws.send_text = AsyncMock()
        
        client = await manager.connect(mock_ws, "client-1")
        
        client.subscriptions.add("topic-1")
        client.subscriptions.add("topic-2")
        
        assert "topic-1" in client.subscriptions
        assert "topic-2" in client.subscriptions
    
    @pytest.mark.asyncio
    async def test_broadcast_to_topic_subscribers_only(self):
        manager = ConnectionManager()
        
        ws1 = AsyncMock()
        ws1.accept = AsyncMock()
        ws1.send_text = AsyncMock()
        
        ws2 = AsyncMock()
        ws2.accept = AsyncMock()
        ws2.send_text = AsyncMock()
        
        client1 = await manager.connect(ws1, "client-1")
        client2 = await manager.connect(ws2, "client-2")
        
        initial_count_1 = ws1.send_text.call_count
        initial_count_2 = ws2.send_text.call_count
        
        client1.subscriptions.add("alerts")
        
        msg = WSMessage(type=WSMessageType.EVENT, data={"event": "alert"})
        await manager.broadcast_to_topic("alerts", msg)
        
        assert ws1.send_text.call_count == initial_count_1 + 1
        assert ws2.send_text.call_count == initial_count_2
    
    def test_get_topic_subscribers(self):
        manager = ConnectionManager()
        
        client1 = MagicMock()
        client1.client_id = "client-1"
        client1.subscriptions = {"alerts", "news"}
        client1.is_alive = True
        
        client2 = MagicMock()
        client2.client_id = "client-2"
        client2.subscriptions = {"news"}
        client2.is_alive = True
        
        manager.clients = {"client-1": client1, "client-2": client2}
        
        alert_subs = manager.get_topic_subscribers("alerts")
        assert "client-1" in alert_subs
        assert "client-2" not in alert_subs
        
        news_subs = manager.get_topic_subscribers("news")
        assert "client-1" in news_subs
        assert "client-2" in news_subs


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])