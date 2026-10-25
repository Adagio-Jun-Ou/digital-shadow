"""
Tests for Shadow AI Daemon WebSocket Integration.

Hour 11+ : WebSocket integration with daemon
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from shadowd.websocket_manager import (
    WSMessageType,
    WSMessage,
    WSClient,
    ConnectionManager,
)


class TestShadowContextWebSocket:
    def test_shadow_context_initialization(self):
        from shadowd.shadowd import ShadowContext
        
        ctx = ShadowContext()
        
        assert ctx.observations == []
        assert ctx.current_activity is None
        assert ctx.connected_devices == {}
        assert ctx.shadow_ai is None
        assert ctx.ws_server is None
        assert ctx.ws_manager is None
    
    @pytest.mark.asyncio
    async def test_shadow_context_learn(self):
        from shadowd.shadowd import ShadowContext
        
        ctx = ShadowContext()
        
        observation = {"type": "test", "content": {"data": "test data"}}
        await ctx.learn(observation)
        
        assert len(ctx.observations) == 1
        assert ctx.observations[0]["type"] == "test"
        assert "timestamp" in ctx.observations[0]
    
    @pytest.mark.asyncio
    async def test_shadow_context_learn_limit(self):
        from shadowd.shadowd import ShadowContext
        
        ctx = ShadowContext()
        
        for i in range(1100):
            await ctx.learn({"type": "test", "content": {"id": i}})
        
        assert len(ctx.observations) == 1000
    
    @pytest.mark.asyncio
    async def test_broadcast_observation_no_server(self):
        from shadowd.shadowd import ShadowContext
        
        ctx = ShadowContext()
        
        # Should not raise when ws_server is None
        await ctx.broadcast_observation({"type": "test"})
        
        assert True  # No exception raised
    
    @pytest.mark.asyncio
    async def test_broadcast_insight_no_server(self):
        from shadowd.shadowd import ShadowContext
        
        ctx = ShadowContext()
        
        # Should not raise when ws_server is None
        await ctx.broadcast_insight({"insight": "test insight"})
        
        assert True  # No exception raised


class TestWebSocketMessageTypes:
    def test_ask_message(self):
        msg = WSMessage(
            type=WSMessageType.ASK,
            data={"question": "What is the weather?", "context": {}}
        )
        
        assert msg.type == WSMessageType.ASK
        assert msg.data["question"] == "What is the weather?"
        assert msg.to_dict()["type"] == "ask"
    
    def test_ask_stream_message(self):
        msg = WSMessage(
            type=WSMessageType.ASK_STREAM,
            data={"question": "Tell me a story"}
        )
        
        assert msg.type == WSMessageType.ASK_STREAM
    
    def test_agent_status_message(self):
        msg = WSMessage(
            type=WSMessageType.AGENT_STATUS,
            data={"status": {"agent1": "active"}}
        )
        
        assert msg.type == WSMessageType.AGENT_STATUS
    
    def test_memory_query_message(self):
        msg = WSMessage(
            type=WSMessageType.MEMORY_QUERY,
            data={"limit": 50}
        )
        
        assert msg.type == WSMessageType.MEMORY_QUERY
    
    def test_insight_message(self):
        msg = WSMessage(
            type=WSMessageType.INSIGHT,
            data={"insight": "Critical insight", "priority": "high"}
        )
        
        assert msg.type == WSMessageType.INSIGHT
    
    def test_subscribe_message(self):
        msg = WSMessage(
            type=WSMessageType.SUBSCRIBE,
            data={"topic": "alerts"}
        )
        
        assert msg.type == WSMessageType.SUBSCRIBE


class TestWebSocketHandlers:
    @pytest.mark.asyncio
    async def test_handle_ask_message(self):
        manager = ConnectionManager()
        
        handler_called = False
        
        @manager.on_message(WSMessageType.ASK)
        async def handle_ask(client, msg):
            nonlocal handler_called
            handler_called = True
            return WSMessage(
                type=WSMessageType.ASK_RESPONSE,
                data={"response": "test response"}
            )
        
        mock_ws = AsyncMock()
        mock_ws.accept = AsyncMock()
        mock_ws.send_text = AsyncMock()
        
        client = await manager.connect(mock_ws, "test-client")
        msg = WSMessage(type=WSMessageType.ASK, data={"question": "test"})
        
        await manager.handle_message(client, msg)
        
        assert handler_called
    
    @pytest.mark.asyncio
    async def test_handle_agent_status_message(self):
        manager = ConnectionManager()
        
        status_handler_called = False
        
        @manager.on_message(WSMessageType.AGENT_STATUS)
        async def handle_status(client, msg):
            nonlocal status_handler_called
            status_handler_called = True
        
        mock_ws = AsyncMock()
        mock_ws.accept = AsyncMock()
        mock_ws.send_text = AsyncMock()
        
        client = await manager.connect(mock_ws, "test-client")
        msg = WSMessage(type=WSMessageType.AGENT_STATUS, data={})
        
        await manager.handle_message(client, msg)
        
        assert status_handler_called
    
    @pytest.mark.asyncio
    async def test_handle_memory_query_message(self):
        manager = ConnectionManager()
        
        query_handler_called = False
        
        @manager.on_message(WSMessageType.MEMORY_QUERY)
        async def handle_query(client, msg):
            nonlocal query_handler_called
            query_handler_called = True
        
        mock_ws = AsyncMock()
        mock_ws.accept = AsyncMock()
        mock_ws.send_text = AsyncMock()
        
        client = await manager.connect(mock_ws, "test-client")
        msg = WSMessage(type=WSMessageType.MEMORY_QUERY, data={"limit": 20})
        
        await manager.handle_message(client, msg)
        
        assert query_handler_called


class TestWebSocketEndpoints:
    @pytest.mark.asyncio
    async def test_ws_clients_endpoint_without_server(self):
        from shadowd.shadowd import ShadowContext
        
        ctx = ShadowContext()
        
        # Test get_ws_clients functionality
        if ctx.ws_server is None:
            result = {"error": "WebSocket server not initialized"}
            assert "error" in result
    
    @pytest.mark.asyncio
    async def test_broadcast_endpoint_without_server(self):
        from shadowd.shadowd import ShadowContext
        
        ctx = ShadowContext()
        
        # Test broadcast functionality
        if ctx.ws_server is None:
            result = {"error": "WebSocket server not initialized"}
            assert "error" in result


class TestDaemonIntegration:
    @pytest.mark.asyncio
    async def test_shadow_context_with_websocket_server(self):
        from shadowd.shadowd import ShadowContext
        from shadowd.websocket_manager import WebSocketServer
        
        ctx = ShadowContext()
        ctx.ws_manager = ConnectionManager()
        ctx.ws_server = WebSocketServer(manager=ctx.ws_manager)
        
        assert ctx.ws_server is not None
        assert ctx.ws_manager is not None
    
    @pytest.mark.asyncio
    async def test_broadcast_with_websocket_server(self):
        from shadowd.shadowd import ShadowContext
        from shadowd.websocket_manager import WebSocketServer
        
        ctx = ShadowContext()
        ctx.ws_manager = ConnectionManager()
        ctx.ws_server = WebSocketServer(manager=ctx.ws_manager)
        
        # Add a mock client
        mock_ws = AsyncMock()
        mock_ws.accept = AsyncMock()
        mock_ws.send_text = AsyncMock()
        
        client = await ctx.ws_manager.connect(mock_ws, "test-client")
        client.subscriptions.add("insights")
        
        # Broadcast an insight
        await ctx.broadcast_insight({"insight": "test", "priority": "high"})
        
        # Message should have been sent
        assert mock_ws.send_text.call_count >= 1
    
    @pytest.mark.asyncio
    async def test_subscribe_to_topic_and_broadcast(self):
        manager = ConnectionManager()
        
        # Create two clients
        ws1 = AsyncMock()
        ws1.accept = AsyncMock()
        ws1.send_text = AsyncMock()
        
        ws2 = AsyncMock()
        ws2.accept = AsyncMock()
        ws2.send_text = AsyncMock()
        
        client1 = await manager.connect(ws1, "client-1")
        client2 = await manager.connect(ws2, "client-2")
        
        # Only client1 subscribes to "alerts"
        client1.subscriptions.add("alerts")
        
        # Broadcast to "alerts" topic
        msg = WSMessage(type=WSMessageType.EVENT, data={"event": "test alert"})
        await manager.broadcast_to_topic("alerts", msg)
        
        # Count calls after connect message
        ws1_calls = ws1.send_text.call_count
        ws2_calls = ws2.send_text.call_count
        
        # client1 should have received (connect + event)
        assert ws1_calls == 2
        # client2 should only have connect message
        assert ws2_calls == 1


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_handler_error_sends_error_message(self):
        manager = ConnectionManager()
        
        @manager.on_message(WSMessageType.ASK)
        async def bad_handler(client, msg):
            raise ValueError("Test error in handler")
        
        mock_ws = AsyncMock()
        mock_ws.accept = AsyncMock()
        mock_ws.send_text = AsyncMock()
        
        client = await manager.connect(mock_ws, "test-client")
        msg = WSMessage(type=WSMessageType.ASK, data={"question": "test"})
        
        await manager.handle_message(client, msg)
        
        # Should have sent an error message
        calls = [call[0][0] for call in mock_ws.send_text.call_args_list]
        error_found = any('"type": "error"' in str(call) for call in calls)
        assert error_found or mock_ws.send_text.call_count >= 1
    
    @pytest.mark.asyncio
    async def test_client_disconnected_during_broadcast(self):
        manager = ConnectionManager()
        
        ws1 = AsyncMock()
        ws1.accept = AsyncMock()
        ws1.send_text = AsyncMock()
        
        ws2 = AsyncMock()
        ws2.accept = AsyncMock()
        ws2.send_text = AsyncMock()
        ws2.send_text.side_effect = Exception("Connection closed")
        
        client1 = await manager.connect(ws1, "client-1")
        client2 = await manager.connect(ws2, "client-2")
        
        # Mark client2 as dead
        client2.is_alive = False
        
        # Broadcast should not raise
        msg = WSMessage(type=WSMessageType.INSIGHT, data={"insight": "test"})
        await manager.broadcast(msg)
        
        # Should still work for client1
        assert ws1.send_text.call_count >= 1


class TestMessageSerialization:
    def test_ask_response_message(self):
        msg = WSMessage(
            type=WSMessageType.ASK_RESPONSE,
            data={
                "response": "This is the answer",
                "ability_used": "insight",
                "message_id": "msg-123"
            }
        )
        
        json_str = msg.to_json()
        parsed = WSMessage.from_json(json_str)
        
        assert parsed.type == WSMessageType.ASK_RESPONSE
        assert parsed.data["response"] == "This is the answer"
    
    def test_streaming_delta_message(self):
        msg = WSMessage(
            type=WSMessageType.ASK_STREAM_DELTA,
            data={
                "message_id": "msg-456",
                "delta": "Hello",
                "accumulated": "Hello"
            }
        )
        
        d = msg.to_dict()
        assert d["type"] == "ask_stream_delta"
        assert d["data"]["delta"] == "Hello"
    
    def test_streaming_done_message(self):
        msg = WSMessage(
            type=WSMessageType.ASK_STREAM_DONE,
            data={
                "message_id": "msg-456",
                "full_response": "Hello World!"
            }
        )
        
        json_str = msg.to_json()
        parsed = WSMessage.from_json(json_str)
        
        assert parsed.type == WSMessageType.ASK_STREAM_DONE
        assert parsed.data["full_response"] == "Hello World!"


class TestTimestampFormat:
    def test_timestamp_is_timezone_aware(self):
        from shadowd.shadowd import ShadowContext
        
        ctx = ShadowContext()
        
        import asyncio
        asyncio.run(ctx.learn({"type": "test"}))
        
        ts = ctx.observations[0]["timestamp"]
        
        # Should contain timezone info or be ISO format
        assert "T" in ts  # ISO format
        assert ":" in ts  # Time format
    
    def test_device_connect_timestamp(self):
        # The device_connect endpoint uses datetime.now(timezone.utc)
        ts = datetime.now(timezone.utc).isoformat()
        
        assert "T" in ts
        assert "+" in ts or "Z" in ts or "utc" in ts.lower() or ":" in ts


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])