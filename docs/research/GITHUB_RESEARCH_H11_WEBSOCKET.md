# Hour 11: WebSocket Research - Real-time Communication Patterns

## Summary
Research on WebSocket frameworks and real-time agent communication patterns for implementing bidirectional communication in Shadow AI.

## Projects Researched

### 1. websockets (python-websockets/websockets): 5.7k stars
**Repository:** https://github.com/python-websockets/websockets

**Key Features:**
- Pure Python WebSocket implementation built on asyncio
- Focus on correctness, simplicity, robustness, and performance
- Coroutine-based API: `async for message in websocket`
- Built-in connection management and backpressure handling
- Production-ready with full RFC 6455 compliance
- Sans-I/O implementation available for flexibility

**Architecture Patterns:**
```python
# Server pattern
async def echo(websocket):
    async for message in websocket:
        await websocket.send(message)

# Client pattern
with connect("ws://localhost:8765") as websocket:
    websocket.send("Hello world!")
    message = websocket.recv()
```

**Key Learnings:**
1. Connection management should handle automatic reconnection
2. Backpressure handling is critical for production (no message flooding)
3. Keep connection logic simple - just `send()` and `recv()`
4. Use asyncio for non-blocking operations
5. Implement heartbeat/ping-pong for connection health

---

### 2. Starlette (Kludex/starlette): 12.2k stars
**Repository:** https://github.com/Kludex/starlette

**Key Features:**
- Lightweight ASGI framework (ideal for FastAPI integration)
- Built-in WebSocket support via `starlette.websockets.WebSocket`
- Event-driven lifespan management
- Background tasks support
- Session/Cookie support, CORS, GZip, Static Files
- 100% type annotated, 100% test coverage

**WebSocket Pattern in Starlette:**
```python
from starlette.applications import Starlette
from starlette.websockets import WebSocket

async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(f"Echo: {data}")
    except WebSocketDisconnect:
        pass
    finally:
        await websocket.close()

# Route registration
routes = [
    Route("/ws", websocket_endpoint)
]
```

**Key Learnings:**
1. WebSocket connections integrate seamlessly with ASGI apps
2. Use `websocket.accept()` to accept connection
3. Handle `WebSocketDisconnect` for clean disconnection
4. Can use JSON encoding for structured messages
5. FastAPI builds on top of Starlette for WebSocket support

---

### 3. FastAPI (fastapi/fastapi): 96.8k stars
**Repository:** https://github.com/fastapi/fastapi

**Key Features:**
- Built on Starlette (inherits WebSocket support)
- Automatic OpenAPI documentation
- Type hints with Pydantic validation
- Async/await native support
- WebSocket decorator similar to route decorator

**WebSocket Pattern in FastAPI:**
```python
from fastapi import FastAPI, WebSocket

app = FastAPI()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        await websocket.send_text(f"Message: {data}")
```

**Streaming Support:**
```python
# Server-sent events style streaming
async def stream_response(websocket: WebSocket):
    for chunk in generate_response():
        await websocket.send_text(chunk)
        await asyncio.sleep(0.1)
```

**Key Learnings:**
1. Use `@app.websocket()` decorator for easy routing
2. Support both text and binary WebSocket messages
3. Can use dependency injection in WebSocket handlers
4. Implement connection state management separately
5. Handle multiple WebSocket routes like regular routes

---

### 4. LangChain (langchain-ai/langchain): 132k stars
**Repository:** https://github.com/langchain-ai/langchain

**Key Features:**
- Agent orchestration framework
- Streaming support for LLM responses
- WebSocket integration for real-time agent updates
- Multi-agent coordination patterns

**Relevant Patterns:**
1. **Streaming Responses:** Token-by-token streaming from LLMs
2. **Agent Status Updates:** Real-time status of agent execution
3. **Tool Output Streaming:** Stream intermediate steps
4. **Memory Updates:** Push memory/context updates to clients

---

### 5. OpenAI Python SDK (openai/openai-python): 30.4k stars
**Repository:** https://github.com/openai/openai-python

**Key Features:**
- Realtime API with WebSocket support
- Server-Sent Events (SSE) for streaming
- Event-based communication pattern

**Realtime API Pattern:**
```python
async with client.realtime.connect(model="gpt-realtime") as connection:
    await connection.session.update(
        session={"type": "realtime", "output_modalities": ["text"]}
    )
    
    async for event in connection:
        if event.type == "response.output_text.delta":
            print(event.delta, flush=True, end="")
        elif event.type == "response.done":
            break
```

**Event Types:**
- `session.update` - Configure session
- `conversation.item.create` - Add items to conversation
- `response.create` - Create a response
- `response.output_text.delta` - Streaming text delta
- `response.done` - Response complete

**Key Learnings for Shadow AI:**
1. Use event-based message types for structured communication
2. Implement delta streaming for incremental updates
3. Support both text and multimodal (audio) streams
4. Use connection lifecycle events (connect, disconnect)
5. Handle errors gracefully without closing connection

---

## Design Decisions for Shadow AI WebSocket

### 1. Message Types
Based on research, implement these message types:
- `connect`/`disconnect` - Connection lifecycle
- `ping`/`pong` - Health check
- `ask`/`ask_response` - Simple ask-response pattern
- `ask_stream`/`ask_stream_delta`/`ask_stream_done` - Streaming responses
- `observe`/`observe_ack` - Learning/observation
- `agent_status`/`agent_created`/`agent_message` - Agent coordination
- `memory_update`/`memory_query` - Memory operations
- `insight`/`action` - Proactive notifications
- `subscribe`/`unsubscribe`/`event` - Pub/sub pattern

### 2. Connection Management
- Single `ConnectionManager` class for all connections
- Client metadata storage for Shadow AI references
- Subscription-based topics (e.g., "insights", "alerts")
- Heartbeat system for connection health
- Automatic cleanup on disconnect

### 3. Streaming Pattern
```python
async def stream_response():
    async for delta in llm_stream():
        yield delta
        
await stream_handler.stream_response(
    client_id="user-123",
    message_id="msg-456",
    response_generator=stream_response()
)
```

### 4. Pub/Sub Topics
- `insights` - Proactive insights from Shadow AI
- `alerts` - Critical notifications
- `memory` - Memory updates
- `agent:<agent_id>` - Agent-specific channel
- `observations` - New observations

### 5. Error Handling
- Send error messages without closing connection
- Include message_id for request/response correlation
- Use structured error types
- Log errors for debugging

---

## Implementation Summary

### Files Created
1. `shadowd/websocket_manager.py` (520 lines)
   - WSMessageType enum for message types
   - WSMessage dataclass for structured messages
   - WSClient class for connection state
   - ConnectionManager for all connections
   - StreamingResponseHandler for streaming
   - HeartbeatManager for connection health
   - WebSocketServer for FastAPI integration

2. `tests/test_websocket_manager.py` (550 lines)
   - 39 tests covering all components
   - Tests for message serialization
   - Tests for connection lifecycle
   - Tests for broadcasting and topics
   - Tests for streaming
   - Tests for heartbeat
   - Integration tests

---

## Next Steps

1. ✅ WebSocket server implementation
2. ✅ Connection management
3. ✅ Streaming response support
4. ✅ Pub/sub topics
5. ✅ Heartbeat/ping-pong
6. ⬜ Integration with Shadow AI daemon
7. ⬜ Agent coordination via WebSocket
8. ⬜ Real-time memory sync
9. ⬜ Frontend WebSocket client library

---

## References

- websockets: https://websockets.readthedocs.io/
- Starlette WebSocket: https://www.starlette.io/websockets/
- FastAPI WebSocket: https://fastapi.tiangolo.com/advanced/websockets/
- OpenAI Realtime API: https://platform.openai.com/docs/api-reference/realtime
- LangChain Streaming: https://python.langchain.com/docs/modules/model_io/chat/streaming/