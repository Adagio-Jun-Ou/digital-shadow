#!/usr/bin/env python3
"""
ShadowVoice - Real-Time Voice Agent for Shadow AI

Implements voice interaction using the existing voice server stack:
- Piper TTS for fast, local text-to-speech
- Whisper STT for local speech-to-text
- WebSocket-based real-time streaming
- OpenAI-compatible voice API

Architecture:
  ShadowVoice
  ├── VoiceSession (manages a single voice conversation)
  ├── VoiceCommand (structured voice commands with intent)
  ├── VoiceStream (real-time audio streaming via WebSocket)
  ├── TTSEngine (Piper via local voice server at :8001)
  ├── STEEngine (Whisper via local voice server at :8001)
  └── VoicePersonality (applies personality to speech patterns)

Connects to the local voice server at http://host.docker.internal:8001
or http://localhost:8001 using the OpenAI-compatible API.

Inspired by:
- OpenAI Realtime API (WebSocket voice streaming)
- ElevenLabs conversational AI patterns
- Alexa/Google Assistant intent parsing patterns
"""

import asyncio
import copy
import json
import re
import threading
import time
import uuid
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import httpx

logger = logging.getLogger("shadowd.cognition.shadow_voice")


class VoiceState(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    THINKING = "thinking"
    ERROR = "error"


class VoiceIntent(str, Enum):
    QUERY = "query"
    COMMAND = "command"
    CONVERSATION = "conversation"
    EMERGENCY = "emergency"
    FEEDBACK = "feedback"
    CANCEL = "cancel"
    GREETING = "greeting"
    FAREWELL = "farewell"
    UNKNOWN = "unknown"


@dataclass
class VoiceCommand:
    command_id: str = ""
    intent: VoiceIntent = VoiceIntent.UNKNOWN
    transcript: str = ""
    confidence: float = 0.0
    entities: Dict[str, Any] = field(default_factory=dict)
    action: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.command_id:
            self.command_id = f"vcmd_{uuid.uuid4().hex[:12]}"

    def to_dict(self) -> Dict:
        return {
            "command_id": self.command_id,
            "intent": self.intent.value,
            "transcript": self.transcript,
            "confidence": round(self.confidence, 3),
            "entities": copy.deepcopy(self.entities),
            "action": self.action,
            "parameters": copy.deepcopy(self.parameters),
            "timestamp": self.timestamp,
        }


@dataclass
class VoiceSession:
    _MAX_COMMANDS = 500
    _MAX_TRANSCRIPT = 1000
    session_id: str = ""
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    ended_at: Optional[str] = None
    state: VoiceState = VoiceState.IDLE
    voice: str = "en_US-ryan-high"
    commands: List[VoiceCommand] = field(default_factory=list)
    transcript: List[Dict] = field(default_factory=list)
    duration_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    _start_time: float = field(default_factory=time.monotonic)

    def __post_init__(self):
        if not self.session_id:
            self.session_id = f"vsess_{uuid.uuid4().hex[:12]}"

    def to_dict(self) -> Dict:
        return {
            "session_id": self.session_id,
            "started_at": self.started_at,
            "state": self.state.value,
            "voice": self.voice,
            "commands_count": len(self.commands),
            "transcript_entries": len(self.transcript),
            "transcript": list(self.transcript[-20:]),
            "duration_seconds": round(self.duration_seconds, 1),
            "metadata": copy.deepcopy(self.metadata) if self.metadata else {},
        }

    def add_transcript(self, speaker: str, text: str):
        speaker = re.sub(r'[<>\x00-\x1f\x7f]', '', speaker).strip()[:200]
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text).strip()[:5000]
        self.transcript.append({
            "speaker": speaker,
            "text": text,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        if len(self.transcript) > self._MAX_TRANSCRIPT:
            self.transcript = self.transcript[-self._MAX_TRANSCRIPT:]

    def end(self):
        self.ended_at = datetime.now(timezone.utc).isoformat()
        self.duration_seconds = time.monotonic() - self._start_time
        self.state = VoiceState.IDLE


class IntentParser:
    INTENT_PATTERNS = {
        VoiceIntent.CANCEL: [r"^(cancel|stop|never mind|forget it|abort)\b", r"\b(cancel|stop|never mind|forget it|abort)\b"],
        VoiceIntent.EMERGENCY: [r"^(urgent|emergency|help|critical|danger|alert)\b", r"\b(urgent|emergency|help|critical|danger|alert)\b"],
        VoiceIntent.QUERY: [r"^(what|how|why|when|where|who|which|is it|can you tell)\b", r"\b(what|how come|why|when|where|who|which)\b"],
        VoiceIntent.COMMAND: [r"^(do|make|create|run|start|set|turn on|turn off|schedule)\b", r"\b(do|make|create|run|start|set|turn on|turn off|schedule)\b"],
        VoiceIntent.FEEDBACK: [r"^(no|wrong|correct|actually|instead|i meant)\b", r"\b(no that's wrong|correct|actually|instead|i meant)\b"],
        VoiceIntent.GREETING: [r"^(hello|hey|hi|good morning|good evening|wake up)\b"],
        VoiceIntent.FAREWELL: [r"^(bye|goodbye|see you|later|good night|sleep)\b"],
        VoiceIntent.CONVERSATION: [],
    }

    def parse(self, transcript: str) -> Tuple[VoiceIntent, float]:
        lower = transcript.lower().strip()

        if not lower:
            return VoiceIntent.UNKNOWN, 0.0

        for intent, patterns in self.INTENT_PATTERNS.items():
            if intent == VoiceIntent.CONVERSATION:
                continue
            for pattern in patterns:
                if re.search(pattern, lower):
                    confidence = 0.8 if re.match(pattern, lower) else 0.6
                    return intent, confidence

        return VoiceIntent.CONVERSATION, 0.5

    def extract_entities(self, transcript: str) -> Dict[str, Any]:
        entities = {}
        words = transcript.lower().split()

        time_words = ["today", "tomorrow", "monday", "tuesday", "wednesday",
                       "thursday", "friday", "saturday", "sunday", "now", "later"]
        for word in words:
            if word in time_words:
                entities["time_reference"] = word

        ability_names = ["oracle", "x-ray", "life pulse", "perfect memory",
                         "pattern sight", "social radar", "intent seer",
                         "omni sense", "shadow council", "reality forge"]
        for ability in ability_names:
            if ' ' in ability:
                if ability in transcript.lower():
                    entities["ability"] = ability
            else:
                if re.search(r'\b' + re.escape(ability) + r'\b', transcript.lower()):
                    entities["ability"] = ability

        domain_words = {"work", "health", "finance", "home", "social", "learning", "creative"}
        for word in words:
            if word in domain_words:
                entities["domain"] = word

        return entities


class TTSEngine:
    def __init__(self, base_url: str = "http://localhost:8001", default_voice: str = "en_US-ryan-high"):
        self.base_url = base_url.rstrip("/")
        self.default_voice = default_voice
        self._client: Optional[httpx.AsyncClient] = None
        self._client_lock = asyncio.Lock()

    async def _get_client(self) -> httpx.AsyncClient:
        async with self._client_lock:
            if self._client is None or self._client.is_closed:
                self._client = httpx.AsyncClient(timeout=30.0)
            return self._client

    async def speak(self, text: str, voice: Optional[str] = None) -> Optional[bytes]:
        client = await self._get_client()
        try:
            response = await client.post(
                f"{self.base_url}/v1/audio/speech",
                json={
                    "input": text,
                    "voice": voice or self.default_voice,
                    "model": "piper-tts",
                },
            )
            response.raise_for_status()
            return response.content
        except (ConnectionError, OSError, TimeoutError, ValueError, RuntimeError, httpx.HTTPError) as e:
            logger.error("TTS error: %s", type(e).__name__)
            return None

    async def list_voices(self) -> List[Dict]:
        client = await self._get_client()
        try:
            response = await client.get(f"{self.base_url}/v1/voices")
            response.raise_for_status()
            return response.json().get("voices", [])
        except (ConnectionError, OSError, TimeoutError, ValueError, RuntimeError, httpx.HTTPError) as e:
            logger.error("List voices error: %s", type(e).__name__)
            return []

    async def health(self) -> Dict:
        client = await self._get_client()
        try:
            response = await client.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except (ConnectionError, OSError, TimeoutError, ValueError, RuntimeError, httpx.HTTPError) as e:
            return {"status": "unavailable", "error": type(e).__name__}

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()


class STTEngine:
    def __init__(self, base_url: str = "http://localhost:8001", model: str = "whisper-base"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client: Optional[httpx.AsyncClient] = None
        self._client_lock = asyncio.Lock()

    async def _get_client(self) -> httpx.AsyncClient:
        async with self._client_lock:
            if self._client is None or self._client.is_closed:
                self._client = httpx.AsyncClient(timeout=30.0)
            return self._client

    async def transcribe(self, audio_data: bytes, language: Optional[str] = None) -> Optional[str]:
        client = await self._get_client()
        try:
            response = await client.post(
                f"{self.base_url}/v1/audio/transcriptions",
                files={"file": ("audio.wav", audio_data, "audio/wav")},
                data={"model": self.model, "language": language or "en"},
            )
            response.raise_for_status()
            return response.json().get("text", "")
        except (ConnectionError, OSError, TimeoutError, ValueError, RuntimeError, httpx.HTTPError) as e:
            logger.error("STT error: %s", type(e).__name__)
            return None

    async def transcribe_stream(self, audio_chunks: asyncio.Queue) -> Optional[str]:
        full_text = []
        while True:
            try:
                chunk = await asyncio.wait_for(audio_chunks.get(), timeout=5.0)
                if chunk is None:
                    break
                result = await self.transcribe(chunk)
                if result:
                    full_text.append(result)
            except asyncio.TimeoutError:
                break
        return " ".join(full_text) if full_text else None

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()


class VoicePersonality:
    def __init__(self, communication_style: str = "casual", verbosity: str = "concise"):
        self.communication_style = communication_style
        self.verbosity = verbosity
        self._fillers = {
            "casual": ["hmm", "let me think", "well", "so"],
            "formal": ["indeed", "certainly", "I see", "I understand"],
        }

    def adapt_response(self, text: str, personality: Dict = None) -> str:
        if personality:
            style = personality.get("communication_style", self.communication_style)
            verbosity = personality.get("verbosity", self.verbosity)
        else:
            style = self.communication_style
            verbosity = self.verbosity

        adapted = text

        if verbosity == "concise" and len(adapted) > 200:
            sentences = adapted.split(". ")
            if len(sentences) > 3:
                adapted = ". ".join(sentences[:3]) + "."

        return adapted

    def get_wake_response(self) -> str:
        styles = {
            "casual": "Hey! What's up?",
            "formal": "Good day. How may I assist you?",
            "technical": "Shadow AI online. Awaiting input.",
        }
        return styles.get(self.communication_style, "Yes?")

    def get_thinking_response(self) -> str:
        styles = {
            "casual": "Let me think about that...",
            "formal": "I'm processing your request...",
            "technical": "Analyzing...",
        }
        return styles.get(self.communication_style, "Thinking...")


class ShadowVoice:
    """
    Voice interface for Shadow AI.

    Connects the shadow's cognitive engine to the real world
    through voice, using the local Piper TTS + Whisper STT server.
    """

    def __init__(
        self,
        voice_server_url: str = "http://localhost:8001",
        default_voice: str = "en_US-ryan-high",
        stt_model: str = "whisper-base",
    ):
        self.tts = TTSEngine(base_url=voice_server_url, default_voice=default_voice)
        self.stt = STTEngine(base_url=voice_server_url, model=stt_model)
        self.intent_parser = IntentParser()
        self.personality = VoicePersonality()

        self.current_session: Optional[VoiceSession] = None
        self._sessions: Dict[str, VoiceSession] = {}
        self._sessions_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._state = VoiceState.IDLE
        self._command_handlers: Dict[VoiceIntent, Callable] = {}
        self._handlers_lock = threading.Lock()
        self._session_timeout: float = 1800.0
        self._max_sessions: int = 100

        self._stats = {
            "sessions_started": 0,
            "sessions_completed": 0,
            "commands_processed": 0,
            "words_spoken": 0,
            "words_heard": 0,
            "tts_calls": 0,
            "stt_calls": 0,
            "errors": 0,
        }

    def register_command_handler(self, intent: VoiceIntent, handler: Callable):
        with self._handlers_lock:
            self._command_handlers[intent] = handler

    async def start_session(self, voice: Optional[str] = None) -> VoiceSession:
        session = VoiceSession(voice=voice or self.tts.default_voice)
        session.state = VoiceState.LISTENING
        with self._sessions_lock:
            self._cleanup_expired_sessions()
            self._sessions[session.session_id] = session
            self.current_session = session
        self._state = VoiceState.LISTENING
        with self._stats_lock:
            self._stats["sessions_started"] += 1

        greeting = self.personality.get_wake_response()
        await self.speak(greeting, voice=voice)
        session.add_transcript("shadow", greeting)

        return session

    async def end_session(self) -> Optional[Dict]:
        if not self.current_session:
            return None

        farewell = "Talk to you later."
        await self.speak(farewell)
        self.current_session.add_transcript("shadow", farewell)
        self.current_session.end()

        with self._stats_lock:
            self._stats["sessions_completed"] += 1
        result = self.current_session.to_dict()
        self.current_session = None
        self._state = VoiceState.IDLE
        return result

    async def process_speech(self, transcript: str, audio_data: Optional[bytes] = None) -> Dict:
        if not self.current_session:
            with self._sessions_lock:
                self._cleanup_expired_sessions()
                self.current_session = VoiceSession()
                self._sessions[self.current_session.session_id] = self.current_session

        transcript = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', transcript).strip()[:5000]

        self._state = VoiceState.PROCESSING
        self.current_session.state = VoiceState.PROCESSING
        with self._stats_lock:
            self._stats["words_heard"] += len(transcript.split())

        intent, confidence = self.intent_parser.parse(transcript)
        entities = self.intent_parser.extract_entities(transcript)

        command = VoiceCommand(
            intent=intent,
            transcript=transcript,
            confidence=confidence,
            entities=entities,
        )
        self.current_session.commands.append(command)
        if len(self.current_session.commands) > self.current_session._MAX_COMMANDS:
            self.current_session.commands = self.current_session.commands[-self.current_session._MAX_COMMANDS:]
        self.current_session.add_transcript("user", transcript)
        with self._stats_lock:
            self._stats["commands_processed"] += 1

        handler = None
        with self._handlers_lock:
            handler = self._command_handlers.get(intent)
        if handler:
            try:
                if asyncio.iscoroutinefunction(handler):
                    result = await handler(command)
                else:
                    result = handler(command)
                return result
            except (ValueError, TypeError, RuntimeError, AttributeError, KeyError) as e:
                logger.error("Command handler error: %s", type(e).__name__)
                with self._stats_lock:
                    self._stats["errors"] += 1

        response_text = f"I heard you say: {transcript}"
        await self.speak(response_text)
        self.current_session.add_transcript("shadow", response_text)
        with self._stats_lock:
            self._stats["words_spoken"] += len(response_text.split())

        self._state = VoiceState.LISTENING
        self.current_session.state = VoiceState.LISTENING

        return {
            "command": command.to_dict(),
            "response": response_text,
            "session_id": self.current_session.session_id,
        }

    async def speak(self, text: str, voice: Optional[str] = None) -> bool:
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text).strip()[:5000]
        with self._stats_lock:
            self._stats["tts_calls"] += 1
        audio = await self.tts.speak(text, voice=voice)
        if audio:
            self._state = VoiceState.SPEAKING
            with self._stats_lock:
                self._stats["words_spoken"] += len(text.split())
            self._state = VoiceState.LISTENING
            return True
        return False

    async def listen(self, audio_data: bytes, language: Optional[str] = None) -> Optional[str]:
        with self._stats_lock:
            self._stats["stt_calls"] += 1
        self._state = VoiceState.LISTENING

        transcript = await self.stt.transcribe(audio_data, language=language)
        if transcript:
            with self._stats_lock:
                self._stats["words_heard"] += len(transcript.split())
            return transcript

        with self._stats_lock:
            self._stats["errors"] += 1
        return None

    async def check_voice_server(self) -> Dict:
        return await self.tts.health()

    async def list_voices(self) -> List[Dict]:
        return await self.tts.list_voices()

    def get_session(self, session_id: str) -> Optional[Dict]:
        session = self._sessions.get(session_id)
        return session.to_dict() if session else None

    def list_sessions(self, limit: int = 20) -> List[Dict]:
        with self._sessions_lock:
            sessions = sorted(self._sessions.values(), key=lambda s: s.started_at, reverse=True)
        return [s.to_dict() for s in sessions[:limit]]

    def get_state(self) -> Dict:
        with self._sessions_lock:
            current = self.current_session
            current_dict = current.to_dict() if current else None
        with self._stats_lock:
            stats_copy = dict(self._stats)
        return {
            "state": self._state.value,
            "current_session": current_dict,
            "stats": stats_copy,
        }

    def get_stats(self) -> Dict:
        with self._sessions_lock:
            sessions_count = len(self._sessions)
        with self._stats_lock:
            stats_copy = dict(self._stats)
        return {
            "state": self._state.value,
            "sessions_total": sessions_count,
            "stats": stats_copy,
        }

    async def close(self):
        await self.tts.close()
        await self.stt.close()

    def _cleanup_expired_sessions(self):
        now = time.monotonic()
        expired = [
            sid for sid, sess in self._sessions.items()
            if now - sess._start_time > self._session_timeout and sess.ended_at is None
        ]
        for sid in expired:
            self._sessions[sid].end()
        if len(self._sessions) > self._max_sessions:
            sorted_sessions = sorted(
                self._sessions.items(), key=lambda x: x[1]._start_time
            )
            self._sessions = dict(sorted_sessions[-(self._max_sessions):])