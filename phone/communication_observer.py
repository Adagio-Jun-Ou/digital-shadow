#!/usr/bin/env python3
"""
Communication Observer for Shadow AI
Monitors calls, messages (metadata only), notifications (privacy-safe)
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger("shadowd.communication")

@dataclass
class CallMetadata:
    """Call metadata (privacy-safe, no content)."""
    contact_id: Optional[str]  # Hashed identifier
    contact_name: Optional[str]  # User-defined name
    phone_number: Optional[str]  # Partially masked
    call_type: str  # incoming, outgoing, missed
    duration_seconds: int
    timestamp: datetime
    device_id: str

@dataclass  
class MessageMetadata:
    """Message metadata (privacy-safe, no content)."""
    app: str  # whatsapp, telegram, messenger, sms
    contact_id: Optional[str]
    message_type: str  # text, voice, media
    timestamp: datetime
    device_id: str
    # NOTE: No message content stored

@dataclass
class NotificationMetadata:
    """Notification metadata (privacy-safe)."""
    app: str
    category: str  # message, social, email, reminder
    timestamp: datetime
    dismissed: bool
    device_id: str

class CommunicationObserver:
    """
    Observes communication patterns (privacy-first).
    
    Tracks only metadata, never content.
    Used by:
    - Social Radar ability (predicting calls)
    - Pattern Sight ability (communication patterns)
    """
    
    def __init__(self):
        self.calls: List[CallMetadata] = []
        self.messages: List[MessageMetadata] = []
        self.notifications: List[NotificationMetadata] = []
        
        # Privacy-safe contact registry
        self.contacts: Dict[str, Dict] = {}
        
        # Pattern tracking
        self.call_patterns: Dict[str, List[datetime]] = {}
        self.message_patterns: Dict[str, List[datetime]] = {}
    
    MAX_CALLS = 500
    MAX_MESSAGES = 1000
    MAX_NOTIFICATIONS = 500
    MAX_PATTERN_TIMES = 200

    def observe_call(self, call: CallMetadata):
        """Record call metadata."""
        self.calls.append(call)
        if len(self.calls) > self.MAX_CALLS:
            self.calls = self.calls[-self.MAX_CALLS:]
        
        contact_key = call.contact_id or "unknown"
        if contact_key not in self.call_patterns:
            self.call_patterns[contact_key] = []
        self.call_patterns[contact_key].append(call.timestamp)
        if len(self.call_patterns[contact_key]) > self.MAX_PATTERN_TIMES:
            self.call_patterns[contact_key] = self.call_patterns[contact_key][-self.MAX_PATTERN_TIMES:]
        
        logger.info(f"Call observed: {call.call_type} from {contact_key}, {call.duration_seconds}s")
    
    def observe_message(self, message: MessageMetadata):
        """Record message metadata."""
        self.messages.append(message)
        if len(self.messages) > self.MAX_MESSAGES:
            self.messages = self.messages[-self.MAX_MESSAGES:]
        
        contact_key = message.contact_id or "unknown"
        if contact_key not in self.message_patterns:
            self.message_patterns[contact_key] = []
        self.message_patterns[contact_key].append(message.timestamp)
        if len(self.message_patterns[contact_key]) > self.MAX_PATTERN_TIMES:
            self.message_patterns[contact_key] = self.message_patterns[contact_key][-self.MAX_PATTERN_TIMES:]
        
        logger.info(f"Message observed: {message.app} from {contact_key}")
    
    def observe_notification(self, notification: NotificationMetadata):
        """Record notification metadata."""
        self.notifications.append(notification)
        if len(self.notifications) > self.MAX_NOTIFICATIONS:
            self.notifications = self.notifications[-self.MAX_NOTIFICATIONS:]
        logger.info(f"Notification observed: {notification.app} ({notification.category})")
    
    def predict_next_call(self, contact_id: str = None) -> Optional[Dict]:
        """
        Predict next call time based on patterns.
        
        Used by Social Radar ability.
        
        Returns:
            Predicted time and confidence, or None
        """
        if not self.call_patterns:
            return None
        
        # Find contact with most calls
        target_patterns = self.call_patterns
        if contact_id:
            target_patterns = {contact_id: self.call_patterns.get(contact_id, [])}
        
        if not target_patterns:
            return None
        
        # Find pattern
        all_times = []
        for times in target_patterns.values():
            all_times.extend(times)
        
        if len(all_times) < 3:
            return None  # Need at least 3 calls
        
        # Sort times
        sorted_times = sorted(all_times[-10:])  # Last 10 calls
        
        # Calculate average interval
        intervals = []
        for i in range(1, len(sorted_times)):
            interval = (sorted_times[i] - sorted_times[i-1]).total_seconds()
            intervals.append(interval)
        
        if not intervals:
            return None
        
        avg_interval_seconds = sum(intervals) / len(intervals)
        
        # Predict next call
        last_call = sorted_times[-1]
        predicted_next = last_call + timedelta(seconds=avg_interval_seconds)
        
        # Confidence based on pattern consistency
        std_dev = sum(abs(i - avg_interval_seconds) for i in intervals) / len(intervals)
        confidence = max(0, 1 - (std_dev / avg_interval_seconds)) if avg_interval_seconds > 0 else 0
        
        return {
            "predicted_time": predicted_next.isoformat(),
            "confidence": confidence,
            "avg_interval_hours": avg_interval_seconds / 3600,
            "sample_size": len(all_times)
        }
    
    def get_communication_summary(self, hours: int = 24) -> Dict:
        """
        Get communication summary for time period.
        
        Returns:
            Summary of calls, messages, notifications
        """
        cutoff = datetime.now() - timedelta(hours=hours)
        
        recent_calls = [c for c in self.calls if c.timestamp > cutoff]
        recent_messages = [m for m in self.messages if m.timestamp > cutoff]
        recent_notifications = [n for n in self.notifications if n.timestamp > cutoff]
        
        # Calculate totals
        total_call_time = sum(c.duration_seconds for c in recent_calls)
        
        # Unique contacts
        call_contacts = set(c.contact_id for c in recent_calls if c.contact_id)
        message_contacts = set(m.contact_id for m in recent_messages if m.contact_id)
        
        return {
            "period_hours": hours,
            "calls": {
                "total": len(recent_calls),
                "total_duration_seconds": total_call_time,
                "unique_contacts": len(call_contacts),
                "incoming": len([c for c in recent_calls if c.call_type == "incoming"]),
                "outgoing": len([c for c in recent_calls if c.call_type == "outgoing"]),
                "missed": len([c for c in recent_calls if c.call_type == "missed"])
            },
            "messages": {
                "total": len(recent_messages),
                "unique_contacts": len(message_contacts),
                "by_app": self._count_by_app(recent_messages)
            },
            "notifications": {
                "total": len(recent_notifications),
                "dismissed": len([n for n in recent_notifications if n.dismissed]),
                "by_app": self._count_by_app(recent_notifications)
            }
        }
    
    def _count_by_app(self, items: List) -> Dict[str, int]:
        """Count items by app."""
        counts = {}
        for item in items:
            app = getattr(item, 'app', 'unknown')
            counts[app] = counts.get(app, 0) + 1
        return counts
    
    def get_top_contacts(self, limit: int = 10) -> List[Dict]:
        """
        Get top contacts by communication frequency.
        
        Returns:
            List of contacts with call/message counts
        """
        contact_stats = {}
        
        # Count calls
        for call in self.calls:
            if call.contact_id:
                if call.contact_id not in contact_stats:
                    contact_stats[call.contact_id] = {
                        "contact_id": call.contact_id,
                        "contact_name": call.contact_name,
                        "calls": 0,
                        "messages": 0,
                        "total_call_duration": 0
                    }
                contact_stats[call.contact_id]["calls"] += 1
                contact_stats[call.contact_id]["total_call_duration"] += call.duration_seconds
        
        # Count messages
        for msg in self.messages:
            if msg.contact_id:
                if msg.contact_id not in contact_stats:
                    contact_stats[msg.contact_id] = {
                        "contact_id": msg.contact_id,
                        "contact_name": None,
                        "calls": 0,
                        "messages": 0,
                        "total_call_duration": 0
                    }
                contact_stats[msg.contact_id]["messages"] += 1
        
        # Sort by total interactions
        sorted_contacts = sorted(
            contact_stats.values(),
            key=lambda x: x["calls"] + x["messages"],
            reverse=True
        )
        
        return sorted_contacts[:limit]

if __name__ == "__main__":
    # Test communication observer
    observer = CommunicationObserver()
    
    from datetime import datetime, timedelta, timedelta
    
    # Add test calls
    for i in range(5):
        observer.observe_call(CallMetadata(
            contact_id=f"contact_{i % 3}",
            contact_name=f"Contact {i % 3}",
            phone_number="+1***",
            call_type="incoming" if i % 2 == 0 else "outgoing",
            duration_seconds=60 * (i + 1),
            timestamp=datetime.now() - timedelta(hours=i),
            device_id="test-phone"
        ))
    
    # Add test messages
    for i in range(10):
        observer.observe_message(MessageMetadata(
            app="whatsapp",
            contact_id=f"contact_{i % 3}",
            message_type="text",
            timestamp=datetime.now() - timedelta(hours=i),
            device_id="test-phone"
        ))
    
    # Test prediction
    prediction = observer.predict_next_call("contact_1")
    print(f"Prediction: {prediction}")
    
    # Get summary
    summary = observer.get_communication_summary(24)
    print(f"Summary: {summary}")
    
    # Get top contacts
    top = observer.get_top_contacts(5)
    print(f"Top contacts: {top}")
