#!/usr/bin/env python3
"""Tests for Communication Observer"""

import pytest
from datetime import datetime, timedelta
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from phone.communication_observer import (
    CommunicationObserver,
    CallMetadata,
    MessageMetadata,
    NotificationMetadata
)

def test_observer_init():
    observer = CommunicationObserver()
    assert observer is not None
    assert len(observer.calls) == 0

def test_observe_call():
    observer = CommunicationObserver()
    
    call = CallMetadata(
        contact_id="contact_1",
        contact_name="John",
        phone_number="+1***",
        call_type="incoming",
        duration_seconds=60,
        timestamp=datetime.now(),
        device_id="test-phone"
    )
    
    observer.observe_call(call)
    assert len(observer.calls) == 1

def test_observe_message():
    observer = CommunicationObserver()
    
    msg = MessageMetadata(
        app="whatsapp",
        contact_id="contact_1",
        message_type="text",
        timestamp=datetime.now(),
        device_id="test-phone"
    )
    
    observer.observe_message(msg)
    assert len(observer.messages) == 1

def test_communication_summary():
    observer = CommunicationObserver()
    
    # Add calls
    for i in range(3):
        observer.observe_call(CallMetadata(
            contact_id=f"contact_{i}",
            contact_name=f"Contact {i}",
            phone_number="+1***",
            call_type="incoming",
            duration_seconds=60,
            timestamp=datetime.now() - timedelta(hours=i),
            device_id="test-phone"
        ))
    
    summary = observer.get_communication_summary(24)
    assert summary["calls"]["total"] == 3

def test_top_contacts():
    observer = CommunicationObserver()
    
    # Add calls to same contact
    for i in range(5):
        observer.observe_call(CallMetadata(
            contact_id="favorite_contact",
            contact_name="Favorite",
            phone_number="+1***",
            call_type="outgoing",
            duration_seconds=60,
            timestamp=datetime.now() - timedelta(hours=i),
            device_id="test-phone"
        ))
    
    # Add single call to another
    observer.observe_call(CallMetadata(
        contact_id="other_contact",
        contact_name="Other",
        phone_number="+1***",
        call_type="incoming",
        duration_seconds=120,
        timestamp=datetime.now(),
        device_id="test-phone"
    ))
    
    top = observer.get_top_contacts(limit=5)
    assert len(top) > 0
    assert top[0]["contact_id"] == "favorite_contact"

def test_prediction():
    observer = CommunicationObserver()
    
    # Add calls with pattern (every 2 hours)
    base_time = datetime.now() - timedelta(hours=10)
    for i in range(6):
        observer.observe_call(CallMetadata(
            contact_id="regular_contact",
            contact_name="Regular",
            phone_number="+1***",
            call_type="incoming",
            duration_seconds=60,
            timestamp=base_time + timedelta(hours=i*2),
            device_id="test-phone"
        ))
    
    prediction = observer.predict_next_call("regular_contact")
    assert prediction is not None
    assert "predicted_time" in prediction
    assert "confidence" in prediction

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
