#!/usr/bin/env python3
"""
Tests for Phone Integration API
"""

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from phone.phone_api_v2 import app

client = TestClient(app)

def test_root():
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "title" in data or "name" in data or "status" in data

def test_pair_device():
    """Test device pairing."""
    response = client.post("/pair", json={
        "device_id": "test-phone-123",
        "device_name": "Test Phone",
        "device_type": "android"
    })
    assert response.status_code == 200
    data = response.json()
    assert data["success"] == True
    assert "api_key" in data

def test_observe_location():
    """Test location observation (requires a paired device + X-API-Key)."""
    pair = client.post("/pair", json={
        "device_id": "test-phone-123", "device_name": "Test Phone", "device_type": "android",
    })
    assert pair.status_code == 200
    api_key = pair.json()["api_key"]
    response = client.post(
        "/observe/location",
        json={
            "latitude": 37.7749,
            "longitude": -122.4194,
            "accuracy": 10.0,
            "timestamp": "2026-04-04T00:00:00",
            "device_id": "test-phone-123",
        },
        headers={"X-API-Key": api_key},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "observed"

def test_list_devices():
    """Test device listing."""
    response = client.get("/devices")
    assert response.status_code == 200
    data = response.json()
    assert "total" in data

def test_health():
    """Test health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])