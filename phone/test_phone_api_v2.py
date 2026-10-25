#!/usr/bin/env python3
"""Tests for Phone API v2"""

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from phone.phone_api_v2 import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data

def test_pair_device():
    response = client.post("/pair", json={
        "device_id": "test-phone-123",
        "device_name": "Test Phone",
        "device_type": "android"
    })
    assert response.status_code == 200
    data = response.json()
    assert data["success"] == True

def test_observe_location():
    # /pair first to mint a real API key, then use it for the protected route.
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
    response = client.get("/devices")
    assert response.status_code == 200
    data = response.json()
    assert "total" in data

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
