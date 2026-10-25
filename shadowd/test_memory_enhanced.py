#!/usr/bin/env python3
import pytest
import sys
from pathlib import Path
import tempfile
sys.path.insert(0, str(Path(__file__).parent.parent))
from shadowd.memory_enhanced import EnhancedMemory

def test_init():
    temp = Path(tempfile.mkdtemp())
    memory = EnhancedMemory(temp)
    assert memory is not None
    assert len(memory.episodes) == 0

def test_add_episode():
    temp = Path(tempfile.mkdtemp())
    memory = EnhancedMemory(temp)
    episode = memory.add_episode("observation", "Test")
    assert episode.id is not None
    assert len(memory.episodes) == 1

def test_search():
    temp = Path(tempfile.mkdtemp())
    memory = EnhancedMemory(temp)
    memory.add_episode("commitment", "Call John")
    memory.add_episode("observation", "Had coffee")
    results = memory.search("John")
    assert len(results) == 1

def test_context():
    temp = Path(tempfile.mkdtemp())
    memory = EnhancedMemory(temp)
    memory.add_to_core("name", "User")
    memory.add_episode("commitment", "Task", importance=0.9)
    context = memory.get_context()
    assert "User" in context

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
