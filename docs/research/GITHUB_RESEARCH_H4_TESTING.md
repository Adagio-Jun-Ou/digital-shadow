# GitHub Research Hour 4: Testing Frameworks

## Overview

Testing frameworks for building robust, well-tested Shadow AI components.

---

## pytest (13.7k stars)

**GitHub**: https://github.com/pytest-dev/pytest

### Core Features
- **Simple assertions**: No need for `self.assert*` methods
- **Auto-discovery**: Automatically finds test files and functions
- **Fixtures**: Modular, reusable test resources
- **Parametrization**: Run same test with different inputs
- **Markers**: Skip, xfail, slow, custom categorization
- **Plugins**: 1300+ external plugins available

### Key Patterns

```python
import pytest

@pytest.fixture
def temp_dir(tmp_path):
    """Create temp directory for tests."""
    return tmp_path / "test_data"

def test_simple_assertion(temp_dir):
    """Use plain assert statements."""
    assert temp_dir.exists()
    
@pytest.mark.parametrize("input,expected", [
    (1, 2),
    (2, 4),
    (3, 6),
])
def test_multiply(input, expected):
    """Test with multiple inputs."""
    assert input * 2 == expected

@pytest.mark.skip(reason="Not implemented yet")
def test_future():
    pass

@pytest.mark.asyncio
async def test_async():
    result = await some_async_function()
    assert result is not None
```

### Fixtures Reference
```python
# Built-in fixtures
tmp_path - Temporary directory (pathlib.Path)
tmp_path_factory - Factory for temp directories
capsys - Capture stdout/stderr
capfd - Capture file descriptors
monkeypatch - Modify objects/dicts/os.environ
request - Test request context
```

---

## Hypothesis (8.6k stars)

**GitHub**: https://github.com/HypothesisWorks/hypothesis

### Property-Based Testing
- Write tests that should pass for ALL inputs in a range
- Hypothesis finds edge cases automatically
- Shrinks failing examples to minimal reproduction

### Key Patterns

```python
from hypothesis import given, strategies as st

@given(st.integers(), st.integers())
def test_addition_commutative(x, y):
    """Property: a + b = b + a"""
    assert x + y == y + x

@given(st.lists(st.integers()))
def test_reversal_invariant(items):
    """Property: reverse twice = original"""
    assert list(reversed(list(reversed(items)))) == items

@given(st.text())
def test_string_properties(s):
    """Any string concatenated with empty = itself"""
    assert s + "" == s == "" + s

# Composite strategies
@st.composite
def user_dicts(draw):
    return {
        "name": draw(st.text(min_size=1)),
        "age": draw(st.integers(min_value=0, max_value=120)),
        "email": draw(st.from_regex(r"[a-z]+@[a-z]+\.[a-z]+")),
    }

@given(user_dicts())
def test_user_schema(user):
    assert isinstance(user["age"], int)
    assert 0 <= user["age"] <= 120
```

### Available Strategies
- `st.integers()`, `st.floats()`, `st.text()`
- `st.lists()`, `st.tuples()`, `st.dictionaries()`
- `st.from_regex()`, `st.datetimes()`
- `st.sampled_from()`, `st.one_of()`
- `st.builds()` for custom classes

---

## Faker (19.2k stars)

**GitHub**: https://github.com/joke2k/faker

### Test Data Generation
- Generate realistic fake data
- Multiple locales supported
- Works great with pytest fixtures

### Key Patterns

```python
from faker import Faker

fake = Faker()
fake.name()          # 'Lucy Cechtelar'
fake.address()       # '426 Jordy Lodge, Cartwrightshire'
fake.email()         # 'sandra@example.com'
fake.date()          # datetime object
fake.uuid4()         # UUID string
fake.sentence()      # Random sentence

# Locale support
fake_fr = Faker('fr_FR')
fake_jp = Faker('ja_JP')
fake_multi = Faker(['en_US', 'fr_FR', 'ja_JP'])

# Pytest integration
import pytest

@pytest.fixture
def fake():
    return Faker()

def test_user_creation(fake):
    user = {
        "name": fake.name(),
        "email": fake.email(),
        "address": fake.address(),
    }
    assert "@" in user["email"]

# Seeded for reproducibility
Faker.seed(12345)
```

### Provider Categories
- Person (name, email, phone)
- Address (street, city, country)
- Date/Time (date, datetime, timezone)
- Text (sentence, paragraph, text)
- Internet (email, url, ipv4)
- Company (company, job)
- Finance (credit_card, currency)
- Python (pystr, pylist)

---

## pytest-asyncio (1.6k stars)

**GitHub**: https://github.com/pytest-dev/pytest-asyncio

### Async Test Support
- Test asyncio coroutines
- Works with pytest fixtures
- Supports `async def test_*` functions

### Key Patterns

```python
import pytest
import asyncio

@pytest.mark.asyncio
async def test_async_function():
    """Test async function directly."""
    result = await some_async_function()
    assert result == expected

# Fixture with async setup/teardown
@pytest.fixture
async def async_resource():
    resource = await create_resource()
    yield resource
    await resource.cleanup()

@pytest.mark.asyncio
async def test_with_async_fixture(async_resource):
    assert async_resource.is_ready()

# Configure mode in pytest.ini
# [pytest]
# asyncio_mode = auto  # All async test functions are marked
# asyncio_mode = strict  # Require explicit @pytest.mark.asyncio
# asyncio_default_fixture_loop_scope = function
```

---

## Locust (27.7k stars)

**GitHub**: https://github.com/locustio/locust

### Load/Performance Testing
- Write load tests in Python
- Distributed testing support
- Web UI for monitoring
- Tests any protocol

### Key Patterns

```python
from locust import HttpUser, task, between

class MyUser(HttpUser):
    wait_time = between(1, 5)
    
    def on_start(self):
        """Run once per user on start."""
        self.client.post("/login", json={"user": "test"})
    
    @task(3)  # Weight of 3
    def view_item(self):
        self.client.get("/item?id=1")
    
    @task(1)  # Weight of 1
    def add_to_cart(self):
        self.client.post("/cart", json={"item_id": 1})

# Custom client for non-HTTP testing
class MyUser(User):
    @task
    def my_task(self):
        # Test any protocol
        pass
```

---

## Playwright (14.5k stars)

**GitHub**: https://github.com/microsoft/playwright-python

### E2E/UI Testing
- Cross-browser: Chromium, Firefox, WebKit
- Auto-wait for elements
- Screenshot/video capture
- Network interception

### Key Patterns

```python
from playwright.sync_api import sync_playwright

def test_page_navigation():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto("https://example.com")
        
        # Auto-waiting
        page.click("button#submit")
        
        # Assertions
        assert page.title() == "Expected Title"
        
        # Screenshots
        page.screenshot(path="screenshot.png")
        
        browser.close()

# Async version
from playwright.async_api import async_playwright

async def test_async():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto("https://example.com")
        await browser.close()

# With pytest-playwright
def test_with_fixture(page):
    page.goto("https://example.com")
    assert page.title() == "Example Domain"
    
    page.click("text=More information")
    page.wait_for_url("*/iana/*")
```

---

## Integration for Shadow AI

### Testing Strategy

#### 1. Unit Tests (pytest)
```python
# tests/unit/test_state_graph.py
def test_node_execution():
    graph = StateGraph()
    graph.add_node("start", start_node)
    result = graph.run()
    assert result.status == "completed"
```

#### 2. Property-Based Tests (hypothesis)
```python
# tests/property/test_block_memory.py
@given(st.lists(st.text()))
def test_memory_persistence(facts):
    memory = BlockMemory()
    for fact in facts:
        memory.add(fact)
    stored = memory.get_all()
    assert all(f in stored for f in facts)
```

#### 3. Integration Tests (pytest + fixtures)
```python
# tests/integration/test_unified.py
@pytest.fixture
async def shadow_system():
    system = UnifiedShadowSystem()
    await system.initialize()
    yield system
    await system.cleanup()

async def test_full_workflow(shadow_system):
    result = await shadow_system.process("Hello")
    assert result is not None
```

#### 4. E2E Tests (playwright)
```python
# tests/e2e/test_phone_api.py
def test_phone_api_dashboard(page):
    page.goto("http://localhost:8766")
    page.click("text=Devices")
    assert page.is_visible("text=Bluetooth")
```

#### 5. Load Tests (locust)
```python
# tests/load/locustfile.py
class ShadowUser(HttpUser):
    wait_time = between(1, 3)
    
    @task
    def process_query(self):
        self.client.post("/process", json={"query": "test"})
```

---

## Test Organization

```
tests/
├── unit/                    # Unit tests
│   ├── test_state_graph.py
│   ├── test_block_memory.py
│   └── test_knowledge_graph.py
├── property/                # Property-based tests
│   └── test_memory_properties.py
├── integration/             # Integration tests
│   ├── test_unified_shadow.py
│   └── test_phone_integration.py
├── e2e/                     # End-to-end tests
│   └── test_phone_api.py
└── load/                    # Performance tests
    └── locustfile.py
```

---

## Summary

| Framework | Stars | Purpose | Best For |
|-----------|-------|---------|----------|
| pytest | 13.7k | Unit/Integration Testing | General testing |
| Hypothesis | 8.6k | Property-Based Testing | Edge case discovery |
| Faker | 19.2k | Test Data Generation | Realistic fake data |
| pytest-asyncio | 1.6k | Async Testing | Async functions |
| Locust | 27.7k | Load Testing | Performance testing |
| Playwright | 14.5k | E2E/UI Testing | Browser automation |