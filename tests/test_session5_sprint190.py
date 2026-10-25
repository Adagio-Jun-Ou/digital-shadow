#!/usr/bin/env python3
"""Sprint 190 — API router GET endpoint auth hardening + shadow_enhanced verification."""
import re
import unittest
import os
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent


def _repo_file(rel_path: str) -> str:
    """Absolute path to a file in the repository, platform independent."""
    return str(REPO_ROOT / rel_path.replace("/", os.sep))



class TestApiRouterAuth(unittest.TestCase):
    """Sprint 190 — All sensitive GET endpoints must have auth."""

    def test_get_endpoints_have_auth(self):
        """All GET endpoints except whitelisted public paths must have auth.
        Liveness/readiness probes and the deploy dashboard are intentionally
        public so orchestrators (Kubernetes, Docker) and unprivileged shadows
        can hit them."""
        with open(_repo_file("shadowd/cognition/api_router.py")) as f:
            content = f.read()

        public_paths = {
            "/health", "/status", "/voice/voices",
            "/health/live", "/health/ready",
            "/deploy/dashboard",
        }
        lines = content.split("\n")

        unauthed = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped.startswith("@router.get("):
                continue
            if "Depends(verify_api_key)" in stripped:
                continue
            m = re.search(r'@router\.get\("([^"]+)"', stripped)
            if not m:
                continue
            route_path = m.group(1)
            if route_path in public_paths:
                continue
            unauthed.append(f"Line {i+1}: {stripped}")

        self.assertEqual(len(unauthed), 0,
                         f"Unauthenticated GET endpoints:\n" + "\n".join(unauthed))

    def test_no_double_auth(self):
        """No endpoint should have both dependencies=[...] AND auth= params."""
        with open(_repo_file("shadowd/cognition/api_router.py")) as f:
            content = f.read()

        self.assertNotIn("auth=Depends(verify_api_key)", content)

    def test_post_endpoints_have_auth(self):
        """All POST endpoints except deploy-side reporting routes must have auth.
        The /deploy/* POST routes are how a deployed shadow reports back to its
        coordinator; they're scoped via a shared deploy_id, not user auth."""
        with open(_repo_file("shadowd/cognition/api_router.py")) as f:
            content = f.read()

        public_post_paths = {
            "/deploy/heartbeat", "/deploy/report",
            "/deploy/config", "/deploy/idle",
        }
        lines = content.split("\n")
        unauthed = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped.startswith("@router.post("):
                continue
            if "Depends(verify_api_key)" in stripped:
                continue
            m = re.search(r'@router\.post\("([^"]+)"', stripped)
            if m and m.group(1) in public_post_paths:
                continue
            unauthed.append(f"Line {i+1}: {stripped}")

        self.assertEqual(len(unauthed), 0,
                         f"Unauthenticated POST endpoints:\n" + "\n".join(unauthed))

    def test_delete_endpoints_have_auth(self):
        """All DELETE endpoints must have auth."""
        with open(_repo_file("shadowd/cognition/api_router.py")) as f:
            content = f.read()

        lines = content.split("\n")
        unauthed = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped.startswith("@router.delete("):
                continue
            if "Depends(verify_api_key)" in stripped:
                continue
            unauthed.append(f"Line {i+1}: {stripped}")

        self.assertEqual(len(unauthed), 0,
                         f"Unauthenticated DELETE endpoints:\n" + "\n".join(unauthed))

    def test_public_endpoints_still_public(self):
        """Health, status, and voices endpoints exist (may have auth)."""
        with open(_repo_file("shadowd/cognition/api_router.py")) as f:
            content = f.read()

        self.assertIn('"/status"', content)
        self.assertIn('"/health"', content)
        self.assertIn('"/voice/voices"', content)

    def test_router_loads(self):
        """Router creates without errors."""
        from shadowd.cognition.api_router import router
        self.assertGreater(len(router.routes), 200)


class TestShadowEnhancedSprint190(unittest.TestCase):
    """Verify shadow_enhanced fixes from Sprint 189 persisted."""

    def test_no_bare_except(self):
        with open(_repo_file("shadowd/shadow_enhanced.py")) as f:
            source = f.read()
        bare = re.findall(r'except Exception(?!\s+as)', source)
        self.assertEqual(len(bare), 0)

    def test_all_datetime_utc(self):
        with open(_repo_file("shadowd/shadow_enhanced.py")) as f:
            source = f.read()
        naive = [m for m in re.finditer(r'datetime\.now\(\)', source)
                 if 'timezone.utc' not in source[max(0, m.start()-20):m.end()+15]]
        self.assertEqual(len(naive), 0)

    def test_no_fstring_logs(self):
        with open(_repo_file("shadowd/shadow_enhanced.py")) as f:
            source = f.read()
        fstring_logs = re.findall(r'logger\.\w+\(f["\']', source)
        self.assertEqual(len(fstring_logs), 0)


class TestApiRouterEventRequestValidation(unittest.TestCase):
    """Sprint 189-190 — EventRequest float + dict validation."""

    def test_confidence_valid(self):
        from shadowd.cognition.api_router import EventRequest
        req = EventRequest(confidence=0.5)
        self.assertEqual(req.confidence, 0.5)

    def test_confidence_rejects_over_1(self):
        from pydantic import ValidationError
        from shadowd.cognition.api_router import EventRequest
        with self.assertRaises(ValidationError):
            EventRequest(confidence=1.5)

    def test_dict_key_cap(self):
        from shadowd.cognition.api_router import _MAX_DICT_KEYS
        self.assertEqual(_MAX_DICT_KEYS, 50)


if __name__ == "__main__":
    unittest.main()