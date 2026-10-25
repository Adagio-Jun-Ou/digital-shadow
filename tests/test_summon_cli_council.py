"""Tests for the summon_cli --council flag.

Mocks the HTTP layer so we don't need a running daemon.
"""

import sys
from unittest import mock

import pytest


@pytest.fixture
def cli():
    """Import the CLI module under test, with a clean sys.argv each time."""
    # Make sure summon/ is on sys.path.
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "summon"))
    import summon_cli  # noqa: E402
    return summon_cli


class TestCouncilFlag:
    def test_cmd_council_basic(self, cli, capsys):
        fake_response = {
            "verdict": "🏛️ **Council of Abilities**\nverdict body",
            "narrative": {
                "summary": "...",
                "signals": ["[tide] falling"],
                "themes": [],
                "health_score": 0.4,
                "contradictions": [],
                "recommendations": [
                    {"title": "Take a break", "rationale": "you need it",
                     "priority": 2, "triggering_abilities": ["tide", "mirror"]},
                ],
            },
            "members": {},
            "leader": None,
        }
        with mock.patch.object(cli, "_post", return_value=fake_response):
            rc = cli.cmd_council("what should I do?", abilities=None)
        captured = capsys.readouterr()
        assert rc == 0
        assert "Council of Abilities" in captured.out
        assert "Take a break" in captured.out
        assert "[2]" in captured.out

    def test_cmd_council_with_abilities(self, cli, capsys):
        captured_args = {}
        def fake_post(path, payload, timeout=120.0):
            captured_args["path"] = path
            captured_args["payload"] = payload
            return {"verdict": "x", "narrative": {"signals": [], "contradictions": [],
                                                    "recommendations": []}, "members": {}}
        with mock.patch.object(cli, "_post", side_effect=fake_post):
            cli.cmd_council("q", abilities=["tide", "compass"])
        assert captured_args["path"] == "/shikigami/council"
        assert captured_args["payload"]["abilities"] == ["tide", "compass"]
        assert captured_args["payload"]["question"] == "q"

    def test_cmd_council_handles_error(self, cli, capsys):
        with mock.patch.object(cli, "_post", return_value={"error": "boom"}):
            rc = cli.cmd_council("q", abilities=None)
        assert rc == 1
        captured = capsys.readouterr()
        assert "boom" in captured.err

    def test_council_prints_contradictions(self, cli, capsys):
        fake = {
            "verdict": "v",
            "narrative": {
                "signals": [],
                "themes": [],
                "health_score": 0.5,
                "contradictions": [
                    {"description": "Mood rising but alignment falling",
                     "abilities": ["tide", "compass"]},
                ],
                "recommendations": [],
            },
        }
        with mock.patch.object(cli, "_post", return_value=fake):
            cli.cmd_council("q", None)
        out = capsys.readouterr().out
        assert "CONTRADICTIONS" in out
        assert "Mood rising" in out

    def test_cmd_health_renders_table(self, cli, capsys):
        fake = {
            "registered": 2,
            "abilities": [
                {"type": "oracle", "name": "Oracle",
                 "count": 5, "successes": 4, "failures": 1,
                 "success_rate": 0.8, "p50_latency_s": 0.02,
                 "p95_latency_s": 0.15, "avg_latency_s": 0.05,
                 "last_invoked_ts": 1715900000.0},
                {"type": "tide", "name": "Tide",
                 "count": 0, "successes": 0, "failures": 0,
                 "success_rate": None, "p50_latency_s": None,
                 "p95_latency_s": None, "avg_latency_s": None,
                 "last_invoked_ts": None},
            ],
        }
        with mock.patch.object(cli, "_get", return_value=fake):
            rc = cli.cmd_health()
        out = capsys.readouterr().out
        assert rc == 0
        assert "oracle" in out
        assert "tide" in out
        assert "80%" in out
        # Tide with no data should render as dashes.
        tide_line = [l for l in out.splitlines() if l.startswith("tide")][0]
        assert "-" in tide_line

    def test_cmd_health_handles_error(self, cli, capsys):
        with mock.patch.object(cli, "_get", return_value={"error": "boom"}):
            rc = cli.cmd_health()
        assert rc == 1

    def test_cmd_health_empty(self, cli, capsys):
        with mock.patch.object(cli, "_get",
                                return_value={"abilities": [], "registered": 0}):
            rc = cli.cmd_health()
        assert rc == 0
        assert "No abilities" in capsys.readouterr().out

    def test_argparse_recognizes_health_flag(self, cli):
        with mock.patch.object(sys, "argv", ["summon", "--health"]):
            with mock.patch.object(cli, "cmd_health", return_value=0) as m:
                cli.main()
            m.assert_called_once()

    def test_argparse_recognizes_council_flag(self, cli):
        with mock.patch.object(sys, "argv",
                               ["summon", "--council", "test question",
                                "--ability", "tide", "--ability", "compass"]):
            captured = {}
            def fake_council(q, abilities):
                captured["q"] = q
                captured["abilities"] = abilities
                return 0
            with mock.patch.object(cli, "cmd_council", side_effect=fake_council):
                cli.main()
            assert captured["q"] == "test question"
            assert captured["abilities"] == ["tide", "compass"]
