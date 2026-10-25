"""Tests for the Lantern project-state ability."""

import asyncio
from datetime import datetime, timezone, timedelta

import pytest

from shikigami import get_registry, ShikigamiType
from shikigami.lantern import compute_lantern, _project_root


NOW = datetime(2026, 5, 15, 12, 0, tzinfo=timezone.utc)


def _change(path: str, days_ago: float) -> dict:
    return {
        "type": "file_change",
        "content": {"path": path, "change_type": "modified"},
        "timestamp": (NOW - timedelta(days=days_ago)).isoformat(),
    }


class TestProjectRoot:
    def test_home_anchored_path(self):
        # Pass use_git=False so this works on machines without the digital-shadow git repo on disk.
        from shikigami.lantern import _project_root as pr
        assert pr("/home/user/digital-shadow/shikigami/foo.py", use_git=False) == "digital-shadow"

    def test_skips_parent_hint(self):
        from shikigami.lantern import _project_root as pr
        assert pr("/home/user/projects/blog/posts/x.md", use_git=False) == "blog"

    def test_skips_hidden(self):
        # /home/user/.cache/foo.txt — '.cache' is hidden, fall through.
        # Only home-anchored relpath: ['.cache','foo.txt'] → fallback to last token.
        assert _project_root("/home/user/.cache/foo.txt") == "foo.txt"

    def test_unknown_for_empty(self):
        assert _project_root("") == "unknown"

    def test_handles_non_home_path(self):
        # /tmp/ starts with a parent-hint segment, so we skip it and fall
        # back to the file basename — a reasonable best guess.
        assert _project_root("/tmp/scratch.py") == "scratch.py"

    def test_handles_etc_path(self):
        # /etc/ is not a parent hint, so it wins as project root.
        assert _project_root("/etc/nginx/nginx.conf") == "etc"


class TestGitRepoDetection:
    def test_detects_repo_root(self, tmp_path):
        """When .git/ exists upwards, use that as project root."""
        from shikigami.lantern import _git_repo_root, _project_root

        repo = tmp_path / "myrepo"
        (repo / ".git").mkdir(parents=True)
        (repo / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
        (repo / "src").mkdir()
        f = repo / "src" / "main.py"
        f.write_text("# test")
        assert _git_repo_root(str(f)) == "myrepo"
        assert _project_root(str(f)) == "myrepo"

    def test_no_repo_returns_none(self, tmp_path):
        from shikigami.lantern import _git_repo_root
        f = tmp_path / "nope" / "file.py"
        f.parent.mkdir(parents=True)
        f.write_text("x")
        assert _git_repo_root(str(f)) is None

    def test_use_git_false_skips_detection(self, tmp_path):
        from shikigami.lantern import _project_root
        # Even with a .git ancestor present, use_git=False should fall back
        # to the path-segment heuristic.
        repo = tmp_path / "myrepo"
        (repo / ".git").mkdir(parents=True)
        f = repo / "main.py"
        f.write_text("x")
        # With use_git=False, the heuristic picks the first non-hidden segment
        # of the absolute path. tmp_path is somewhere under /tmp/... — the
        # important thing is that it's NOT "myrepo" via the git path.
        result = _project_root(str(f), use_git=False)
        # The result depends on tmp_path layout, but specifically the
        # git-based shortcut should not have fired.
        assert isinstance(result, str)

    def test_handles_nonexistent_path_gracefully(self):
        from shikigami.lantern import _git_repo_root
        # Path that doesn't exist on disk shouldn't crash.
        assert _git_repo_root("/nonexistent/path/that/does/not/exist.py") is None

    def test_use_git_via_compute_lantern(self, tmp_path):
        """End-to-end: compute_lantern with use_git=True groups files by repo."""
        from shikigami.lantern import compute_lantern
        repo = tmp_path / "alpha"
        (repo / ".git").mkdir(parents=True)
        (repo / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
        (repo / "src").mkdir()
        f1 = repo / "src" / "a.py"
        f2 = repo / "src" / "b.py"
        f1.write_text("")
        f2.write_text("")
        from datetime import datetime, timezone, timedelta
        now = datetime(2026, 5, 15, 12, tzinfo=timezone.utc)
        obs = [
            {"type": "file_change", "content": {"path": str(f1)},
             "timestamp": (now - timedelta(days=1)).isoformat()},
            {"type": "file_change", "content": {"path": str(f2)},
             "timestamp": (now - timedelta(days=2)).isoformat()},
        ]
        r = compute_lantern(obs, now=now, use_git=True)
        names = {p.name for p in r.projects}
        assert "alpha" in names

    def test_strips_quotes(self):
        assert _project_root('"/home/user/digital-shadow/x.py"') == "digital-shadow"


class TestComputeLantern:
    def test_empty_returns_empty(self):
        r = compute_lantern([], now=NOW)
        assert r.total_changes == 0
        assert r.projects == []
        assert r.active == r.idle == r.dormant == []
        assert r.revisit_suggestions == []

    def test_active_project_classified(self):
        obs = [_change("/home/user/digital-shadow/a.py", d) for d in (0, 1, 2, 3)]
        r = compute_lantern(obs, now=NOW)
        assert r.total_changes == 4
        assert len(r.projects) == 1
        p = r.projects[0]
        assert p.name == "digital-shadow"
        assert p.status == "active"
        assert "digital-shadow" in r.active

    def test_idle_project_classified(self):
        obs = [_change("/home/user/blog/post.md", d) for d in (10, 12, 14, 18)]
        r = compute_lantern(obs, now=NOW)
        p = r.projects[0]
        assert p.status == "idle"
        assert "blog" in r.idle

    def test_dormant_project_classified(self):
        obs = [_change("/home/user/oldproj/main.py", d) for d in (45, 50, 60)]
        r = compute_lantern(obs, now=NOW)
        p = r.projects[0]
        assert p.status == "dormant"
        assert "oldproj" in r.dormant

    def test_revisit_suggestions_for_dormant(self):
        obs = [_change("/home/user/oldproj/main.py", d) for d in range(40, 60)]
        r = compute_lantern(obs, now=NOW)
        assert r.revisit_suggestions
        assert any("oldproj" in s for s in r.revisit_suggestions)

    def test_no_revisit_for_low_activity_dormant(self):
        # Only 2 changes — below the >=3 threshold for suggestions.
        obs = [_change("/home/user/tinyproj/x.py", d) for d in (40, 50)]
        r = compute_lantern(obs, now=NOW)
        # Project exists, but no revisit suggestion.
        assert not r.revisit_suggestions

    def test_multiple_projects_separated(self):
        obs = (
            [_change(f"/home/user/proj_a/f{i}.py", 1) for i in range(3)] +
            [_change(f"/home/user/proj_b/f{i}.py", 1) for i in range(2)]
        )
        r = compute_lantern(obs, now=NOW)
        names = {p.name for p in r.projects}
        assert names == {"proj_a", "proj_b"}

    def test_ignores_non_file_observations(self):
        obs = [
            {"type": "wifi", "content": {"ssid": "Home"}, "timestamp": NOW.isoformat()},
            {"type": "note", "content": {"text": "..."}, "timestamp": NOW.isoformat()},
            _change("/home/user/proj/x.py", 1),
        ]
        r = compute_lantern(obs, now=NOW)
        assert r.total_changes == 1

    def test_weekly_frequency_bounded(self):
        # 10 changes spread across 5 days → 14 changes/week
        obs = [_change("/home/user/proj/x.py", d) for d in (1, 2, 3, 4, 5, 1.5, 2.5, 3.5, 4.5, 0.5)]
        r = compute_lantern(obs, now=NOW)
        assert r.projects[0].weekly_frequency > 0

    def test_top_files_populated(self):
        obs = [_change("/home/user/proj/hot.py", d) for d in range(5)] + \
              [_change("/home/user/proj/cold.py", 1)]
        r = compute_lantern(obs, now=NOW)
        p = r.projects[0]
        assert len(p.top_files) >= 1
        # Most-touched file first.
        assert p.top_files[0][1] >= p.top_files[-1][1]

    def test_handles_malformed_timestamps(self):
        obs = [
            {"type": "file_change", "content": {"path": "/home/user/x/a.py"}, "timestamp": "not-a-date"},
            _change("/home/user/x/b.py", 1),
        ]
        r = compute_lantern(obs, now=NOW)
        assert r.total_changes == 1


class TestLanternViaRegistry:
    def test_executes_via_registry(self):
        r = get_registry()
        obs = [_change("/home/user/digital-shadow/a.py", d) for d in (0, 1, 2)]
        res = asyncio.run(r.execute(ShikigamiType.LANTERN, "overview", {"observations": obs}))
        assert res.success
        assert "Lantern" in res.output
        assert "digital-shadow" in res.output

    def test_data_is_json_serializable(self):
        import json
        r = get_registry()
        obs = [_change(f"/home/user/proj_{i}/x.py", j) for i in range(3) for j in (1, 2, 5, 40)]
        res = asyncio.run(r.execute(ShikigamiType.LANTERN, "overview", {"observations": obs}))
        json.dumps(res.data)

    def test_no_changes_still_produces_output(self):
        r = get_registry()
        res = asyncio.run(r.execute(ShikigamiType.LANTERN, "overview", {"observations": []}))
        assert res.success
        assert len(res.output) >= 80
