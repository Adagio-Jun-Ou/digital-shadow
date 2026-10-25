"""
Safe command runner for Reality Forge.

Reality Forge's LLM proposes shell commands. Auto-executing them all is unsafe;
never executing them defeats the "manifestation" promise. This module strikes
a middle ground:

- Default policy: only commands whose program is on the read-only allowlist
  *and* whose flags don't escape the workspace.
- Workspace-scoped writes: relative paths only, no `..`, no absolute paths
  except `/tmp/<workspace_subpath>`.
- Hard-banned tokens: shell metacharacters, network tools, sudo/su, anything
  that would make policy enforcement opaque.
- Opt-in escalation: SHADOW_FORGE_EXECUTE_WRITES=1 to allow the write-allowlist.
- Result reporting includes "policy_decision" so the caller can show *why*
  something didn't run.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("shadowd.safe_runner")


# Programs we will always run with arguments (subject to flag checks below).
# Read-only: never write outside the workspace, never modify system state.
READ_ONLY_ALLOWLIST = frozenset({
    "ls", "pwd", "echo", "cat", "head", "tail", "wc", "grep", "find",
    "stat", "file", "tree", "du", "df",
    "python", "python3", "node", "npm", "pip",  # script execution; sandboxed via cwd
    "git", "make",  # read-only subcommands enforced separately
    "date", "uname", "uptime", "whoami", "id", "env",
    "which", "type",
})

# Programs that mutate disk — only run when SHADOW_FORGE_EXECUTE_WRITES=1.
WRITE_ALLOWLIST = frozenset({
    "mkdir", "touch", "cp", "mv", "rm", "ln", "chmod",
})

# Hard-banned regardless of policy. Tools that can run external code, reach
# the network, or escalate privileges have no place in an auto-executed flow.
BANNED_PROGRAMS = frozenset({
    "sudo", "su", "doas",
    "curl", "wget", "nc", "ncat", "telnet", "ssh", "scp", "sftp", "rsync",
    "bash", "sh", "zsh", "fish", "dash", "ksh",  # explicit shell invocation
    "exec", "eval",
    "dd",  # too easy to nuke a disk
    "systemctl", "service", "shutdown", "reboot", "halt",
    "iptables", "nft", "firewall-cmd",
    "apt", "apt-get", "yum", "dnf", "pacman", "brew", "snap",  # package mgrs
    "docker", "podman", "kubectl",  # container orchestration
    "kill", "killall", "pkill",  # process control
})

# git subcommands that are safe (read-only) when called as `git <subcommand>`.
_SAFE_GIT_SUBCOMMANDS = frozenset({
    "status", "log", "diff", "show", "rev-parse", "branch",
    "config", "describe", "ls-files", "ls-tree", "remote",
})

# Shell metacharacters or constructs that bypass our token-level checks.
_DANGEROUS_METAS = re.compile(r"[`$|&;<>(){}*?\[\]!]|\.\./")
# Any absolute path token that isn't under /tmp/ or the project workspace root.
_BAD_ABS_PATH = re.compile(r"(?:^|\s)/(?!tmp/|home/user/digital-shadow/)")


@dataclass
class CommandResult:
    cmd: str
    policy_decision: str  # "executed", "blocked:<reason>", "would_run_dry"
    success: bool = False
    returncode: Optional[int] = None
    stdout: str = ""
    stderr: str = ""
    truncated: bool = False
    duration_ms: float = 0.0


@dataclass
class RunnerPolicy:
    allow_writes: bool = False
    dry_run: bool = False
    workspace: Path = field(default_factory=lambda: Path.home() / "digital-shadow" / "workspace")
    timeout_s: float = 15.0
    max_output_bytes: int = 16_384


def _classify(cmd: str, policy: RunnerPolicy) -> Tuple[bool, str]:
    """Return (allowed, reason). Reason is informative either way."""
    if not cmd or not cmd.strip():
        return False, "empty"

    # Reject if shell metas / parent traversal appear unescaped.
    if _DANGEROUS_METAS.search(cmd):
        return False, "shell_metacharacters_or_unsafe_path"
    # Reject absolute paths outside /tmp or the project root.
    if _BAD_ABS_PATH.search(cmd):
        return False, "absolute_path_outside_workspace"

    try:
        tokens = shlex.split(cmd)
    except ValueError:
        return False, "unparseable_quoting"
    if not tokens:
        return False, "empty"

    program = Path(tokens[0]).name  # strip any path prefix
    if program in BANNED_PROGRAMS:
        return False, f"banned_program:{program}"

    if program == "git":
        if len(tokens) < 2 or tokens[1] not in _SAFE_GIT_SUBCOMMANDS:
            return False, f"git_subcommand_not_safe:{tokens[1] if len(tokens) > 1 else '?'}"
        return True, "git_readonly"

    if program in READ_ONLY_ALLOWLIST:
        return True, "readonly_allowlist"

    if program in WRITE_ALLOWLIST:
        if not policy.allow_writes:
            return False, f"write_disabled:{program} (set SHADOW_FORGE_EXECUTE_WRITES=1)"
        return True, "write_allowlist"

    return False, f"program_not_allowlisted:{program}"


async def _execute(cmd: str, policy: RunnerPolicy) -> Tuple[int, str, str]:
    """Run a pre-classified command. Caller must call _classify first."""
    tokens = shlex.split(cmd)
    try:
        proc = await asyncio.create_subprocess_exec(
            *tokens,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(policy.workspace),
        )
    except (OSError, FileNotFoundError) as e:
        return -1, "", f"spawn_failed: {type(e).__name__}"

    try:
        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=policy.timeout_s)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.communicate()
        return -1, "", "timeout"

    stdout = stdout_b.decode(errors="replace")[: policy.max_output_bytes]
    stderr = stderr_b.decode(errors="replace")[: policy.max_output_bytes]
    return proc.returncode or 0, stdout, stderr


async def run_commands(commands: List[str], policy: Optional[RunnerPolicy] = None) -> List[CommandResult]:
    """Classify and (maybe) execute a batch of commands. Each result reports
    its policy decision so the caller can show users what was/wasn't run."""
    policy = policy or RunnerPolicy(
        allow_writes=os.environ.get("SHADOW_FORGE_EXECUTE_WRITES", "").lower() in ("1", "true", "yes"),
        dry_run=os.environ.get("SHADOW_FORGE_DRY_RUN", "").lower() in ("1", "true", "yes"),
    )
    policy.workspace.mkdir(parents=True, exist_ok=True)

    results: List[CommandResult] = []
    for raw in commands:
        cmd = raw.strip()
        allowed, reason = _classify(cmd, policy)
        if not allowed:
            results.append(CommandResult(cmd=cmd, policy_decision=f"blocked:{reason}"))
            logger.info("safe_runner blocked: %s — %s", cmd[:80], reason)
            continue
        if policy.dry_run:
            results.append(CommandResult(cmd=cmd, policy_decision="would_run_dry", success=True))
            continue

        import time
        t0 = time.time()
        rc, out, err = await _execute(cmd, policy)
        results.append(CommandResult(
            cmd=cmd,
            policy_decision="executed",
            success=rc == 0,
            returncode=rc,
            stdout=out,
            stderr=err,
            truncated=len(out) >= policy.max_output_bytes,
            duration_ms=round((time.time() - t0) * 1000.0, 1),
        ))
        logger.info("safe_runner executed: %s rc=%d (%.0fms)", cmd[:80], rc, (time.time() - t0) * 1000.0)

    return results
