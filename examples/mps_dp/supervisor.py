#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Lifecycle supervisor for same-GPU MPS replicas.

The router owns routing state. This process owns only replica process
lifecycle and keeps a worker disabled in the router until the replacement has
passed its health, KV-capacity, and MPS-attachment gates.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import fcntl
import json
import os
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any


@dataclasses.dataclass(frozen=True)
class ReplicaSpec:
    index: int
    port: int
    log: str
    expected_tokens: int | None
    command: list[str]
    env: dict[str, str]
    cwd: str | None = None


@dataclasses.dataclass(frozen=True)
class ReplicaRecord:
    index: int
    pid: int
    pgid: int
    port: int
    log: str
    leader_start: str

    @classmethod
    def from_tsv(cls, line: str) -> "ReplicaRecord":
        fields = line.rstrip("\n").split("\t")
        if len(fields) != 6:
            raise ValueError(f"invalid replicas.tsv row: {line!r}")
        return cls(
            index=int(fields[0]),
            pid=int(fields[1]),
            pgid=int(fields[2]),
            port=int(fields[3]),
            log=fields[4],
            leader_start=fields[5],
        )

    def to_tsv(self) -> str:
        return (
            f"{self.index}\t{self.pid}\t{self.pgid}\t{self.port}\t"
            f"{self.log}\t{self.leader_start}\n"
        )


@dataclasses.dataclass(frozen=True)
class PendingRestart:
    schema_version: int
    index: int
    port: int
    log: str
    token: str
    created_unix_s: float
    pid: int | None = None
    pgid: int | None = None
    leader_start: str | None = None

    @classmethod
    def from_path(cls, path: Path) -> "PendingRestart":
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):
            raise ValueError(f"invalid pending restart record: {path}")
        record = cls(**payload)
        if record.schema_version != 1:
            raise ValueError(
                f"unsupported pending restart schema {record.schema_version}: {path}"
            )
        if record.index < 0 or record.port <= 0 or not record.token:
            raise ValueError(f"invalid pending restart identity: {path}")
        identity = (record.pid, record.pgid, record.leader_start)
        if any(value is not None for value in identity) and not all(
            value is not None for value in identity
        ):
            raise ValueError(f"incomplete pending restart process identity: {path}")
        return record

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=2, sort_keys=True) + "\n"

    def as_replica_record(self) -> ReplicaRecord | None:
        if self.pid is None or self.pgid is None or self.leader_start is None:
            return None
        return ReplicaRecord(
            index=self.index,
            pid=self.pid,
            pgid=self.pgid,
            port=self.port,
            log=self.log,
            leader_start=self.leader_start,
        )


class ReplicaSupervisor:
    def __init__(
        self,
        *,
        state: Path,
        launch_script: Path,
        router_url: str,
        interval_secs: float,
        health_failure_threshold: int,
        health_timeout_secs: float,
        health_tries: int,
        health_interval_secs: float,
    ) -> None:
        self.state = state.resolve()
        self.launch_script = launch_script.resolve()
        self.router_url = router_url.rstrip("/")
        self.interval_secs = interval_secs
        self.health_failure_threshold = health_failure_threshold
        self.health_timeout_secs = health_timeout_secs
        self.health_tries = health_tries
        self.health_interval_secs = health_interval_secs
        self._health_failures: dict[int, int] = {}
        self._last_attempted_index: int | None = None

    @contextlib.contextmanager
    def singleton_lock(self) -> Iterator[None]:
        lock_path = self.state / "supervisor.lock"
        lock_file = lock_path.open("a+", encoding="utf-8")
        try:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(
                    f"a supervisor is already active for {self.state}"
                ) from exc
            lock_file.seek(0)
            lock_file.truncate()
            lock_file.write(f"{os.getpid()}\n")
            lock_file.flush()
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            lock_file.close()

    def run(self) -> None:
        with self.singleton_lock():
            cleanup_pending_restarts(self.state)
            self._record_event("supervisor_started")
            while True:
                try:
                    self.check_once()
                except Exception as exc:
                    self._record_event(
                        "supervisor_check_failed",
                        error=f"{type(exc).__name__}: {exc}",
                    )
                time.sleep(self.interval_secs)

    def check_once(self) -> None:
        records = self._load_replica_records()
        for spec in self._sweep_order(self._load_specs()):
            record = records.get(spec.index)
            healthy = (
                record is not None
                and self._identity_matches(record)
                and self._health_ok(spec.port)
            )
            if healthy:
                self._health_failures[spec.index] = 0
                continue

            failures = self._health_failures.get(spec.index, 0) + 1
            self._health_failures[spec.index] = failures
            if failures < self.health_failure_threshold:
                continue

            if record is None:
                raise RuntimeError(f"replica {spec.index} has no process record")
            # Record the attempt before it can raise: a sweep aborted by a failed
            # restart must still resume at the next replica, or one permanently
            # unrestartable member starves every other member behind it.
            self._last_attempted_index = spec.index
            self.restart_replica(spec, record)
            self._health_failures[spec.index] = 0
            records[spec.index] = self._load_replica_records()[spec.index]

    def _sweep_order(self, specs: list[ReplicaSpec]) -> list[ReplicaSpec]:
        # Round-robin: resume at the replica after the last one this supervisor
        # attempted to restart, so repeated failures rotate instead of pinning
        # every sweep to the lowest index.
        if self._last_attempted_index is None:
            return specs
        offset = next(
            (
                position + 1
                for position, spec in enumerate(specs)
                if spec.index == self._last_attempted_index
            ),
            0,
        )
        return specs[offset:] + specs[:offset]

    def restart_replica(self, spec: ReplicaSpec, old: ReplicaRecord) -> None:
        # This process holds the singleton lock for its full lifetime. Restarts
        # therefore cannot overlap CUDA-graph capture or memory profiling.
        self._record_event("restart_started", replica=spec.index, old_pid=old.pid)
        replacement: ReplicaRecord | None = None
        pending: PendingRestart | None = None
        try:
            self._set_router_disabled(spec, True)
            self._terminate_replica(old)
            self._wait_for_port_release(spec.port)
            log_offset = Path(spec.log).stat().st_size if Path(spec.log).exists() else 0
            pending = self._prepare_pending_restart(spec)
            replacement = self._launch_replica(spec, pending)
            self._replace_replica_record(replacement)
            self._clear_pending_restart(pending)
            pending = None
            self._wait_for_health(spec, replacement)
            self._validate_kv_capacity(spec, log_offset=log_offset)
            self._verify_mps_attach(spec)
            self._set_router_disabled(spec, False)
        except Exception as exc:
            # A healthy HTTP endpoint is not sufficient to re-enter the pool.
            # If any post-launch gate (including router re-registration) fails,
            # stop the replacement so the next check retries the full chain.
            if replacement is not None:
                self._terminate_replica(replacement)
            if pending is not None:
                cleanup_pending_restarts(self.state, indices={spec.index})
            self._record_event(
                "restart_failed",
                replica=spec.index,
                error=f"{type(exc).__name__}: {exc}",
            )
            raise

        self._record_event(
            "restart_completed",
            replica=spec.index,
            new_pid=replacement.pid,
        )

    def _load_specs(self) -> list[ReplicaSpec]:
        payload = json.loads((self.state / "replica_specs.json").read_text())
        if not isinstance(payload, list):
            raise ValueError("replica_specs.json must contain a list")
        specs = [ReplicaSpec(**item) for item in payload]
        if len({spec.index for spec in specs}) != len(specs):
            raise ValueError("replica_specs.json contains duplicate indices")
        return sorted(specs, key=lambda spec: spec.index)

    def _load_replica_records(self) -> dict[int, ReplicaRecord]:
        records = [
            ReplicaRecord.from_tsv(line)
            for line in (self.state / "replicas.tsv").read_text().splitlines()
            if line
        ]
        if len({record.index for record in records}) != len(records):
            raise ValueError("replicas.tsv contains duplicate indices")
        return {record.index: record for record in records}

    def _replace_replica_record(self, replacement: ReplicaRecord) -> None:
        records = self._load_replica_records()
        if replacement.index not in records:
            raise ValueError(f"replica {replacement.index} has no existing record")
        records[replacement.index] = replacement
        content = "".join(records[index].to_tsv() for index in sorted(records))
        _atomic_write_text(self.state / "replicas.tsv", content)

    def _prepare_pending_restart(self, spec: ReplicaSpec) -> PendingRestart:
        pending = PendingRestart(
            schema_version=1,
            index=spec.index,
            port=spec.port,
            log=spec.log,
            token=uuid.uuid4().hex,
            created_unix_s=time.time(),
        )
        path = _pending_restart_path(self.state, spec.index)
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if path.exists():
            raise RuntimeError(
                f"replica {spec.index} has an unreconciled pending restart"
            )
        _atomic_write_text(path, pending.to_json())
        self._record_event(
            "pending_restart_prepared",
            replica=spec.index,
            token=pending.token,
        )
        return pending

    def _clear_pending_restart(self, pending: PendingRestart) -> None:
        path = _pending_restart_path(self.state, pending.index)
        if path.exists():
            current = PendingRestart.from_path(path)
            if current.token != pending.token:
                raise RuntimeError(
                    f"pending restart token changed for replica {pending.index}"
                )
            path.unlink()
        self._record_event(
            "pending_restart_committed",
            replica=pending.index,
            token=pending.token,
        )

    def _identity_matches(self, record: ReplicaRecord) -> bool:
        return _record_identity_matches(record)

    def _health_ok(self, port: int) -> bool:
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health",
                timeout=self.health_timeout_secs,
            ) as response:
                return 200 <= response.status < 300
        except (OSError, urllib.error.URLError):
            return False

    def _wait_for_port_release(self, port: int) -> None:
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                try:
                    probe.bind(("127.0.0.1", port))
                except OSError:
                    time.sleep(0.2)
                    continue
                return
        raise TimeoutError(f"replica port {port} did not become available")

    def _set_router_disabled(self, spec: ReplicaSpec, disabled: bool) -> None:
        worker_url = f"http://127.0.0.1:{spec.port}"
        worker_id = urllib.parse.quote(worker_url, safe="")
        payload: dict[str, bool] = {"disabled": disabled}
        if disabled:
            payload["is_dead"] = True
        else:
            payload["is_dead"] = False
        request = urllib.request.Request(
            f"{self.router_url}/workers/{worker_id}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="PUT",
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.health_timeout_secs,
            ) as response:
                if response.status != 200:
                    raise RuntimeError(
                        f"router worker update returned HTTP {response.status}"
                    )
        except (OSError, urllib.error.URLError) as exc:
            action = "disable" if disabled else "enable"
            raise RuntimeError(
                f"could not {action} replica {spec.index} in router"
            ) from exc

    def _terminate_replica(self, record: ReplicaRecord) -> None:
        _terminate_record(record)

    def _launch_replica(
        self,
        spec: ReplicaSpec,
        pending: PendingRestart,
    ) -> ReplicaRecord:
        environment = os.environ.copy()
        environment.update(spec.env)
        log_path = Path(spec.log)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        pending_path = _pending_restart_path(self.state, spec.index)
        wrapper_command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "exec-replica",
            "--pending-record",
            str(pending_path),
            "--token",
            pending.token,
            "--",
            *spec.command,
        ]
        with log_path.open("ab", buffering=0) as log_file:
            process = subprocess.Popen(
                wrapper_command,
                cwd=spec.cwd,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        start_time = _wait_for_start_time(process.pid)
        replacement = ReplicaRecord(
            index=spec.index,
            pid=process.pid,
            pgid=process.pid,
            port=spec.port,
            log=spec.log,
            leader_start=start_time,
        )
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            recorded = PendingRestart.from_path(pending_path)
            if recorded.token != pending.token:
                raise RuntimeError(
                    f"pending restart token changed for replica {spec.index}"
                )
            if recorded.as_replica_record() == replacement:
                return replacement
            if not _pid_is_live(process.pid):
                break
            time.sleep(0.02)
        raise RuntimeError(
            f"replacement replica {spec.index} did not persist its pending identity"
        )

    def _wait_for_health(
        self,
        spec: ReplicaSpec,
        record: ReplicaRecord,
    ) -> None:
        for _ in range(self.health_tries):
            if not self._identity_matches(record):
                raise RuntimeError(
                    f"replacement replica {spec.index} exited before health"
                )
            if self._health_ok(spec.port):
                return
            time.sleep(self.health_interval_secs)
        raise TimeoutError(
            f"replacement replica {spec.index} did not become healthy "
            f"after {self.health_tries} checks"
        )

    def _validate_kv_capacity(
        self,
        spec: ReplicaSpec,
        *,
        log_offset: int = 0,
    ) -> None:
        if spec.expected_tokens is None:
            return
        resolved: int | None = None
        with Path(spec.log).open("rb") as stream:
            stream.seek(log_offset)
            current_generation_log = stream.read().decode(errors="replace")
        for line in current_generation_log.splitlines():
            match = re.search(r"#tokens:\s*([0-9]+)", line)
            if match is not None:
                resolved = int(match.group(1))
        if resolved != spec.expected_tokens:
            raise RuntimeError(
                f"replica {spec.index} resolved {resolved!r} KV tokens; "
                f"expected {spec.expected_tokens}"
            )

    def _attach_scope(self, spec: ReplicaSpec) -> list[int]:
        # The restarting replica is always gated. Peers are gated only while
        # they are live and healthy: a peer that is dead or being replaced has
        # no process group left to match an MPS client, so including it would
        # make the gate structurally unpassable whenever more than one replica
        # is down. An unexpectedly detached HEALTHY peer is still in scope and
        # still fails the gate, so the single-fault property is unchanged.
        scope = {spec.index}
        for index, record in self._load_replica_records().items():
            if index == spec.index:
                continue
            if self._identity_matches(record) and self._health_ok(record.port):
                scope.add(index)
        return sorted(scope)

    def _verify_mps_attach(self, spec: ReplicaSpec) -> None:
        environment = os.environ.copy()
        environment["STATE_ROOT"] = str(self.state.parents[1])
        scope = ",".join(str(index) for index in self._attach_scope(spec))
        result = subprocess.run(
            ["bash", str(self.launch_script), "verify", self.state.name, scope],
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise RuntimeError(f"MPS attach verification failed: {detail}")

    def _record_event(self, event: str, **fields: Any) -> None:
        payload = {
            "time_unix_s": time.time(),
            "event": event,
            **fields,
        }
        with (self.state / "supervisor-events.jsonl").open(
            "a",
            encoding="utf-8",
        ) as stream:
            stream.write(json.dumps(payload, sort_keys=True) + "\n")


def _pid_is_live(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    try:
        status = Path(f"/proc/{pid}/stat").read_text().split()[2]
    except (FileNotFoundError, IndexError):
        return False
    return status != "Z"


def _pid_start_time(pid: int) -> str | None:
    result = subprocess.run(
        ["ps", "-o", "lstart=", "-p", str(pid)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    value = result.stdout.rstrip("\n")
    return value if value.strip() else None


def _wait_for_start_time(pid: int) -> str:
    for _ in range(50):
        start_time = _pid_start_time(pid)
        if start_time is not None:
            return start_time
        time.sleep(0.02)
    raise RuntimeError(f"replica PID {pid} exited before identity was recorded")


def _pending_restart_path(state: Path, index: int) -> Path:
    return state / "pending-restarts" / f"replica_{index}.json"


def _record_identity_matches(record: ReplicaRecord) -> bool:
    if not _pid_is_live(record.pid):
        return False
    try:
        pgid = os.getpgid(record.pid)
    except ProcessLookupError:
        return False
    return pgid == record.pgid and _pid_start_time(record.pid) == record.leader_start


def _terminate_record(record: ReplicaRecord) -> None:
    if not _record_identity_matches(record):
        return
    os.killpg(record.pgid, signal.SIGTERM)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if not _record_identity_matches(record):
            return
        time.sleep(0.1)
    if _record_identity_matches(record):
        os.killpg(record.pgid, signal.SIGKILL)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if not _record_identity_matches(record):
                return
            time.sleep(0.05)
        raise RuntimeError(
            f"pending replica {record.index} process group "
            f"{record.pgid} survived SIGKILL"
        )


def _find_pending_wrapper(token: str) -> ReplicaRecord | None:
    marker = token.encode()
    for proc_path in Path("/proc").iterdir():
        if not proc_path.name.isdigit():
            continue
        try:
            if proc_path.stat().st_uid != os.geteuid():
                continue
        except (FileNotFoundError, PermissionError):
            continue
        pid = int(proc_path.name)
        try:
            command = (proc_path / "cmdline").read_bytes().split(b"\0")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if marker not in command:
            continue
        try:
            pgid = os.getpgid(pid)
        except ProcessLookupError:
            continue
        start = _pid_start_time(pid)
        if start is None:
            continue
        return ReplicaRecord(
            index=-1,
            pid=pid,
            pgid=pgid,
            port=0,
            log="",
            leader_start=start,
        )
    return None


def _listening_socket_inodes(port: int) -> set[str]:
    inodes: set[str] = set()
    for table_path in (Path("/proc/net/tcp"), Path("/proc/net/tcp6")):
        try:
            lines = table_path.read_text().splitlines()[1:]
        except (FileNotFoundError, PermissionError):
            continue
        for line in lines:
            fields = line.split()
            if len(fields) <= 9 or fields[3] != "0A":
                continue
            try:
                local_port = int(fields[1].rsplit(":", 1)[1], 16)
            except (IndexError, ValueError):
                continue
            if local_port == port:
                inodes.add(fields[9])
    return inodes


def _find_port_listener(port: int) -> ReplicaRecord | None:
    socket_targets = {f"socket:[{inode}]" for inode in _listening_socket_inodes(port)}
    if not socket_targets:
        return None
    for proc_path in Path("/proc").iterdir():
        if not proc_path.name.isdigit():
            continue
        try:
            if proc_path.stat().st_uid != os.geteuid():
                continue
        except (FileNotFoundError, PermissionError):
            continue
        pid = int(proc_path.name)
        try:
            descriptors = (proc_path / "fd").iterdir()
            owns_listener = any(
                os.readlink(descriptor) in socket_targets for descriptor in descriptors
            )
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if not owns_listener:
            continue
        try:
            pgid = os.getpgid(pid)
        except ProcessLookupError:
            continue
        start = _pid_start_time(pid)
        if start is None:
            continue
        return ReplicaRecord(
            index=-1,
            pid=pid,
            pgid=pgid,
            port=port,
            log="",
            leader_start=start,
        )
    return None


def _process_command(pid: int) -> list[str] | None:
    try:
        fields = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None
    return [os.fsdecode(field) for field in fields if field]


def _expected_replica_command(state: Path, index: int) -> list[str] | None:
    path = state / "replica_specs.json"
    try:
        payload = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, PermissionError):
        return None
    if not isinstance(payload, list):
        return None
    for item in payload:
        if not isinstance(item, dict) or item.get("index") != index:
            continue
        command = item.get("command")
        if not isinstance(command, list) or not all(
            isinstance(part, str) for part in command
        ):
            return None
        return command
    return None


def _append_event(state: Path, event: str, **fields: Any) -> None:
    payload = {"time_unix_s": time.time(), "event": event, **fields}
    with (state / "supervisor-events.jsonl").open(
        "a",
        encoding="utf-8",
    ) as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")


def cleanup_pending_restarts(
    state: Path,
    *,
    indices: set[int] | None = None,
    identity_grace_secs: float = 2,
) -> list[int]:
    state = state.resolve()
    pending_dir = state / "pending-restarts"
    if not pending_dir.exists():
        return []
    cleaned: list[int] = []
    for path in sorted(pending_dir.glob("replica_*.json")):
        pending = PendingRestart.from_path(path)
        if indices is not None and pending.index not in indices:
            continue
        remaining_grace = max(
            0,
            identity_grace_secs - (time.time() - pending.created_unix_s),
        )
        deadline = time.monotonic() + remaining_grace
        target: ReplicaRecord | None = None
        while True:
            pending = PendingRestart.from_path(path)
            target = pending.as_replica_record()
            if target is not None and _record_identity_matches(target):
                break
            target = _find_pending_wrapper(pending.token)
            if target is None:
                target = _find_port_listener(pending.port)
                if target is not None:
                    expected_command = _expected_replica_command(
                        state,
                        pending.index,
                    )
                    actual_command = _process_command(target.pid)
                    if expected_command is None or actual_command != expected_command:
                        _append_event(
                            state,
                            "pending_restart_identity_refused",
                            replica=pending.index,
                            token=pending.token,
                            candidate_pid=target.pid,
                            reason="port listener command does not match replica spec",
                        )
                        print(
                            "warning: refusing to terminate PID "
                            f"{target.pid} on replica {pending.index} port "
                            f"{pending.port}: command does not match replica spec",
                            file=sys.stderr,
                        )
                        target = None
                        break
            if target is not None:
                target = dataclasses.replace(
                    target,
                    index=pending.index,
                    port=pending.port,
                    log=pending.log,
                )
                break
            if time.monotonic() >= deadline:
                target = None
                break
            time.sleep(0.05)
        if target is not None:
            _terminate_record(target)
        path.unlink(missing_ok=True)
        _append_event(
            state,
            "pending_restart_cleaned",
            replica=pending.index,
            token=pending.token,
            terminated_pid=target.pid if target is not None else None,
        )
        cleaned.append(pending.index)
    with contextlib.suppress(OSError):
        pending_dir.rmdir()
    return cleaned


def _atomic_write_text(path: Path, content: str) -> None:
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as stream:
        temporary_path = Path(stream.name)
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    temporary_path.replace(path)


def _exec_replica(args: argparse.Namespace) -> None:
    pending_path = args.pending_record.resolve()
    pending = PendingRestart.from_path(pending_path)
    if pending.token != args.token:
        raise RuntimeError(
            f"pending restart token mismatch for replica {pending.index}"
        )
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("replacement command must not be empty")
    pid = os.getpid()
    identified = dataclasses.replace(
        pending,
        pid=pid,
        pgid=os.getpgid(pid),
        leader_start=_wait_for_start_time(pid),
    )
    _atomic_write_text(pending_path, identified.to_json())
    _append_event(
        pending_path.parents[1],
        "pending_restart_identified",
        replica=pending.index,
        token=pending.token,
        pid=pid,
        pgid=identified.pgid,
    )
    os.execvpe(command[0], command, os.environ)


def _add_replica(args: argparse.Namespace) -> None:
    state = args.state.resolve()
    path = state / "replica_specs.json"
    payload: list[dict[str, Any]]
    if path.exists():
        loaded = json.loads(path.read_text())
        if not isinstance(loaded, list):
            raise ValueError("replica_specs.json must contain a list")
        payload = loaded
    else:
        payload = []
    if any(item.get("index") == args.index for item in payload):
        raise ValueError(f"replica spec {args.index} already exists")
    environment: dict[str, str] = {}
    for item in args.env:
        key, separator, value = item.partition("=")
        if not separator or not key:
            raise ValueError(f"--env requires KEY=VALUE, got {item!r}")
        environment[key] = value
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("replica command must not be empty")
    spec = ReplicaSpec(
        index=args.index,
        port=args.port,
        log=str(args.log.resolve()),
        expected_tokens=args.expected_tokens,
        command=command,
        env=environment,
        cwd=str(args.cwd.resolve()) if args.cwd else None,
    )
    payload.append(dataclasses.asdict(spec))
    payload.sort(key=lambda item: item["index"])
    _atomic_write_text(path, json.dumps(payload, indent=2) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command_name", required=True)

    add_parser = subparsers.add_parser("add-replica")
    add_parser.add_argument("--state", type=Path, required=True)
    add_parser.add_argument("--index", type=int, required=True)
    add_parser.add_argument("--port", type=int, required=True)
    add_parser.add_argument("--log", type=Path, required=True)
    add_parser.add_argument("--expected-tokens", type=int, default=None)
    add_parser.add_argument("--cwd", type=Path, default=None)
    add_parser.add_argument("--env", action="append", default=[])
    add_parser.add_argument("command", nargs=argparse.REMAINDER)

    exec_parser = subparsers.add_parser("exec-replica")
    exec_parser.add_argument("--pending-record", type=Path, required=True)
    exec_parser.add_argument("--token", required=True)
    exec_parser.add_argument("command", nargs=argparse.REMAINDER)

    cleanup_parser = subparsers.add_parser("cleanup-pending")
    cleanup_parser.add_argument("--state", type=Path, required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--state", type=Path, required=True)
    run_parser.add_argument("--launch-script", type=Path, required=True)
    run_parser.add_argument("--router-url", required=True)
    run_parser.add_argument("--interval-secs", type=float, default=5)
    run_parser.add_argument("--health-failure-threshold", type=int, default=3)
    run_parser.add_argument("--health-timeout-secs", type=float, default=3)
    run_parser.add_argument("--health-tries", type=int, default=50)
    run_parser.add_argument("--health-interval-secs", type=float, default=6)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command_name == "add-replica":
        _add_replica(args)
        return
    if args.command_name == "exec-replica":
        _exec_replica(args)
        return
    if args.command_name == "cleanup-pending":
        cleanup_pending_restarts(args.state)
        return
    supervisor = ReplicaSupervisor(
        state=args.state,
        launch_script=args.launch_script,
        router_url=args.router_url,
        interval_secs=args.interval_secs,
        health_failure_threshold=args.health_failure_threshold,
        health_timeout_secs=args.health_timeout_secs,
        health_tries=args.health_tries,
        health_interval_secs=args.health_interval_secs,
    )
    supervisor.run()


if __name__ == "__main__":
    main()
