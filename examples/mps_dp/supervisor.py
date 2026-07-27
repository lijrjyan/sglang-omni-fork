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
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
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
        for spec in self._load_specs():
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
            self.restart_replica(spec, record)
            self._health_failures[spec.index] = 0
            records[spec.index] = self._load_replica_records()[spec.index]

    def restart_replica(self, spec: ReplicaSpec, old: ReplicaRecord) -> None:
        # This process holds the singleton lock for its full lifetime. Restarts
        # therefore cannot overlap CUDA-graph capture or memory profiling.
        self._record_event("restart_started", replica=spec.index, old_pid=old.pid)
        self._set_router_disabled(spec, True)
        replacement: ReplicaRecord | None = None
        try:
            self._terminate_replica(old)
            self._wait_for_port_release(spec.port)
            log_offset = Path(spec.log).stat().st_size if Path(spec.log).exists() else 0
            replacement = self._launch_replica(spec)
            self._replace_replica_record(replacement)
            self._wait_for_health(spec, replacement)
            self._validate_kv_capacity(spec, log_offset=log_offset)
            self._verify_mps_attach()
            self._set_router_disabled(spec, False)
        except Exception as exc:
            # A healthy HTTP endpoint is not sufficient to re-enter the pool.
            # If any post-launch gate (including router re-registration) fails,
            # stop the replacement so the next check retries the full chain.
            if replacement is not None:
                self._terminate_replica(replacement)
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

    def _identity_matches(self, record: ReplicaRecord) -> bool:
        if not _pid_is_live(record.pid):
            return False
        try:
            pgid = os.getpgid(record.pid)
        except ProcessLookupError:
            return False
        return (
            pgid == record.pgid and _pid_start_time(record.pid) == record.leader_start
        )

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
        if not self._identity_matches(record):
            return
        os.killpg(record.pgid, signal.SIGTERM)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if not self._identity_matches(record):
                return
            time.sleep(0.2)
        if self._identity_matches(record):
            os.killpg(record.pgid, signal.SIGKILL)
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                if not self._identity_matches(record):
                    return
                time.sleep(0.1)
            raise RuntimeError(
                f"replica {record.index} process group {record.pgid} survived SIGKILL"
            )

    def _launch_replica(self, spec: ReplicaSpec) -> ReplicaRecord:
        environment = os.environ.copy()
        environment.update(spec.env)
        log_path = Path(spec.log)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("ab", buffering=0) as log_file:
            process = subprocess.Popen(
                spec.command,
                cwd=spec.cwd,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        start_time = _wait_for_start_time(process.pid)
        return ReplicaRecord(
            index=spec.index,
            pid=process.pid,
            pgid=process.pid,
            port=spec.port,
            log=spec.log,
            leader_start=start_time,
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

    def _verify_mps_attach(self) -> None:
        environment = os.environ.copy()
        environment["STATE_ROOT"] = str(self.state.parents[1])
        result = subprocess.run(
            ["bash", str(self.launch_script), "verify", self.state.name],
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
