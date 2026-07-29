# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SUPERVISOR_PATH = REPO_ROOT / "examples" / "mps_dp" / "supervisor.py"

_spec = importlib.util.spec_from_file_location("mps_dp_supervisor", SUPERVISOR_PATH)
assert _spec is not None and _spec.loader is not None
mps_dp_supervisor = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = mps_dp_supervisor
_spec.loader.exec_module(mps_dp_supervisor)


def _supervisor(tmp_path: Path):
    state = tmp_path / "gpu-0" / "run-test"
    state.mkdir(parents=True, exist_ok=True)
    supervisor = mps_dp_supervisor.ReplicaSupervisor(
        state=state,
        launch_script=REPO_ROOT / "examples" / "mps_dp" / "launch.sh",
        router_url="http://127.0.0.1:8799",
        interval_secs=0.01,
        health_failure_threshold=2,
        health_timeout_secs=0.01,
        health_tries=2,
        health_interval_secs=0.01,
    )
    return supervisor


def test_launch_manifest_records_driver_and_host_timezone() -> None:
    launch_text = (REPO_ROOT / "examples" / "mps_dp" / "launch.sh").read_text(
        encoding="utf-8"
    )

    assert "--query-gpu=driver_version" in launch_text
    assert 'echo "driver_version=$driver_version"' in launch_text
    assert 'echo "host_timezone=$host_timezone"' in launch_text
    assert 'echo "started_at=$started_at"' in launch_text


def test_restart_gate_orders_router_kv_attach_and_registration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = _supervisor(tmp_path)
    spec = mps_dp_supervisor.ReplicaSpec(
        index=1,
        port=8802,
        log=str(tmp_path / "replica_1.log"),
        expected_tokens=4096,
        command=["python", "-m", "fake_server"],
        env={"CUDA_MPS_PIPE_DIRECTORY": "/tmp/private-mps"},
    )
    old = mps_dp_supervisor.ReplicaRecord(
        index=1,
        pid=101,
        pgid=101,
        port=8802,
        log=spec.log,
        leader_start="old-start",
    )
    new = mps_dp_supervisor.ReplicaRecord(
        index=1,
        pid=202,
        pgid=202,
        port=8802,
        log=spec.log,
        leader_start="new-start",
    )
    events: list[str] = []

    monkeypatch.setattr(
        supervisor,
        "_set_router_disabled",
        lambda _spec, disabled: events.append(
            "router_disable" if disabled else "router_enable"
        ),
    )
    monkeypatch.setattr(
        supervisor,
        "_terminate_replica",
        lambda _record: events.append("terminate"),
    )
    monkeypatch.setattr(
        supervisor,
        "_launch_replica",
        lambda _spec, _pending: events.append("launch") or new,
    )
    monkeypatch.setattr(
        supervisor,
        "_prepare_pending_restart",
        lambda _spec: events.append("pending_prepare") or object(),
        raising=False,
    )
    monkeypatch.setattr(
        supervisor,
        "_clear_pending_restart",
        lambda _pending: events.append("pending_clear"),
        raising=False,
    )
    monkeypatch.setattr(
        supervisor,
        "_wait_for_port_release",
        lambda _port: events.append("port_release"),
    )
    monkeypatch.setattr(
        supervisor,
        "_replace_replica_record",
        lambda _record: events.append("record"),
    )
    monkeypatch.setattr(
        supervisor,
        "_wait_for_health",
        lambda _spec, _record: events.append("health"),
    )
    monkeypatch.setattr(
        supervisor,
        "_validate_kv_capacity",
        lambda _spec, **_kwargs: events.append("kv"),
    )
    monkeypatch.setattr(
        supervisor,
        "_verify_mps_attach",
        lambda _spec: events.append("mps_attach"),
    )

    supervisor.restart_replica(spec, old)

    assert events == [
        "router_disable",
        "terminate",
        "port_release",
        "pending_prepare",
        "launch",
        "record",
        "pending_clear",
        "health",
        "kv",
        "mps_attach",
        "router_enable",
    ]


def test_failed_restart_keeps_worker_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = _supervisor(tmp_path)
    spec = mps_dp_supervisor.ReplicaSpec(
        index=0,
        port=8801,
        log=str(tmp_path / "replica_0.log"),
        expected_tokens=4096,
        command=["false"],
        env={},
    )
    old = mps_dp_supervisor.ReplicaRecord(
        index=0,
        pid=101,
        pgid=101,
        port=8801,
        log=spec.log,
        leader_start="old-start",
    )
    disabled: list[bool] = []

    monkeypatch.setattr(
        supervisor,
        "_set_router_disabled",
        lambda _spec, value: disabled.append(value),
    )
    monkeypatch.setattr(supervisor, "_terminate_replica", lambda _record: None)
    monkeypatch.setattr(supervisor, "_wait_for_port_release", lambda _port: None)
    monkeypatch.setattr(
        supervisor,
        "_prepare_pending_restart",
        lambda _spec: object(),
        raising=False,
    )
    monkeypatch.setattr(
        supervisor,
        "_launch_replica",
        lambda _spec, _pending: (_ for _ in ()).throw(RuntimeError("launch failed")),
    )

    with pytest.raises(RuntimeError, match="launch failed"):
        supervisor.restart_replica(spec, old)

    assert disabled == [True]


def test_router_disable_failure_is_classified_as_restart_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = _supervisor(tmp_path)
    spec = mps_dp_supervisor.ReplicaSpec(
        index=0,
        port=8801,
        log=str(tmp_path / "replica_0.log"),
        expected_tokens=None,
        command=["false"],
        env={},
    )
    old = mps_dp_supervisor.ReplicaRecord(0, 101, 101, 8801, spec.log, "old")
    recorded: list[tuple[str, dict[str, object]]] = []

    monkeypatch.setattr(
        supervisor,
        "_set_router_disabled",
        lambda _spec, _disabled: (_ for _ in ()).throw(
            RuntimeError("router unavailable")
        ),
    )
    monkeypatch.setattr(
        supervisor,
        "_record_event",
        lambda event, **fields: recorded.append((event, fields)),
    )

    with pytest.raises(RuntimeError, match="router unavailable"):
        supervisor.restart_replica(spec, old)

    assert [event for event, _fields in recorded] == [
        "restart_started",
        "restart_failed",
    ]


@pytest.mark.skipif(os.name != "posix", reason="process groups require POSIX")
def test_cleanup_pending_restart_terminates_uncommitted_replacement(
    tmp_path: Path,
) -> None:
    state = tmp_path / "gpu-0" / "run-test"
    pending_dir = state / "pending-restarts"
    pending_dir.mkdir(parents=True)
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(300)"],
        start_new_session=True,
    )
    start = mps_dp_supervisor._wait_for_start_time(process.pid)
    pending = {
        "schema_version": 1,
        "index": 0,
        "port": 8801,
        "log": str(state / "logs" / "replica_0.log"),
        "token": "test-pending-token",
        "created_unix_s": time.time(),
        "pid": process.pid,
        "pgid": process.pid,
        "leader_start": start,
    }
    pending_path = pending_dir / "replica_0.json"
    pending_path.write_text(json.dumps(pending))

    try:
        cleaned = mps_dp_supervisor.cleanup_pending_restarts(state)
        process.wait(timeout=5)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)

    assert cleaned == [0]
    assert not pending_path.exists()


def test_cleanup_pending_restart_removes_stale_placeholder(tmp_path: Path) -> None:
    state = tmp_path / "gpu-0" / "run-test"
    pending_dir = state / "pending-restarts"
    pending_dir.mkdir(parents=True)
    pending_path = pending_dir / "replica_1.json"
    pending_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "index": 1,
                "port": 8802,
                "log": str(state / "logs" / "replica_1.log"),
                "token": "stale-token",
                "created_unix_s": time.time() - 60,
                "pid": None,
                "pgid": None,
                "leader_start": None,
            }
        )
    )

    assert mps_dp_supervisor.cleanup_pending_restarts(state) == [1]
    assert not pending_path.exists()


@pytest.mark.skipif(os.name != "posix", reason="/proc port ownership requires Linux")
def test_cleanup_pending_restart_refuses_unrelated_port_owner(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    state = tmp_path / "gpu-0" / "run-test"
    pending_dir = state / "pending-restarts"
    pending_dir.mkdir(parents=True)
    process = subprocess.Popen(
        [
            sys.executable,
            "-u",
            "-c",
            (
                "import socket, time; "
                "s=socket.socket(); s.bind(('127.0.0.1', 0)); s.listen(); "
                "print(s.getsockname()[1], flush=True); time.sleep(300)"
            ),
        ],
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    assert process.stdout is not None
    port = int(process.stdout.readline())
    (state / "replica_specs.json").write_text(
        json.dumps(
            [
                {
                    "index": 0,
                    "port": port,
                    "log": str(state / "logs" / "replica_0.log"),
                    "expected_tokens": None,
                    "command": ["python", "-m", "expected_server"],
                    "env": {},
                    "cwd": None,
                }
            ]
        )
    )
    pending_path = pending_dir / "replica_0.json"
    pending_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "index": 0,
                "port": port,
                "log": str(state / "logs" / "replica_0.log"),
                "token": "token-not-present-in-process-command",
                "created_unix_s": time.time() - 60,
                "pid": None,
                "pgid": None,
                "leader_start": None,
            }
        )
    )

    try:
        cleaned = mps_dp_supervisor.cleanup_pending_restarts(state)
        survived_cleanup = process.poll() is None
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)

    assert cleaned == [0]
    assert survived_cleanup
    assert not pending_path.exists()
    events = [
        json.loads(line)
        for line in (state / "supervisor-events.jsonl").read_text().splitlines()
    ]
    refused = [
        event
        for event in events
        if event["event"] == "pending_restart_identity_refused"
    ]
    assert len(refused) == 1
    assert refused[0]["candidate_pid"] == process.pid
    assert refused[0]["replica"] == 0
    assert f"refusing to terminate PID {process.pid}" in capsys.readouterr().err


@pytest.mark.skipif(os.name != "posix", reason="/proc port ownership requires Linux")
def test_cleanup_pending_restart_kills_matching_port_owner(tmp_path: Path) -> None:
    state = tmp_path / "gpu-0" / "run-test"
    pending_dir = state / "pending-restarts"
    pending_dir.mkdir(parents=True)
    server_code = (
        "import socket, time; "
        "s=socket.socket(); s.bind(('127.0.0.1', 0)); s.listen(); "
        "print(s.getsockname()[1], flush=True); time.sleep(300)"
    )
    command = [sys.executable, "-u", "-c", server_code]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    assert process.stdout is not None
    port = int(process.stdout.readline())
    (state / "replica_specs.json").write_text(
        json.dumps(
            [
                {
                    "index": 0,
                    "port": port,
                    "log": str(state / "logs" / "replica_0.log"),
                    "expected_tokens": None,
                    "command": command,
                    "env": {},
                    "cwd": None,
                }
            ]
        )
    )
    pending_path = pending_dir / "replica_0.json"
    pending_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "index": 0,
                "port": port,
                "log": str(state / "logs" / "replica_0.log"),
                "token": "token-not-present-in-process-command",
                "created_unix_s": time.time() - 60,
                "pid": None,
                "pgid": None,
                "leader_start": None,
            }
        )
    )

    try:
        cleaned = mps_dp_supervisor.cleanup_pending_restarts(state)
        process.wait(timeout=5)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)

    assert cleaned == [0]
    assert not pending_path.exists()


@pytest.mark.skipif(os.name != "posix", reason="fault injection requires fork")
def test_crash_mid_restart_leaves_teardown_owned_replacement(tmp_path: Path) -> None:
    supervisor = _supervisor(tmp_path)
    state = supervisor.state
    log = state / "logs" / "replica_0.log"
    old = mps_dp_supervisor.ReplicaRecord(
        0,
        999999,
        999999,
        8801,
        str(log),
        "old",
    )
    (state / "replicas.tsv").write_text(old.to_tsv())
    spec = mps_dp_supervisor.ReplicaSpec(
        index=0,
        port=8801,
        log=str(log),
        expected_tokens=None,
        command=[sys.executable, "-c", "import time; time.sleep(300)"],
        env={},
    )

    supervisor_pid = os.fork()
    if supervisor_pid == 0:
        try:
            child_supervisor = _supervisor(tmp_path)
            child_supervisor._set_router_disabled = lambda _spec, _disabled: None
            child_supervisor._terminate_replica = lambda _record: None
            child_supervisor._wait_for_port_release = lambda _port: None
            child_supervisor._replace_replica_record = lambda _record: time.sleep(300)
            child_supervisor.restart_replica(spec, old)
        finally:
            os._exit(1)

    pending_path = state / "pending-restarts" / "replica_0.json"
    replacement_pid: int | None = None
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if pending_path.exists():
                pending = json.loads(pending_path.read_text())
                if pending.get("pid") is not None:
                    replacement_pid = int(pending["pid"])
                    break
            time.sleep(0.02)
        assert replacement_pid is not None
        assert mps_dp_supervisor._pid_is_live(replacement_pid)
        assert (state / "replicas.tsv").read_text() == old.to_tsv()

        os.kill(supervisor_pid, signal.SIGKILL)
        os.waitpid(supervisor_pid, 0)
        supervisor_pid = -1

        assert mps_dp_supervisor._pid_is_live(replacement_pid)
        assert mps_dp_supervisor.cleanup_pending_restarts(state) == [0]
        assert not mps_dp_supervisor._pid_is_live(replacement_pid)
        assert not pending_path.exists()
    finally:
        if supervisor_pid > 0:
            with contextlib.suppress(ProcessLookupError):
                os.kill(supervisor_pid, signal.SIGKILL)
            os.waitpid(supervisor_pid, 0)
        if replacement_pid is not None and mps_dp_supervisor._pid_is_live(
            replacement_pid
        ):
            with contextlib.suppress(ProcessLookupError):
                os.killpg(replacement_pid, signal.SIGKILL)


def test_prepare_pending_restart_refuses_unreconciled_record(tmp_path: Path) -> None:
    supervisor = _supervisor(tmp_path)
    pending_path = supervisor.state / "pending-restarts" / "replica_0.json"
    pending_path.parent.mkdir(parents=True)
    pending_path.write_text("{}")
    spec = mps_dp_supervisor.ReplicaSpec(
        index=0,
        port=8801,
        log=str(supervisor.state / "logs" / "replica_0.log"),
        expected_tokens=None,
        command=["false"],
        env={},
    )

    with pytest.raises(RuntimeError, match="unreconciled pending restart"):
        supervisor._prepare_pending_restart(spec)


def test_supervisor_reconciles_pending_before_first_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = _supervisor(tmp_path)
    events: list[str] = []
    monkeypatch.setattr(
        mps_dp_supervisor,
        "cleanup_pending_restarts",
        lambda state: events.append(f"cleanup:{state.name}") or [],
    )
    monkeypatch.setattr(
        supervisor,
        "_record_event",
        lambda event, **_fields: events.append(event),
    )
    monkeypatch.setattr(
        supervisor,
        "check_once",
        lambda: (_ for _ in ()).throw(KeyboardInterrupt),
    )

    with pytest.raises(KeyboardInterrupt):
        supervisor.run()

    assert events == ["cleanup:run-test", "supervisor_started"]


@pytest.mark.skipif(os.name != "posix", reason="launch.sh needs a POSIX shell")
def test_down_consumes_pending_restart_ownership(tmp_path: Path) -> None:
    state_root = tmp_path / "state"
    state = state_root / "gpu-0" / "run-test"
    pending_dir = state / "pending-restarts"
    pending_dir.mkdir(parents=True)
    (state / "replicas.tsv").write_text("")
    (state / "services.tsv").write_text("")
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(300)"],
        start_new_session=True,
    )
    start = mps_dp_supervisor._wait_for_start_time(process.pid)
    (pending_dir / "replica_0.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "index": 0,
                "port": 8801,
                "log": str(state / "logs" / "replica_0.log"),
                "token": "down-pending-token",
                "created_unix_s": time.time(),
                "pid": process.pid,
                "pgid": process.pid,
                "leader_start": start,
            }
        )
    )
    environment = os.environ.copy()
    environment.update(
        {
            "STATE_ROOT": str(state_root),
            "PYTHON_BIN": sys.executable,
            "DRAIN_TRIES": "1",
            "DRAIN_INTERVAL": "0.01",
        }
    )

    try:
        result = subprocess.run(
            [
                "bash",
                str(REPO_ROOT / "examples" / "mps_dp" / "launch.sh"),
                "down",
                "run-test",
            ],
            env=environment,
            capture_output=True,
            text=True,
            timeout=15,
        )
        survived_down = process.poll() is None
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)

    assert result.returncode == 0, result.stdout + result.stderr
    assert not survived_down
    assert not state.exists()


def test_failed_post_launch_gate_stops_replacement_and_keeps_worker_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = _supervisor(tmp_path)
    spec = mps_dp_supervisor.ReplicaSpec(
        index=0,
        port=8801,
        log=str(tmp_path / "replica_0.log"),
        expected_tokens=4096,
        command=["python", "-m", "fake_server"],
        env={},
    )
    old = mps_dp_supervisor.ReplicaRecord(0, 101, 101, 8801, spec.log, "old")
    replacement = mps_dp_supervisor.ReplicaRecord(0, 202, 202, 8801, spec.log, "new")
    disabled: list[bool] = []
    terminated: list[int] = []

    monkeypatch.setattr(
        supervisor,
        "_set_router_disabled",
        lambda _spec, value: disabled.append(value),
    )
    monkeypatch.setattr(
        supervisor,
        "_terminate_replica",
        lambda record: terminated.append(record.pid),
    )
    monkeypatch.setattr(
        supervisor,
        "_prepare_pending_restart",
        lambda _spec: object(),
        raising=False,
    )
    monkeypatch.setattr(
        supervisor,
        "_clear_pending_restart",
        lambda _pending: None,
        raising=False,
    )
    monkeypatch.setattr(
        supervisor,
        "_launch_replica",
        lambda _spec, _pending: replacement,
    )
    monkeypatch.setattr(supervisor, "_wait_for_port_release", lambda _port: None)
    monkeypatch.setattr(supervisor, "_replace_replica_record", lambda _record: None)
    monkeypatch.setattr(supervisor, "_wait_for_health", lambda _spec, _record: None)
    monkeypatch.setattr(
        supervisor,
        "_validate_kv_capacity",
        lambda _spec, **_kwargs: (_ for _ in ()).throw(RuntimeError("KV mismatch")),
    )

    with pytest.raises(RuntimeError, match="KV mismatch"):
        supervisor.restart_replica(spec, old)

    assert disabled == [True]
    assert terminated == [101, 202]


def test_kv_validation_uses_replacement_generation_latest_log_entry(
    tmp_path: Path,
) -> None:
    supervisor = _supervisor(tmp_path)
    log = tmp_path / "replica_0.log"
    log.write_text("old boot #tokens: 4096,\nnew boot #tokens: 2048,\n")
    spec = mps_dp_supervisor.ReplicaSpec(
        index=0,
        port=8801,
        log=str(log),
        expected_tokens=4096,
        command=["true"],
        env={},
    )

    with pytest.raises(RuntimeError, match="resolved 2048 KV tokens"):
        supervisor._validate_kv_capacity(spec)


def test_kv_validation_does_not_reuse_previous_generation(
    tmp_path: Path,
) -> None:
    supervisor = _supervisor(tmp_path)
    log = tmp_path / "replica_0.log"
    old_log = "old boot #tokens: 4096,\n"
    log.write_text(old_log + "replacement reached health without KV marker\n")
    spec = mps_dp_supervisor.ReplicaSpec(
        index=0,
        port=8801,
        log=str(log),
        expected_tokens=4096,
        command=["true"],
        env={},
    )

    with pytest.raises(RuntimeError, match="resolved None KV tokens"):
        supervisor._validate_kv_capacity(
            spec,
            log_offset=len(old_log.encode()),
        )


def test_health_failures_restart_only_after_threshold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = _supervisor(tmp_path)
    spec = mps_dp_supervisor.ReplicaSpec(
        index=0,
        port=8801,
        log=str(tmp_path / "replica_0.log"),
        expected_tokens=None,
        command=["python", "-m", "fake_server"],
        env={},
    )
    record = mps_dp_supervisor.ReplicaRecord(
        index=0,
        pid=101,
        pgid=101,
        port=8801,
        log=spec.log,
        leader_start="old-start",
    )
    restarts: list[int] = []
    monkeypatch.setattr(supervisor, "_load_specs", lambda: [spec])
    monkeypatch.setattr(supervisor, "_load_replica_records", lambda: {0: record})
    monkeypatch.setattr(supervisor, "_identity_matches", lambda _record: True)
    monkeypatch.setattr(supervisor, "_health_ok", lambda _port: False)
    monkeypatch.setattr(
        supervisor,
        "restart_replica",
        lambda target, _record: restarts.append(target.index),
    )

    supervisor.check_once()
    assert restarts == []

    supervisor.check_once()
    assert restarts == [0]


def _replica_pair(tmp_path: Path):
    specs = [
        mps_dp_supervisor.ReplicaSpec(
            index=index,
            port=8801 + index,
            log=str(tmp_path / f"replica_{index}.log"),
            expected_tokens=None,
            command=["python", "-m", "fake_server"],
            env={},
        )
        for index in (0, 1)
    ]
    records = {
        spec.index: mps_dp_supervisor.ReplicaRecord(
            index=spec.index,
            pid=101 + spec.index,
            pgid=101 + spec.index,
            port=spec.port,
            log=spec.log,
            leader_start="old-start",
        )
        for spec in specs
    }
    return specs, records


def test_double_fault_restarts_both_replicas(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Round 4 livelock scenario: both replicas die at once, so every attach
    # verification runs while a peer is dead. Both replicas must recover.
    supervisor = _supervisor(tmp_path)
    attached = tmp_path / "attached.txt"
    attached.write_text("")
    # Emulates launch.sh verify: a scoped replica with no attached MPS client
    # fails the gate. Replicas outside the scope cannot fail it.
    script = tmp_path / "verify.sh"
    script.write_text(
        "#!/bin/bash\n"
        f'attached=" $(cat {attached}) "\n'
        'for idx in $(echo "$3" | tr "," " "); do\n'
        '  case "$attached" in\n'
        '    *" $idx "*) ;;\n'
        '    *) echo "replica $idx has no attached MPS client" >&2; exit 1;;\n'
        "  esac\ndone\nexit 0\n"
    )
    script.chmod(0o755)
    supervisor.launch_script = script
    specs, records = _replica_pair(tmp_path)
    healthy: set[int] = set()
    attempts: list[int] = []

    def fake_restart(spec, _record) -> None:
        # The replacement process is up and attached; the peer may still be dead.
        attempts.append(spec.index)
        live = sorted(healthy | {spec.index})
        attached.write_text(" ".join(str(index) for index in live))
        supervisor._verify_mps_attach(spec)
        healthy.add(spec.index)

    monkeypatch.setattr(supervisor, "_load_specs", lambda: specs)
    monkeypatch.setattr(supervisor, "_load_replica_records", lambda: dict(records))
    monkeypatch.setattr(supervisor, "_identity_matches", lambda _record: True)
    monkeypatch.setattr(
        supervisor,
        "_health_ok",
        lambda port: (port - 8801) in healthy,
    )
    monkeypatch.setattr(supervisor, "restart_replica", fake_restart)

    for _ in range(12):
        with contextlib.suppress(RuntimeError):
            supervisor.check_once()
        if healthy == {0, 1}:
            break

    assert healthy == {0, 1}
    assert set(attempts) == {0, 1}


def test_sweep_rotates_after_a_failed_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = _supervisor(tmp_path)
    specs, records = _replica_pair(tmp_path)
    attempts: list[int] = []

    def always_fails(spec, _record) -> None:
        attempts.append(spec.index)
        raise RuntimeError("MPS attach verification failed")

    monkeypatch.setattr(supervisor, "_load_specs", lambda: specs)
    monkeypatch.setattr(supervisor, "_load_replica_records", lambda: dict(records))
    monkeypatch.setattr(supervisor, "_identity_matches", lambda _record: True)
    monkeypatch.setattr(supervisor, "_health_ok", lambda _port: False)
    monkeypatch.setattr(supervisor, "restart_replica", always_fails)

    # First sweep only accrues a health failure; each later sweep attempts one
    # replica and aborts, so the attempts must alternate rather than pin to 0.
    for _ in range(5):
        with contextlib.suppress(RuntimeError):
            supervisor.check_once()

    assert attempts == [0, 1, 0, 1]


def _scope_probe_script(tmp_path: Path, detached_index: int) -> Path:
    script = tmp_path / "verify.sh"
    script.write_text(
        "#!/bin/bash\n"
        'case " $(echo "$3" | tr "," " ") " in\n'
        f'  *" {detached_index} "*)\n'
        f'    echo "replica {detached_index} has no process in the MPS client list"'
        " >&2; exit 1;;\nesac\nexit 0\n"
    )
    script.chmod(0o755)
    return script


def test_attach_gate_still_detects_a_detached_healthy_peer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = _supervisor(tmp_path)
    supervisor.launch_script = _scope_probe_script(tmp_path, 1)
    specs, records = _replica_pair(tmp_path)
    monkeypatch.setattr(supervisor, "_load_specs", lambda: specs)
    monkeypatch.setattr(supervisor, "_load_replica_records", lambda: dict(records))
    monkeypatch.setattr(supervisor, "_identity_matches", lambda _record: True)
    monkeypatch.setattr(supervisor, "_health_ok", lambda _port: True)

    assert supervisor._attach_scope(specs[0]) == [0, 1]
    with pytest.raises(RuntimeError, match="MPS attach verification failed"):
        supervisor._verify_mps_attach(specs[0])


def test_attach_gate_scope_excludes_a_dead_peer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = _supervisor(tmp_path)
    supervisor.launch_script = _scope_probe_script(tmp_path, 1)
    specs, records = _replica_pair(tmp_path)
    monkeypatch.setattr(supervisor, "_load_specs", lambda: specs)
    monkeypatch.setattr(supervisor, "_load_replica_records", lambda: dict(records))
    monkeypatch.setattr(
        supervisor,
        "_identity_matches",
        lambda record: record.index == 0,
    )
    monkeypatch.setattr(supervisor, "_health_ok", lambda port: port == 8801)

    assert supervisor._attach_scope(specs[0]) == [0]
    supervisor._verify_mps_attach(specs[0])


def test_singleton_lock_rejects_second_supervisor(tmp_path: Path) -> None:
    first = _supervisor(tmp_path)
    second = _supervisor(tmp_path)

    with first.singleton_lock():
        with pytest.raises(RuntimeError, match="already active"):
            with second.singleton_lock():
                pass


@pytest.mark.skipif(os.name != "posix", reason="process groups require POSIX")
def test_restart_fault_injection_runs_full_gate_chain(tmp_path: Path) -> None:
    router_updates: list[dict[str, bool]] = []

    class RouterHandler(BaseHTTPRequestHandler):
        def do_PUT(self) -> None:
            length = int(self.headers["Content-Length"])
            router_updates.append(json.loads(self.rfile.read(length)))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')

        def log_message(self, _format: str, *args: object) -> None:
            pass

    router = ThreadingHTTPServer(("127.0.0.1", 0), RouterHandler)
    router_thread = threading.Thread(target=router.serve_forever, daemon=True)
    router_thread.start()

    state = tmp_path / "gpu-0" / "run-test"
    state.mkdir(parents=True)
    log = state / "logs" / "replica_0.log"
    old = mps_dp_supervisor.ReplicaRecord(0, 999999, 999999, 0, str(log), "old")
    (state / "replicas.tsv").write_text(old.to_tsv())
    verify_script = tmp_path / "verify.sh"
    verify_script.write_text("#!/bin/bash\nexit 0\n")

    worker_code = """
from http.server import BaseHTTPRequestHandler, HTTPServer
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200 if self.path == "/health" else 404)
        self.end_headers()
    def log_message(self, format, *args):
        pass
print("#tokens: 4096", flush=True)
HTTPServer(("127.0.0.1", 0), Handler).serve_forever()
"""
    # Replace the ephemeral port in the command after reserving it.
    probe = ThreadingHTTPServer(("127.0.0.1", 0), RouterHandler)
    worker_port = probe.server_port
    probe.server_close()
    worker_code = worker_code.replace(
        'HTTPServer(("127.0.0.1", 0)',
        f'HTTPServer(("127.0.0.1", {worker_port})',
    )
    spec = mps_dp_supervisor.ReplicaSpec(
        index=0,
        port=worker_port,
        log=str(log),
        expected_tokens=4096,
        command=[sys.executable, "-u", "-c", worker_code],
        env={},
    )
    supervisor = mps_dp_supervisor.ReplicaSupervisor(
        state=state,
        launch_script=verify_script,
        router_url=f"http://127.0.0.1:{router.server_port}",
        interval_secs=0.01,
        health_failure_threshold=1,
        health_timeout_secs=1,
        health_tries=50,
        health_interval_secs=0.02,
    )

    replacement = None
    try:
        supervisor.restart_replica(spec, old)
        replacement = supervisor._load_replica_records()[0]
        assert supervisor._identity_matches(replacement)
        assert supervisor._health_ok(worker_port)
        assert not (state / "pending-restarts" / "replica_0.json").exists()
        lifecycle_events = [
            json.loads(line)["event"]
            for line in (state / "supervisor-events.jsonl").read_text().splitlines()
        ]
        assert lifecycle_events == [
            "restart_started",
            "pending_restart_prepared",
            "pending_restart_identified",
            "pending_restart_committed",
            "restart_completed",
        ]
        assert router_updates == [
            {"disabled": True, "is_dead": True},
            {"disabled": False, "is_dead": False},
        ]
    finally:
        if replacement is not None and supervisor._identity_matches(replacement):
            os.killpg(replacement.pgid, signal.SIGTERM)
        router.shutdown()
        router.server_close()


def _spawn_group_with_stubborn_child(
    ready_marker: Path,
    exit_after_secs: float,
) -> tuple[subprocess.Popen[str], int]:
    # Leader exits on SIGTERM; its child in the same process group ignores
    # SIGTERM and only leaves on its own schedule (or on SIGKILL). This is the
    # topology that a leader-only termination wait cannot clean up. The leader
    # publishes the child PID only after the child's handler is installed, so a
    # signal can never win a race against an interpreter that is still starting.
    child_code = (
        "import signal, sys, time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "open(sys.argv[1], 'w').write('ready'); "
        "time.sleep(float(sys.argv[2]))"
    )
    leader_code = f"""
import os, subprocess, sys, time
child = subprocess.Popen([
    sys.executable, "-u", "-c", {child_code!r},
    {str(ready_marker)!r}, {str(exit_after_secs)!r},
])
while not os.path.exists({str(ready_marker)!r}):
    time.sleep(0.01)
print(child.pid, flush=True)
time.sleep(300)
"""
    leader = subprocess.Popen(
        [sys.executable, "-u", "-c", leader_code],
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    assert leader.stdout is not None
    child_pid = int(leader.stdout.readline())
    return leader, child_pid


@pytest.mark.skipif(os.name != "posix", reason="process groups require POSIX")
def test_terminate_waits_for_group_member_after_leader_exits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mps_dp_supervisor, "TERMINATE_TERM_GRACE_SECS", 10)
    supervisor = _supervisor(tmp_path)
    leader, child_pid = _spawn_group_with_stubborn_child(tmp_path / "ready", 1.5)
    leader_start = mps_dp_supervisor._wait_for_start_time(leader.pid)
    old = mps_dp_supervisor.ReplicaRecord(
        index=0,
        pid=leader.pid,
        pgid=leader.pid,
        port=8801,
        log=str(tmp_path / "replica_0.log"),
        leader_start=leader_start,
    )
    spec = mps_dp_supervisor.ReplicaSpec(
        index=0,
        port=8801,
        log=old.log,
        expected_tokens=None,
        command=["python", "-m", "fake_server"],
        env={},
    )
    new = mps_dp_supervisor.ReplicaRecord(0, 202, 202, 8801, old.log, "new-start")
    child_live_at_gate: list[bool] = []

    def _launch(_spec: object, _pending: object) -> object:
        child_live_at_gate.append(mps_dp_supervisor._pid_is_live(child_pid))
        return new

    for name, replacement in (
        ("_set_router_disabled", lambda _spec, _disabled: None),
        ("_launch_replica", _launch),
        ("_prepare_pending_restart", lambda _spec: object()),
        ("_clear_pending_restart", lambda _pending: None),
        ("_wait_for_port_release", lambda _port: None),
        ("_replace_replica_record", lambda _record: None),
        ("_wait_for_health", lambda _spec, _record: None),
        ("_validate_kv_capacity", lambda _spec, **_kwargs: None),
        ("_verify_mps_attach", lambda _spec: None),
    ):
        monkeypatch.setattr(supervisor, name, replacement, raising=False)

    try:
        supervisor.restart_replica(spec, old)
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.kill(child_pid, signal.SIGKILL)
        with contextlib.suppress(ProcessLookupError):
            os.killpg(leader.pid, signal.SIGKILL)
        leader.wait(timeout=5)

    assert not mps_dp_supervisor._pid_is_live(leader.pid)
    assert not mps_dp_supervisor._pid_is_live(child_pid)
    # The replacement must not reach any gate while a member of the group it
    # replaces is still alive holding that replica's MPS client and memory.
    assert child_live_at_gate == [False]


@pytest.mark.skipif(os.name != "posix", reason="process groups require POSIX")
def test_terminate_kills_surviving_member_but_spares_a_reused_pid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mps_dp_supervisor, "TERMINATE_TERM_GRACE_SECS", 1)
    monkeypatch.setattr(mps_dp_supervisor, "TERMINATE_KILL_GRACE_SECS", 5)
    leader, child_pid = _spawn_group_with_stubborn_child(tmp_path / "ready", 300)
    leader_start = mps_dp_supervisor._wait_for_start_time(leader.pid)
    # Stands in for a PID recycled after the freeze: the number is in the frozen
    # member list but the process behind it is no longer the one that was frozen.
    innocent = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            "time.sleep(300)",
        ],
        start_new_session=True,
    )
    innocent_start = mps_dp_supervisor._wait_for_start_time(innocent.pid)
    real_freeze = mps_dp_supervisor._freeze_group_members
    monkeypatch.setattr(
        mps_dp_supervisor,
        "_freeze_group_members",
        lambda record: [
            *real_freeze(record),
            (innocent.pid, "Thu Jan  1 00:00:00 1970"),
        ],
    )
    record = mps_dp_supervisor.ReplicaRecord(
        index=0,
        pid=leader.pid,
        pgid=leader.pid,
        port=8801,
        log=str(tmp_path / "replica_0.log"),
        leader_start=leader_start,
    )

    try:
        mps_dp_supervisor._terminate_record(record)
        child_survived = mps_dp_supervisor._pid_is_live(child_pid)
        innocent_survived = mps_dp_supervisor._pid_start_time(innocent.pid) == (
            innocent_start
        )
    finally:
        for pid in (child_pid, innocent.pid, leader.pid):
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)
        innocent.wait(timeout=5)
        leader.wait(timeout=5)

    assert not child_survived
    assert innocent_survived
