# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import os
import signal
import sys
import threading
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
        lambda _spec: events.append("launch") or new,
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
        lambda _spec: events.append("kv"),
    )
    monkeypatch.setattr(
        supervisor,
        "_verify_mps_attach",
        lambda: events.append("mps_attach"),
    )

    supervisor.restart_replica(spec, old)

    assert events == [
        "router_disable",
        "terminate",
        "port_release",
        "launch",
        "record",
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
        "_launch_replica",
        lambda _spec: (_ for _ in ()).throw(RuntimeError("launch failed")),
    )

    with pytest.raises(RuntimeError, match="launch failed"):
        supervisor.restart_replica(spec, old)

    assert disabled == [True]


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
    monkeypatch.setattr(supervisor, "_launch_replica", lambda _spec: replacement)
    monkeypatch.setattr(supervisor, "_wait_for_port_release", lambda _port: None)
    monkeypatch.setattr(supervisor, "_replace_replica_record", lambda _record: None)
    monkeypatch.setattr(supervisor, "_wait_for_health", lambda _spec, _record: None)
    monkeypatch.setattr(
        supervisor,
        "_validate_kv_capacity",
        lambda _spec: (_ for _ in ()).throw(RuntimeError("KV mismatch")),
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
        assert router_updates == [
            {"disabled": True, "is_dead": True},
            {"disabled": False, "is_dead": False},
        ]
    finally:
        if replacement is not None and supervisor._identity_matches(replacement):
            os.killpg(replacement.pgid, signal.SIGTERM)
        router.shutdown()
        router.server_close()
