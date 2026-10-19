#!/usr/bin/env python
"""
Live-trial harness: host the RL API and run a simulated trial against it
(Monitoring-Spec §11.3 "Simulated trial with injected faults"; implementation
contract §4 ``SimulationConfig``, D4, D20, D21).

What one run does:

1. Starts the API as a subprocess (werkzeug, threaded) with
   ``config.SimulationConfig`` and ``SQLALCHEMY_DATABASE_URI =
   sqlite:///<run_dir>/api.db``, ``ADAPTS_RUN_DIR``, ``ADAPTS_HYPERPARAMS``
   and the scenario's ``FAULT_*`` variables; waits until it listens.
2. Runs the monitoring program in this process: ``create_app(
   "config.SimulationConfig")`` on the same database file, then
   ``app.monitoring.run_monitoring(now=<logical time>)`` on a fixed logical
   grid (default every 15 logical minutes), skipping ticks only during an
   injected "monitoring stopped" fault.
3. Drives ``tests.simulate_adapts_hct.HostSimulator`` over HTTP for N dyads
   over the calendar weeks, with staggered any-weekday enrollment, waiting
   (wall time) for each accepted ``/update`` to finish before the logical
   clock moves on, and applying the scenario's server actions (restarts,
   kills, database tampering).
4. Writes ``<run_dir>/report.json`` and ``<run_dir>/report.md``: requests by
   endpoint/status, warm-up vs learned decisions per agent per week, the
   learned-pi distribution per agent per week, policy versions per week,
   alarms by item/severity, unexpected statuses, and the scenario verdict.

Usage::

    python tools/live_trial.py --scenario baseline --dyads 10 --weeks 16 --seed 42
    python tools/live_trial.py --scenario all --run-dir /tmp/adapts_trials
    python tools/live_trial.py --list

The ``serve`` sub-command is internal (the server subprocess).
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import os
import shutil
import signal
import socket
import sqlite3
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FAULT_ENV_KEYS = ("FAULT_ACTION_CRASH", "FAULT_FORCE_PI", "FAULT_SAMPLER_SHIFT", "FAULT_UPDATE_HANG_SECONDS")
DEFAULT_START = dt.date(2026, 11, 2)


# ------------------------------------------------------------------ server


def serve(port: int) -> None:
    """Server subprocess: the API under ``config.SimulationConfig`` (environment set by the harness)."""
    from werkzeug.serving import run_simple

    from app import create_app

    app = create_app("config.SimulationConfig")
    run_simple("127.0.0.1", port, app, threaded=True, use_reloader=False, use_debugger=False)


def free_port() -> int:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def port_open(port: int) -> bool:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1", port)) == 0


class ServerProcess:
    """The API subprocess; every start/stop is logged in ``events`` with the logical time it happened at."""

    def __init__(self, run_dir: Path, port: int, base_env: dict):
        self.run_dir = run_dir
        self.port = port
        self.base_env = dict(base_env)
        self.proc: subprocess.Popen | None = None
        self.events: list[dict] = []
        self._n_starts = 0

    def start(self, extra_env: dict | None = None, logical_time=None, timeout: float = 180.0) -> bool:
        """Start the server; ``True`` once it listens, ``False`` if it exited first (exit code logged)."""
        if self.proc is not None and self.proc.poll() is None:
            raise RuntimeError("server already running")
        if port_open(self.port):
            raise RuntimeError(f"port {self.port} is already in use")
        env = dict(self.base_env)
        for key, value in (extra_env or {}).items():
            if value is None:
                env.pop(key, None)
            else:
                env[key] = str(value)
        self._n_starts += 1
        log_path = self.run_dir / f"server_{self._n_starts:02d}.log"
        log = open(log_path, "w")
        self.proc = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "serve", "--port", str(self.port)],
            cwd=str(self.run_dir),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        started = time.monotonic()
        while time.monotonic() - started < timeout:
            code = self.proc.poll()
            if code is not None:
                log.close()
                self.events.append(self._event("exited_at_start", logical_time, extra_env, code=code, log=log_path.name))
                self.proc = None
                return False
            if port_open(self.port):
                self.events.append(self._event("started", logical_time, extra_env, log=log_path.name))
                return True
            time.sleep(0.2)
        self.stop(kill=True, logical_time=logical_time)
        raise TimeoutError(f"server did not listen on port {self.port} within {timeout}s (see {log_path})")

    def stop(self, kill: bool = False, logical_time=None) -> None:
        if self.proc is None:
            return
        if self.proc.poll() is None:
            if kill:
                self.proc.send_signal(signal.SIGKILL)
            else:
                self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=30)
        self.events.append(self._event("killed" if kill else "stopped", logical_time, None, code=self.proc.returncode))
        self.proc = None
        deadline = time.monotonic() + 30
        while port_open(self.port) and time.monotonic() < deadline:
            time.sleep(0.2)

    def restart(self, extra_env: dict | None = None, logical_time=None, kill: bool = False) -> bool:
        self.stop(kill=kill, logical_time=logical_time)
        return self.start(extra_env, logical_time)

    @staticmethod
    def _event(kind: str, logical_time, extra_env, **extra) -> dict:
        entry = {
            "event": kind,
            "logical_time": logical_time.isoformat() if logical_time else None,
            "wall_time": dt.datetime.now().isoformat(timespec="seconds"),
        }
        if extra_env:
            entry["env"] = {k: v for k, v in extra_env.items()}
        entry.update(extra)
        return entry


# --------------------------------------------------------------- database


class TrialDb:
    """Read-mostly access to the run's SQLite file (stdlib ``sqlite3``; tolerant of missing tables)."""

    def __init__(self, path: Path):
        self.path = path

    @contextlib.contextmanager
    def connect(self):
        conn = sqlite3.connect(str(self.path), timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def rows(self, sql: str, params: tuple = ()) -> list[dict]:
        try:
            with self.connect() as conn:
                return [dict(r) for r in conn.execute(sql, params).fetchall()]
        except sqlite3.Error:
            return []

    def execute(self, sql: str, params: tuple = ()) -> int:
        with self.connect() as conn:
            cursor = conn.execute(sql, params)
            conn.commit()
            return cursor.rowcount

    def update_status(self, rid: str) -> str | None:
        rows = self.rows("SELECT status FROM model_update_requests WHERE rid = ?", (rid,))
        return rows[0]["status"] if rows else None


# ---------------------------------------------------------------- harness


class LiveTrial:
    """One scenario run: server, monitoring ticks, simulator, report."""

    def __init__(self, scenario, args, run_dir: Path):
        from tests.simulate_adapts_hct import TrialPlan
        from tools.fault_scenarios import plan_with_extra_dyads, resolve_host_faults

        self.scenario = scenario
        self.args = args
        self.run_dir = run_dir
        self.db = TrialDb(run_dir / "api.db")
        base_plan = TrialPlan(
            num_dyads=args.dyads,
            num_weeks=args.weeks,
            start_date=args.start_date,
            seed=args.seed,
            strict=False,
            analytics_gap_rate=args.analytics_gap_rate,
        )
        self.plan = plan_with_extra_dyads(base_plan, scenario)
        if self.plan.num_weeks < scenario.min_weeks:
            raise SystemExit(f"scenario {scenario.name} needs --weeks >= {scenario.min_weeks}")
        self.faults = resolve_host_faults(scenario, self.plan)
        self.yaml_path = run_dir / "hyperparameters.yaml"
        self.yaml_settings = scenario.hyperparams or {}
        self.port = args.port or free_port()
        self.server: ServerProcess | None = None
        self.monitor_app = None
        self.tick_stats = Counter()
        self.tick_errors: list[dict] = []
        self.tick_seconds: list[float] = []
        self.skip_wait: set[str] = set()
        self.update_waits: list[dict] = []
        self.notes: list[str] = []
        self.flags: dict = {}
        self.started_wall = time.monotonic()

    # ------------------------------------------------------------ env

    def _write_yaml(self, settings: dict) -> None:
        import yaml

        header = f"# live_trial scenario {self.scenario.name}\n"
        self.yaml_path.write_text(header + (yaml.safe_dump(settings, sort_keys=True) if settings else ""))

    def _env(self) -> dict:
        env = dict(os.environ)
        for key in FAULT_ENV_KEYS:
            env.pop(key, None)
        env.update(
            {
                "PYTHONPATH": str(REPO_ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""),
                "SQLALCHEMY_DATABASE_URI": f"sqlite:///{self.run_dir / 'api.db'}",
                "ADAPTS_RUN_DIR": str(self.run_dir),
                "ADAPTS_HYPERPARAMS": str(self.yaml_path),
                "RL_ALGORITHM": "inf_lsvi_eb",
                "CLOCK_MODE": "host",
                "PYTHONUNBUFFERED": "1",
            }
        )
        for key, value in self.scenario.server_env.items():
            env[key] = str(value)
        return env

    # ------------------------------------------------------- monitoring

    def _boot_monitor(self) -> None:
        """create_app on the same DB in this process (env must match the server's)."""
        env = self._env()
        for key in ("SQLALCHEMY_DATABASE_URI", "ADAPTS_RUN_DIR", "ADAPTS_HYPERPARAMS", "RL_ALGORITHM", "CLOCK_MODE"):
            os.environ[key] = env[key]
        for key in FAULT_ENV_KEYS:
            os.environ.pop(key, None)
        os.chdir(self.run_dir)  # app logging writes ./logs
        from app import create_app

        self.monitor_app = create_app("config.SimulationConfig")

    def on_tick(self, now: dt.datetime) -> None:
        pause = self.scenario.monitor_pause
        if pause is not None:
            start = pause.start.resolve(self.plan)
            if start <= now < start + pause.duration:
                self.tick_stats["skipped_paused"] += 1
                return
        from app import monitoring

        started = time.perf_counter()
        try:
            with self.monitor_app.app_context():
                monitoring.run_monitoring(now=now)
            self.tick_stats["ok"] += 1
        except Exception as exc:  # noqa: BLE001 - keep the trial going, report it
            self.tick_stats["error"] += 1
            if len(self.tick_errors) < 50:
                self.tick_errors.append(
                    {"logical_time": now.isoformat(), "error": repr(exc), "traceback": traceback.format_exc(limit=4)}
                )
        self.tick_seconds.append(time.perf_counter() - started)

    # ------------------------------------------------------------ hooks

    def wait_for_update(self, rid: str, when: dt.datetime) -> None:
        if rid in self.skip_wait:
            return
        started = time.monotonic()
        status = self.db.update_status(rid)
        while status == "processing" and time.monotonic() - started < self.args.update_timeout:
            time.sleep(0.2)
            status = self.db.update_status(rid)
        self.update_waits.append(
            {"rid": rid, "logical_time": when.isoformat(), "status": status, "wall_seconds": round(time.monotonic() - started, 2)}
        )

    def after_request(self, record, simulator) -> None:
        if record.endpoint == "update":
            print(
                f"[live_trial] {self.scenario.name}: /update {record.status} at logical {record.logical_time} "
                f"(wall {time.monotonic() - self.started_wall:.0f} s, ticks {dict(self.tick_stats)})",
                flush=True,
            )
        for action in self.scenario.server_actions:
            if action.kind == "kill_during_update" and record.endpoint == "update" and record.status == 202:
                monday = action.when.resolve(self.plan).date()
                if record.logical_time.date() == monday and not self.flags.get("killed"):
                    self.flags["killed"] = record.rid
                    self.skip_wait.add(record.rid)
                    self.server.stop(kill=True, logical_time=record.logical_time)
                    self.server.start({"FAULT_UPDATE_HANG_SECONDS": None}, record.logical_time)
                    self.notes.append(f"killed the server right after /update {record.rid} at {record.logical_time}")
        tamper = self.scenario.db_tamper
        if (
            tamper is not None
            and not self.flags.get("tampered")
            and record.endpoint == "action"
            and record.status in (200, 201)
            and record.logical_time >= tamper.after.resolve(self.plan)
            and (record.response or {}).get("warmup") is False
        ):
            self.flags["tampered"] = self._tamper(tamper, record)

    def _tamper(self, tamper, record) -> dict | None:
        if tamper.name != "shift_posterior":
            raise ValueError(f"unknown tamper {tamper.name!r}")
        rows = self.db.rows("SELECT id, model_parameters_id FROM actions WHERE rid = ?", (record.rid,))
        if not rows or rows[0]["model_parameters_id"] is None:
            self.notes.append(f"tamper skipped: action {record.rid} has no posterior row")
            return None
        mp_id = rows[0]["model_parameters_id"]
        theta_rows = self.db.rows("SELECT theta FROM model_parameters WHERE id = ?", (mp_id,))
        theta = json.loads(theta_rows[0]["theta"]) if theta_rows and theta_rows[0]["theta"] else None
        if not theta:
            self.notes.append(f"tamper skipped: posterior {mp_id} has no theta")
            return None
        theta[0] = float(theta[0]) + tamper.delta
        self.db.execute("UPDATE model_parameters SET theta = ? WHERE id = ?", (json.dumps(theta), mp_id))
        entry = {"action_rid": record.rid, "model_parameters_id": mp_id, "delta": tamper.delta, "logical_time": record.logical_time.isoformat()}
        self.notes.append(f"tampered posterior {mp_id} used by {record.rid}")
        return entry

    def _timed_hooks(self) -> list:
        hooks = []
        window = self.scenario.server_env_window
        if window is not None:
            env_on = dict(window.env)
            env_off = {k: None for k in window.env}
            hooks.append((window.start.resolve(self.plan), lambda now: self._restart(env_on, now)))
            hooks.append((window.end.resolve(self.plan), lambda now: self._restart(env_off, now)))
        for action in self.scenario.server_actions:
            when = action.when.resolve(self.plan)
            if action.kind == "kill_during_update":
                hooks.append((when, lambda now, a=action: self._restart({"FAULT_UPDATE_HANG_SECONDS": a.hang_seconds}, now)))
            elif action.kind == "restart_with_yaml":
                hooks.append((when, lambda now, a=action: self._restart_with_yaml(a, now)))
            else:
                raise ValueError(f"unknown server action {action.kind!r}")
        return hooks

    def _restart(self, env: dict, now: dt.datetime) -> None:
        if not self.server.restart(env, now):
            raise RuntimeError(f"server failed to restart at {now} with {env}")

    def _restart_with_yaml(self, action, now: dt.datetime) -> None:
        from tools.fault_scenarios import merge_yaml

        self.server.stop(logical_time=now)
        self._write_yaml(merge_yaml(self.yaml_settings, action.yaml_overrides))
        refused = not self.server.start(None, now, timeout=120)
        self.flags["restart_refused"] = refused
        self.notes.append(f"restart with edited settings at {now}: {'refused' if refused else 'STARTED (expected refusal)'}")
        if not refused:
            self.server.stop(logical_time=now)
        self._write_yaml(self.yaml_settings)
        if not self.server.start(None, now):
            raise RuntimeError("server failed to start again after restoring the settings file")

    # -------------------------------------------------------------- run

    def run(self) -> dict:
        from tests.simulate_adapts_hct import HostSimulator, HttpTransport, SimulatorHooks

        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._write_yaml(self.yaml_settings)
        self.server = ServerProcess(self.run_dir, self.port, self._env())
        if not self.server.start():
            raise RuntimeError(f"the API did not start; see {self.run_dir}/server_01.log")
        error = None
        result = None
        try:
            self._boot_monitor()
            hooks = SimulatorHooks(
                tick_every=dt.timedelta(minutes=self.args.tick_minutes) if self.args.tick_minutes > 0 else None,
                on_tick=self.on_tick if self.args.tick_minutes > 0 else None,
                wait_for_update=self.wait_for_update,
                after_request=self.after_request,
                timed=self._timed_hooks(),
            )
            log_file = open(self.run_dir / "requests.log", "w")
            simulator = HostSimulator(
                HttpTransport(f"http://127.0.0.1:{self.port}", timeout=self.args.http_timeout),
                self.plan,
                self.faults,
                hooks,
                log=lambda line: log_file.write(line + "\n"),
            )
            result = simulator.run()
            log_file.close()
        except Exception as exc:  # noqa: BLE001 - always write a report
            error = traceback.format_exc()
            self.notes.append(f"run aborted: {exc!r}")
        finally:
            if self.server is not None:
                self.server.stop(logical_time=None)
        report = self.build_report(result, error)
        (self.run_dir / "report.json").write_text(json.dumps(report, indent=2, default=str))
        (self.run_dir / "report.md").write_text(render_markdown(report))
        if result is not None:
            with open(self.run_dir / "requests.jsonl", "w") as handle:
                for record in result.records:
                    handle.write(json.dumps(record.to_json(), default=str) + "\n")
        return report

    # ----------------------------------------------------------- report

    def build_report(self, result, error: str | None) -> dict:
        from tools.fault_scenarios import BASELINE_ALLOWED_ITEMS, CHANCE_ITEMS, judge_alarms

        plan = self.plan
        db_alarms = self.db.rows(
            "SELECT event_id, item_id, severity, owner, raised_at, source, group_id, decision_type, message FROM monitor_alarms ORDER BY raised_at, id"
        )
        alarm_items: dict[str, Counter] = defaultdict(Counter)
        for alarm in db_alarms:
            alarm_items[alarm["item_id"]][alarm["severity"]] += 1
        alarm_items = {k: dict(v) for k, v in sorted(alarm_items.items())}
        verdict_alarms = judge_alarms(self.scenario, alarm_items)
        pulled_ids = set(result.alarms) if result is not None else set()
        not_pulled = [a["event_id"] for a in db_alarms if a["event_id"] not in pulled_ids]

        checks = {name: self._check(name, result) for name in self.scenario.checks}
        failures = []
        if error:
            failures.append("run aborted (see error)")
        if verdict_alarms["missing"]:
            failures.append(f"expected alarm items missing: {verdict_alarms['missing']}")
        if verdict_alarms["unexplained"]:
            failures.append(f"unexplained red/yellow alarm items: {verdict_alarms['unexplained']}")
        if result is not None and result.unexpected:
            failures.append(f"{len(result.unexpected)} unexpected HTTP statuses")
        if result is not None and result.violations:
            failures.append(f"{len(result.violations)} host-side contract/snapshot violations")
        if result is not None and result.calendar_mismatches():
            failures.append("decision counts differ from the calendar")
        if self.tick_stats.get("error"):
            failures.append(f"{self.tick_stats['error']} monitoring ticks raised")
        for name, outcome in checks.items():
            if not outcome.get("ok"):
                failures.append(f"check {name} failed: {outcome.get('detail')}")

        report = {
            "scenario": self.scenario.to_json(),
            "run_dir": str(self.run_dir),
            "port": self.port,
            "wall_seconds": round(time.monotonic() - self.started_wall, 1),
            "plan": plan.to_json(),
            "faults": self.faults.to_json(),
            "server_events": self.server.events if self.server else [],
            "notes": self.notes,
            "error": error,
            "verdict": {
                "pass": not failures,
                "failures": failures,
                "chance_items": verdict_alarms["chance"],
                "baseline_allowed_items": list(BASELINE_ALLOWED_ITEMS),
                "chance_item_catalog": list(CHANCE_ITEMS),
                "checks": checks,
            },
            "requests": result.counts_by_endpoint() if result else {},
            "unexpected_statuses": [r.to_json() for r in result.unexpected] if result else [],
            "violations": result.violations if result else [],
            "calendar_mismatches": result.calendar_mismatches() if result else [],
            "decisions_by_agent_week": self._decisions_by_week(),
            "pi_by_agent_week": self._pi_by_week(),
            "policy_versions_by_week": self._policy_versions(),
            "updates": self.db.rows(
                "SELECT rid, status, request_timestamp, study_week_monday, started_at, completed_at, published_at_study, error_message FROM model_update_requests ORDER BY id"
            ),
            "update_waits": self.update_waits,
            "alarms": {
                "by_item": alarm_items,
                "n_db": len(db_alarms),
                "n_pulled": len(pulled_ids),
                "in_db_not_pulled": len(not_pulled),
                "first_of_each_item": self._first_alarms(db_alarms),
            },
            "monitoring": {
                "ticks": dict(self.tick_stats),
                "tick_seconds": _summary(self.tick_seconds),
                "tick_errors": self.tick_errors,
                "monitor_runs": self.db.rows("SELECT kind, status, COUNT(*) AS n FROM monitor_runs GROUP BY kind, status"),
            },
            "api_error_codes": self.db.rows("SELECT code, http_status, COUNT(*) AS n FROM api_error_events GROUP BY code, http_status ORDER BY code"),
            "upload_quarantine_rows": (self.db.rows("SELECT COUNT(*) AS n FROM upload_quarantine") or [{"n": None}])[0]["n"],
            "host_side": result.summary() if result else None,
        }
        return report

    def _calendar_week(self, timestamp: str | None) -> int | None:
        if not timestamp:
            return None
        day = dt.datetime.fromisoformat(str(timestamp)).date()
        return self.plan.calendar_week(day)

    def _decisions_by_week(self) -> dict:
        rows = self.db.rows("SELECT decision_type, request_timestamp, is_warmup, warmup_reason FROM actions")
        table: dict[str, dict[int, Counter]] = defaultdict(lambda: defaultdict(Counter))
        for row in rows:
            cell = table[row["decision_type"]][self._calendar_week(row["request_timestamp"])]
            cell["n"] += 1
            if row["is_warmup"]:
                cell[f"warmup:{row['warmup_reason']}"] += 1
            else:
                cell["learned"] += 1
        return {agent: {week: dict(c) for week, c in sorted(weeks.items())} for agent, weeks in sorted(table.items())}

    def _pi_by_week(self) -> dict:
        rows = self.db.rows("SELECT decision_type, request_timestamp, action_prob, action FROM actions WHERE is_warmup = 0")
        values: dict[str, dict[int, list]] = defaultdict(lambda: defaultdict(list))
        for row in rows:
            values[row["decision_type"]][self._calendar_week(row["request_timestamp"])].append(row["action_prob"])
        return {agent: {week: _summary(v) for week, v in sorted(weeks.items())} for agent, weeks in sorted(values.items())}

    def _policy_versions(self) -> dict:
        rows = self.db.rows(
            "SELECT decision_type, study_week_monday, status, n_dyads, published_at_study, reason, update_rid FROM policy_versions ORDER BY study_week_monday, id"
        )
        out: dict[str, list] = defaultdict(list)
        for row in rows:
            out[str(row["study_week_monday"])].append(
                {k: row[k] for k in ("decision_type", "status", "n_dyads", "published_at_study", "reason", "update_rid")}
            )
        return dict(out)

    @staticmethod
    def _first_alarms(db_alarms: list[dict]) -> dict:
        first: dict[str, dict] = {}
        for alarm in db_alarms:
            first.setdefault(alarm["item_id"], alarm)
        return first

    # ------------------------------------------------------- post checks

    def _check(self, name: str, result) -> dict:
        if name == "no_unexplained_alarms":
            return {"ok": True, "detail": "covered by the alarm verdict"}
        if name == "quarantine_rows":
            n = (self.db.rows("SELECT COUNT(*) AS n FROM upload_quarantine") or [{"n": 0}])[0]["n"]
            return {"ok": bool(n), "detail": f"{n} upload_quarantine rows (F-U1)"}
        if name == "update_timed_out":
            rid = self.flags.get("killed")
            rows = self.db.rows("SELECT status, error_message FROM model_update_requests WHERE rid = ?", (rid,)) if rid else []
            ok = bool(rows) and rows[0]["status"] == "failed" and (rows[0]["error_message"] or "") == "timeout"
            return {"ok": ok, "detail": {"rid": rid, "row": rows[0] if rows else None}}
        if name == "restart_refused":
            return {"ok": bool(self.flags.get("restart_refused")), "detail": self.flags.get("restart_refused")}
        if name.startswith("no_item:"):
            item = name.split(":", 1)[1]
            n = len(self.db.rows("SELECT id FROM monitor_alarms WHERE item_id = ?", (item,)))
            return {"ok": n == 0, "detail": f"{n} {item} alarms"}
        if name.startswith("fallback_reason:"):
            reason = name.split(":", 1)[1]
            n = len(self.db.rows("SELECT id FROM actions WHERE warmup_reason = ?", (reason,)))
            return {"ok": n > 0, "detail": f"{n} actions with warmup_reason={reason}"}
        if name == "extra_dyad_partial_week":
            if result is None:
                return {"ok": False, "detail": "no result"}
            extra = [d for d in result.simulator.dyads.values() if d.spec.index >= self.args.dyads]
            problems = []
            for dyad in extra:
                first = sorted(dyad.decisions.values(), key=lambda d: d.time)[:3]
                if dyad.consent_start.weekday() != 1:
                    problems.append(f"{dyad.group_id} does not start on a Tuesday")
                for decision in first:
                    if decision.study_week != 0 or decision.warmup is not True:
                        problems.append(f"{dyad.group_id} {decision.decision_type}#{decision.decision_idx}: week {decision.study_week}, warmup {decision.warmup}")
            return {"ok": bool(extra) and not problems, "detail": problems or [d.group_id for d in extra]}
        return {"ok": False, "detail": f"unknown check {name!r}"}


def _summary(values: list) -> dict:
    values = sorted(float(v) for v in values if v is not None)
    if not values:
        return {"n": 0}

    def q(p: float) -> float:
        k = (len(values) - 1) * p
        lo, hi = int(k), min(int(k) + 1, len(values) - 1)
        return values[lo] + (values[hi] - values[lo]) * (k - lo)

    return {
        "n": len(values),
        "mean": round(statistics.fmean(values), 4),
        "p05": round(q(0.05), 4),
        "p25": round(q(0.25), 4),
        "p50": round(q(0.5), 4),
        "p75": round(q(0.75), 4),
        "p95": round(q(0.95), 4),
        "max": round(values[-1], 4),
    }


# ------------------------------------------------------------------ report


def render_markdown(report: dict) -> str:
    """Human-readable ``report.md`` from ``report.json``."""
    lines: list[str] = []
    scenario = report["scenario"]
    verdict = report["verdict"]
    lines.append(f"# Live trial: {scenario['name']}")
    lines.append("")
    lines.append(f"- Fault table row: {scenario['spec_row']}")
    lines.append(f"- {scenario['description']}")
    plan = report["plan"]
    lines.append(
        f"- Plan: {plan['num_dyads']} dyads, {plan['num_weeks']} calendar weeks from {plan['start_monday']}, seed {plan['seed']}"
    )
    lines.append(f"- Wall time: {report['wall_seconds']} s; run dir `{report['run_dir']}`")
    lines.append(f"- **Verdict: {'PASS' if verdict['pass'] else 'FAIL'}**")
    for failure in verdict["failures"]:
        lines.append(f"  - {failure}")
    if verdict["chance_items"]:
        lines.append(f"  - chance items present (warning only): {verdict['chance_items']}")
    lines.append(f"- Expected alarm items: {scenario['expected_items'] or 'none'}")
    lines.append("")

    lines.append("## Requests by endpoint and status")
    lines.append("")
    lines.append("| endpoint | status counts |")
    lines.append("|---|---|")
    for endpoint, counts in report["requests"].items():
        lines.append(f"| {endpoint} | {', '.join(f'{s}: {n}' for s, n in counts.items())} |")
    lines.append("")

    lines.append("## Alarms by item and severity")
    lines.append("")
    alarms = report["alarms"]
    lines.append(f"{alarms['n_db']} alarm rows; {alarms['n_pulled']} collected by the host's pulls; {alarms['in_db_not_pulled']} never pulled.")
    lines.append(
        "(With CLOCK_MODE=host the pull window ends at the latest host request timestamp, so an alarm raised after the "
        "last request before a pull is collected one pull later; alarms raised after the final pull are never pulled.)"
    )
    lines.append("")
    if alarms["by_item"]:
        lines.append("| item | severities |")
        lines.append("|---|---|")
        for item, severities in alarms["by_item"].items():
            lines.append(f"| {item} | {', '.join(f'{s}: {n}' for s, n in severities.items())} |")
        lines.append("")

    lines.append("## Decisions per agent and calendar week")
    lines.append("")
    for agent, weeks in report["decisions_by_agent_week"].items():
        lines.append(f"**{agent}**")
        lines.append("")
        lines.append("| week | n | learned | warm-up (by reason) |")
        lines.append("|---|---|---|---|")
        for week, cell in weeks.items():
            warm = ", ".join(f"{k.split(':', 1)[1]}: {v}" for k, v in sorted(cell.items()) if k.startswith("warmup:"))
            lines.append(f"| {week} | {cell.get('n', 0)} | {cell.get('learned', 0)} | {warm} |")
        lines.append("")

    lines.append("## Learned pi per agent and calendar week")
    lines.append("")
    for agent, weeks in report["pi_by_agent_week"].items():
        lines.append(f"**{agent}**")
        lines.append("")
        lines.append("| week | n | mean | p05 | p50 | p95 |")
        lines.append("|---|---|---|---|---|---|")
        for week, s in weeks.items():
            lines.append(f"| {week} | {s['n']} | {s.get('mean')} | {s.get('p05')} | {s.get('p50')} | {s.get('p95')} |")
        lines.append("")

    lines.append("## Policy versions per week")
    lines.append("")
    if report["policy_versions_by_week"]:
        lines.append("| week | agent | status | n_dyads | published_at_study |")
        lines.append("|---|---|---|---|---|")
        for week, versions in report["policy_versions_by_week"].items():
            for v in versions:
                lines.append(f"| {week} | {v['decision_type']} | {v['status']} | {v['n_dyads']} | {v['published_at_study']} |")
    else:
        lines.append("No policy versions recorded.")
    lines.append("")

    lines.append("## Updates")
    lines.append("")
    lines.append("| rid | request | status | error |")
    lines.append("|---|---|---|---|")
    for row in report["updates"]:
        lines.append(f"| {row['rid']} | {row['request_timestamp']} | {row['status']} | {row.get('error_message') or ''} |")
    lines.append("")

    lines.append("## Unexpected statuses and host-side violations")
    lines.append("")
    if not report["unexpected_statuses"] and not report["violations"]:
        lines.append("None.")
    grouped: dict[tuple, list] = defaultdict(list)
    for record in report["unexpected_statuses"]:
        grouped[(record["method"], record["endpoint"], record["status"], record.get("title"), record.get("message"))].append(
            record["logical_time"]
        )
    for (method, endpoint, status, title, message), times in grouped.items():
        lines.append(f"- {len(times)} x {method} {endpoint} -> {status} {title}: {message} (first {times[0]}, last {times[-1]})")
    for violation in report["violations"][:50]:
        lines.append(f"- {violation['logical_time']} {violation['what']} {violation.get('group_id')}: {'; '.join(violation['issues'][:3])}")
    lines.append("")

    lines.append("## Monitoring program")
    lines.append("")
    mon = report["monitoring"]
    lines.append(f"Ticks: {mon['ticks']}; seconds per tick: {mon['tick_seconds']}")
    for err in mon["tick_errors"][:5]:
        lines.append(f"- {err['logical_time']}: {err['error']}")
    lines.append("")
    if report["server_events"]:
        lines.append("## Server process events")
        lines.append("")
        for event in report["server_events"]:
            lines.append(f"- {event}")
        lines.append("")
    if report["notes"]:
        lines.append("## Notes")
        lines.append("")
        lines.extend(f"- {note}" for note in report["notes"])
        lines.append("")
    if report["error"]:
        lines.append("## Error")
        lines.append("")
        lines.append("```")
        lines.append(report["error"])
        lines.append("```")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------- CLI


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command")
    serve_parser = sub.add_parser("serve", help="internal: run the API server subprocess")
    serve_parser.add_argument("--port", type=int, required=True)

    parser.add_argument("--scenario", action="append", help="scenario name (repeatable) or 'all'; default baseline")
    parser.add_argument("--list", action="store_true", help="list the scenarios and exit")
    parser.add_argument("--dyads", type=int, default=10)
    parser.add_argument("--weeks", type=int, default=16, help="calendar weeks (16+ for the fault table)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start-date", type=dt.date.fromisoformat, default=DEFAULT_START)
    parser.add_argument("--port", type=int, default=None, help="server port (default: a free port)")
    parser.add_argument("--run-dir", type=Path, default=None, help="parent directory for run dirs (default: a temp dir)")
    parser.add_argument(
        "--tick-minutes",
        type=int,
        default=15,
        help="logical minutes between monitoring runs (0 = none); above 15 every run looks like an outage (E5)",
    )
    parser.add_argument("--update-timeout", type=float, default=900.0, help="wall seconds to wait for one /update")
    parser.add_argument("--http-timeout", type=float, default=120.0)
    parser.add_argument("--analytics-gap-rate", type=float, default=0.02)
    parser.add_argument("--keep", action="store_true", help="keep an existing run dir (default: replace it)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "serve":
        serve(args.port)
        return 0
    from tools.fault_scenarios import SCENARIOS

    if args.list:
        for scenario in SCENARIOS.values():
            print(f"{scenario.name:22s} expects {list(scenario.expected_items) or '-'}: {scenario.spec_row}")
        return 0
    names = args.scenario or ["baseline"]
    if "all" in names:
        names = list(SCENARIOS)
    unknown = [n for n in names if n not in SCENARIOS]
    if unknown:
        raise SystemExit(f"unknown scenario(s) {unknown}; see --list")
    parent = args.run_dir or Path(tempfile.mkdtemp(prefix="adapts_live_trial_"))
    parent = parent.resolve()
    parent.mkdir(parents=True, exist_ok=True)
    if len(names) > 1 and args.port:
        raise SystemExit("--port cannot be combined with several scenarios (each run starts its own server)")

    overall = {}
    for name in names:
        run_dir = parent / name
        if run_dir.exists() and not args.keep:
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"[live_trial] {name}: run dir {run_dir}", flush=True)
        if len(names) > 1:
            # Each scenario in a fresh process: create_app and config read the environment once per process.
            code = subprocess.call(
                [sys.executable, str(Path(__file__).resolve()), "--scenario", name, "--dyads", str(args.dyads),
                 "--weeks", str(args.weeks), "--seed", str(args.seed), "--start-date", args.start_date.isoformat(),
                 "--run-dir", str(parent), "--tick-minutes", str(args.tick_minutes), "--update-timeout",
                 str(args.update_timeout), "--http-timeout", str(args.http_timeout), "--analytics-gap-rate",
                 str(args.analytics_gap_rate), "--keep"],
                cwd=str(REPO_ROOT),
            )
            report_path = run_dir / "report.json"
            overall[name] = json.loads(report_path.read_text())["verdict"]["pass"] if report_path.exists() else False
            print(f"[live_trial] {name}: {'PASS' if overall[name] else 'FAIL'} (exit {code})", flush=True)
            continue
        report = LiveTrial(SCENARIOS[name], args, run_dir).run()
        overall[name] = report["verdict"]["pass"]
        print(f"[live_trial] {name}: {'PASS' if overall[name] else 'FAIL'}", flush=True)
        for failure in report["verdict"]["failures"]:
            print(f"    - {failure}", flush=True)
        print(f"    report: {run_dir / 'report.md'}", flush=True)
    if len(names) > 1:
        (parent / "summary.json").write_text(json.dumps(overall, indent=2))
        print(json.dumps(overall, indent=2))
    return 0 if all(overall.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
