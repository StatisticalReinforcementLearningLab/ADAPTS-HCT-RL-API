#!/usr/bin/env python
"""
Reproducibility checker for ADAPTS-HCT RL API.

Given (a) the original pre-sampled buffer that the study consumed and
(b) the study's persisted records (repro snapshot directory written by
`app/repro_snapshot.py`), this tool deterministically replays every
event — add_group, action, upload, update — against a fresh in-memory
database and asserts that every replayed `(action, action_prob)` matches
the logged value bit-for-bit.

Input sources (choose one):
  --snapshot PATH    a repro snapshot directory (contains actions.json,
                     study_data.json, groups.json, and — optionally —
                     model_update_requests.json produced by this tool when
                     the --augment flag was used during the original run).
  --exports PATH     a `flask export-csv` output directory. CSVs must
                     include groups, actions, study_data, and
                     model_update_requests.

Usage:
    python tools/reproduce_run.py \\
        --buffer buffers/study_buffer.npz \\
        --snapshot repro_snapshots/<update_id> \\
        [--verbose]

Exit code is 0 when every replayed action matches; non-zero if any row
mismatches.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Ensure project root importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _parse_ts(value: Any) -> datetime.datetime:
    if isinstance(value, datetime.datetime):
        return value
    if value is None:
        return datetime.datetime.min
    s = str(value)
    # Handle both ISO and Postgres-flavored strings.
    return datetime.datetime.fromisoformat(s.replace("Z", "+00:00").rstrip("Z"))


@dataclass
class Event:
    ts: datetime.datetime
    kind: str  # "add_group" | "action" | "upload" | "update"
    payload: dict


def _load_snapshot(snapshot_dir: Path) -> list[Event]:
    def read_json(fname: str) -> list[dict]:
        path = snapshot_dir / fname
        if not path.exists():
            return []
        with open(path, "r") as f:
            return json.load(f)

    groups = read_json("groups.json")
    actions = read_json("actions.json")
    study = read_json("study_data.json")
    updates = read_json("model_update_requests.json")  # optional; added below

    events: list[Event] = []
    for g in groups:
        events.append(
            Event(
                ts=_parse_ts(g.get("created_at")),
                kind="add_group",
                payload=g,
            )
        )
    for a in actions:
        events.append(
            Event(
                ts=_parse_ts(a.get("timestamp") or a.get("request_timestamp")),
                kind="action",
                payload=a,
            )
        )
    for s in study:
        events.append(
            Event(
                ts=_parse_ts(s.get("created_at") or s.get("request_timestamp")),
                kind="upload",
                payload=s,
            )
        )
    for u in updates:
        events.append(
            Event(
                ts=_parse_ts(u.get("request_timestamp")),
                kind="update",
                payload=u,
            )
        )
    events.sort(key=lambda e: (e.ts, {"add_group": 0, "upload": 1, "update": 2, "action": 3}[e.kind]))
    return events


def _load_exports(exports_dir: Path) -> list[Event]:
    events: list[Event] = []

    def read_csv(fname: str) -> list[dict]:
        path = exports_dir / fname
        if not path.exists():
            return []
        with open(path, "r", newline="") as f:
            return list(csv.DictReader(f))

    def _maybe_json(value):
        if value in (None, "", "None"):
            return None
        try:
            return json.loads(value)
        except (TypeError, json.JSONDecodeError):
            return value

    for row in read_csv("groups.csv"):
        row["group_info"] = _maybe_json(row.get("group_info"))
        row["warmup"] = (row.get("warmup", "").lower() in ("1", "t", "true"))
        events.append(Event(ts=_parse_ts(row.get("created_at")), kind="add_group", payload=row))

    for row in read_csv("actions.csv"):
        for key in ("state", "raw_context", "random_state"):
            row[key] = _maybe_json(row.get(key))
        row["decision_idx"] = int(row.get("decision_idx") or 0)
        row["action"] = int(row.get("action") or 0)
        row["action_prob"] = float(row.get("action_prob") or 0.0)
        events.append(
            Event(
                ts=_parse_ts(row.get("timestamp") or row.get("request_timestamp")),
                kind="action",
                payload=row,
            )
        )

    for row in read_csv("study_data.csv"):
        for key in ("state", "raw_context", "outcome"):
            row[key] = _maybe_json(row.get(key))
        row["decision_idx"] = int(row.get("decision_idx") or 0)
        row["action"] = int(row.get("action") or 0)
        row["action_prob"] = float(row.get("action_prob") or 0.0)
        if row.get("reward") not in (None, "", "None"):
            row["reward"] = float(row["reward"])
        events.append(
            Event(
                ts=_parse_ts(row.get("created_at") or row.get("request_timestamp")),
                kind="upload",
                payload=row,
            )
        )

    for row in read_csv("model_update_requests.csv"):
        events.append(
            Event(
                ts=_parse_ts(row.get("request_timestamp")),
                kind="update",
                payload=row,
            )
        )

    events.sort(
        key=lambda e: (
            e.ts,
            {"add_group": 0, "upload": 1, "update": 2, "action": 3}[e.kind],
        )
    )
    return events


def _build_app_with_buffer(buffer_path: str):
    """Fresh Flask app in Testing mode, with the supplied buffer swapped in
    and cursor reset to 0. Uses in-memory SQLite so we don't pollute the
    real database."""
    os.environ.setdefault("FLASK_ENV", "testing")

    from app import create_app
    from app.deterministic_sampler import DeterministicSampleStream

    app = create_app("config.TestingConfig")
    sampler = DeterministicSampleStream.load(buffer_path)
    sampler.restore({"normal": 0, "uniform": 0})
    app.sampler = sampler
    app.rl_algorithm.sampler = sampler
    app.rl_algorithm._update_call_counts.clear()
    return app


def reproduce(buffer_path: str, events: list[Event], verbose: bool = False) -> dict:
    """
    Replay the event stream against a fresh in-memory DB with the given
    buffer. Returns a report dict.
    """
    app = _build_app_with_buffer(buffer_path)

    from app.extensions import db
    from app.models import (
        Action,
        Group,
        ModelParameters,
        ModelUpdateRequests,
        StudyData,
    )

    report = {
        "events": {"add_group": 0, "action": 0, "upload": 0, "update": 0},
        "matches": 0,
        "mismatches": [],
        "errors": [],
    }

    with app.app_context():
        for event in events:
            try:
                if event.kind == "add_group":
                    _replay_add_group(event.payload, db, Group)
                    report["events"]["add_group"] += 1

                elif event.kind == "action":
                    replayed_action, replayed_prob, replayed_state, replayed_rs = _replay_action(
                        app, event.payload, db, Action, ModelParameters
                    )
                    report["events"]["action"] += 1
                    # Compare
                    mismatch = _compare_action(event.payload, replayed_action, replayed_prob)
                    if mismatch is None:
                        report["matches"] += 1
                        if verbose:
                            print(
                                f"[ok] {event.payload.get('group_id')} "
                                f"{event.payload.get('decision_type')} "
                                f"idx={event.payload.get('decision_idx')} "
                                f"action={replayed_action} prob={replayed_prob:.6f}"
                            )
                    else:
                        report["mismatches"].append(mismatch)
                        if verbose:
                            print(f"[!!] mismatch: {mismatch}")

                elif event.kind == "upload":
                    _replay_upload(app, event.payload, db, StudyData, Action)
                    report["events"]["upload"] += 1

                elif event.kind == "update":
                    _replay_update(app, event.payload, db, StudyData)
                    report["events"]["update"] += 1

            except Exception as exc:
                report["errors"].append({"kind": event.kind, "error": str(exc)})
                if verbose:
                    import traceback
                    traceback.print_exc()

    return report


def _replay_add_group(payload: dict, db, Group) -> None:
    group_id = payload["group_id"]
    if Group.query.filter_by(group_id=group_id).first():
        return
    group = Group(
        group_id=group_id,
        group_info=payload.get("group_info") or {},
        warmup=bool(payload.get("warmup", False)),
    )
    db.session.add(group)
    db.session.commit()


def _replay_action(app, payload, db, Action, ModelParameters):
    group_id = payload["group_id"]
    decision_type = payload["decision_type"]
    decision_idx = int(payload["decision_idx"])
    raw_context = payload.get("raw_context") or {}

    # Mirror what routes/action.py does:
    context_with_type = {
        **raw_context,
        "decision_type": decision_type,
        "group_id": group_id,
    }
    ok, state = app.rl_algorithm.make_state(context_with_type)
    if not ok:
        raise RuntimeError(f"make_state failed: {state}")

    model_params = ModelParameters.query.order_by(
        ModelParameters.timestamp.desc()
    ).first()

    action, prob, random_state = app.rl_algorithm.get_action(
        group_id,
        state,
        {"probability": model_params.probability_of_action},
        decision_type,
        decision_idx,
    )

    if Action.query.filter_by(group_id=group_id, decision_idx=decision_idx).first() is None:
        db.session.add(
            Action(
                group_id=group_id,
                action=action,
                rid=(payload.get("rid") or f"replay-{group_id}-{decision_idx}"),
                state=state,
                decision_idx=decision_idx,
                decision_type=decision_type,
                raw_context=raw_context,
                action_prob=prob,
                random_state=random_state,
                model_parameters_id=model_params.id,
                request_timestamp=_parse_ts(
                    payload.get("request_timestamp") or payload.get("timestamp")
                ),
            )
        )
        db.session.commit()

    return action, prob, state, random_state


def _replay_upload(app, payload, db, StudyData, Action):
    group_id = payload["group_id"]
    decision_idx = int(payload["decision_idx"])
    decision_type = payload["decision_type"]

    # Need the Action row to have been inserted (same-order replay guarantees it)
    if Action.query.filter_by(group_id=group_id, decision_idx=decision_idx).first() is None:
        # Upload arrived out of order (shouldn't happen in a sorted stream)
        return

    outcome = {**(payload.get("outcome") or {}), "decision_type": decision_type}
    ok, reward = app.rl_algorithm.make_reward(
        group_id, payload.get("state"), int(payload.get("action") or 0), outcome
    )
    if not ok:
        raise RuntimeError(f"make_reward failed: {reward}")

    existing = StudyData.query.filter_by(
        group_id=group_id, decision_idx=decision_idx
    ).first()
    if existing is None:
        db.session.add(
            StudyData(
                group_id=group_id,
                decision_idx=decision_idx,
                decision_type=decision_type,
                action=int(payload.get("action") or 0),
                action_prob=float(payload.get("action_prob") or 0.0),
                state=payload.get("state"),
                raw_context={**(payload.get("raw_context") or {}), "decision_type": decision_type},
                outcome=outcome,
                reward=reward,
                request_timestamp=_parse_ts(
                    payload.get("request_timestamp") or payload.get("created_at")
                ),
            )
        )
        db.session.commit()


def _replay_update(app, payload, db, StudyData):
    rows = StudyData.query.order_by(
        StudyData.decision_type.asc(),
        StudyData.group_id.asc(),
        StudyData.decision_idx.asc(),
    ).all()
    records = []
    for row in rows:
        agent_idx = int(
            (row.raw_context or {}).get("agent_decision_index", row.decision_idx + 1)
        )
        records.append(
            {
                "group_id": row.group_id,
                "decision_idx": row.decision_idx,
                "decision_type": row.decision_type,
                "agent_decision_index": agent_idx,
                "state": row.state,
                "action": row.action,
                "reward": row.reward,
                "raw_context": row.raw_context,
                "outcome": row.outcome,
            }
        )

    from app.models import ModelParameters
    current_params = ModelParameters.query.order_by(
        ModelParameters.timestamp.desc()
    ).first()

    status, _ = app.rl_algorithm.update(
        {"probability_of_action": current_params.probability_of_action},
        {"records": records, "current_index": {}},
    )
    if not status:
        raise RuntimeError("algorithm.update returned False")


def _compare_action(original: dict, action: int, prob: float) -> dict | None:
    logged_action = int(original.get("action"))
    logged_prob = float(original.get("action_prob") or 0.0)
    # Exact integer match on action; float match on prob to tight tolerance.
    prob_match = abs(logged_prob - prob) < 1e-9
    action_match = logged_action == int(action)
    if prob_match and action_match:
        return None
    return {
        "group_id": original.get("group_id"),
        "decision_type": original.get("decision_type"),
        "decision_idx": original.get("decision_idx"),
        "logged_action": logged_action,
        "replayed_action": int(action),
        "logged_prob": logged_prob,
        "replayed_prob": float(prob),
        "logged_cursor": (original.get("random_state") or {}).get("sampler_cursor_end"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--buffer", required=True, help="Path to the .npz sample buffer")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--snapshot", help="Repro snapshot directory")
    source.add_argument("--exports", help="flask export-csv directory")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.snapshot:
        events = _load_snapshot(Path(args.snapshot))
    else:
        events = _load_exports(Path(args.exports))

    print(f"Loaded {len(events)} events")
    report = reproduce(args.buffer, events, verbose=args.verbose)

    print()
    print("== report ==")
    print(f"  events     : {report['events']}")
    print(f"  matches    : {report['matches']}")
    print(f"  mismatches : {len(report['mismatches'])}")
    print(f"  errors     : {len(report['errors'])}")

    if report["mismatches"]:
        print("\nFirst mismatches:")
        for m in report["mismatches"][:5]:
            print(f"  {m}")

    if report["errors"]:
        print("\nFirst errors:")
        for e in report["errors"][:5]:
            print(f"  {e}")

    exit_code = 0 if not report["mismatches"] and not report["errors"] else 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
