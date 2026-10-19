"""
Order-of-magnitude resource estimate for a trial, from the host simulator.

The host calendar (API-Spec §3) is played without a server
(``OfflineTransport``): the request counts and payload sizes are exact for the
simulated calendar; database rows written by ``/update`` (contract D14, D15:
``local_fit`` / ``hyper`` / ``posterior`` rows in ``model_parameters`` and one
``policy_versions`` row per agent per learning update) are counted from the
enrollment calendar and the cohort warm-up rule (contract D2).

``python tests/resource_estimate.py [--dyads 25] [--weeks 30]`` prints the JSON.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.simulate_adapts_hct import (  # noqa: E402
    HostSimulator,
    OfflineTransport,
    TrialPlan,
    week1_start,
    week_monday,
)

NUM_DYADS = 25
DEFAULT_NUM_WEEKS = 30

# Design dimensions (main_new.tex Table 3; contract D7): local fit theta = [beta (p_h), alpha (p_f)].
P_H = {"aya_message": 7, "cp_message": 6, "dyad_game": 7}
P_F = {"aya_message": 4, "cp_message": 4, "dyad_game": 2}
AGENTS = tuple(P_H)
N_WARM = 5  # warmup.n_warm (main_new.tex Table 1)

_JSON_NUMBER_BYTES = 22  # a float rendered in JSON, with separator
_ROW_OVERHEAD_BYTES = 200  # ids, timestamps, rids, metadata keys
_APP_LOG_LINE_BYTES_EST = 200  # method/path/status/size only, never bodies (app/__init__.py)
_RL_LOG_LINE_BYTES_EST = 600


def _matrix_row_bytes(p: int) -> int:
    """A model_parameters row holding a length-p mean and a p x p covariance as JSON."""
    return _ROW_OVERHEAD_BYTES + _JSON_NUMBER_BYTES * (p + p * p)


def estimate_trial_resources(
    base_date: dt.date | None = None,
    num_weeks: int = DEFAULT_NUM_WEEKS,
    num_dyads: int = NUM_DYADS,
    seed: int = 42,
) -> dict:
    plan = TrialPlan(
        num_dyads=num_dyads,
        num_weeks=num_weeks,
        start_date=base_date or dt.date(2026, 11, 2),
        seed=seed,
        record_payloads=True,
        pull_alarms=True,
    )
    simulator = HostSimulator(OfflineTransport(seed=seed), plan)
    result = simulator.run()

    counts: Counter = Counter()
    request_bytes: Counter = Counter()
    weekly_actions: Counter = Counter()
    hourly_actions: Counter = Counter()
    for record in result.records:
        counts[record.endpoint] += 1
        request_bytes[record.endpoint] += len(json.dumps(record.request or {}, separators=(",", ":")))
        if record.endpoint == "action":
            weekly_actions[week_monday(record.logical_time.date())] += 1
            hourly_actions[record.logical_time.replace(minute=0, second=0)] += 1
    counts["add_group"] = counts["register_group"]

    # Learning updates and the rows they write (contract D2, D8, D9, D14, D15).
    specs = list(simulator.dyads.values())
    model_rows = Counter()
    model_bytes = 0
    learning_updates = 0
    for record in result.records:
        if record.endpoint != "update":
            continue
        monday = record.logical_time.date()
        completed_week1 = sum(1 for d in specs if week1_start(d.consent_start) <= monday - dt.timedelta(days=7))
        if completed_week1 < N_WARM:
            continue
        learning_updates += 1
        enrolled_with_data = [d for d in specs if d.consent_start < monday]
        active = [d for d in specs if 1 <= (monday - week1_start(d.consent_start)).days // 7 + 1 <= plan.trial_weeks]
        for agent in AGENTS:
            p_full, p_f = P_H[agent] + P_F[agent], P_F[agent]
            model_rows["local_fit"] += len(enrolled_with_data)
            model_rows["hyper"] += 1
            model_rows["posterior"] += len(active)
            model_bytes += len(enrolled_with_data) * _matrix_row_bytes(p_full)
            model_bytes += _matrix_row_bytes(p_f) + len(active) * _matrix_row_bytes(p_f)

    avg = {name: (request_bytes[name] / counts[name]) if counts[name] else 0.0 for name in request_bytes}
    row_counts = {
        "groups": counts["register_group"],
        "group_history": counts["register_group"],
        "actions": counts["action"],
        "data_uploads": counts["upload_data"],
        "study_data": counts["action"],
        "model_update_requests": counts["update"],
        "policy_versions": learning_updates * len(AGENTS),
        "model_parameters": 1 + sum(model_rows.values()),
        "monitor_pulls": counts["monitor_alarms"],
    }
    action_row_bytes = _ROW_OVERHEAD_BYTES + avg.get("upload_data", 0.0) + _JSON_NUMBER_BYTES * 24  # raw_context + state
    estimated_storage_bytes = int(
        row_counts["data_uploads"] * (avg.get("upload_data", 0.0) + _ROW_OVERHEAD_BYTES)
        + row_counts["actions"] * action_row_bytes
        + row_counts["study_data"] * action_row_bytes
        + row_counts["groups"] * 2 * (avg.get("register_group", 0.0) + _ROW_OVERHEAD_BYTES)
        + (row_counts["model_update_requests"] + row_counts["policy_versions"] + row_counts["monitor_pulls"]) * _ROW_OVERHEAD_BYTES
        + model_bytes
    )

    logging_budget = _estimate_logging_bytes(dict(counts))
    repro_bytes = int(estimated_storage_bytes * 0.85 * 1.05) * counts["update"]
    ram_profile = _estimate_ram_mb(row_counts["study_data"], max(P_H[a] + P_F[a] for a in AGENTS), row_counts["groups"])
    peak_hourly = max(hourly_actions.values(), default=0)
    core_bytes = estimated_storage_bytes + logging_budget["total_log_bytes_est"] + repro_bytes

    first = min(d.consent_start for d in specs)
    last = max(min(d.spec.consent_end, plan.horizon_last_day) for d in specs)
    return {
        "event_counts": dict(counts),
        "row_counts": row_counts,
        "model_parameter_rows": dict(model_rows),
        "learning_updates": learning_updates,
        "average_payload_bytes": {k: round(v, 1) for k, v in avg.items()},
        "estimated_storage_bytes": estimated_storage_bytes,
        "estimated_storage_mb": round(estimated_storage_bytes / (1024 * 1024), 3),
        "traffic_profile": {
            "peak_weekly_actions": max(weekly_actions.values(), default=0),
            "peak_hourly_actions": peak_hourly,
        },
        "hosting_guidance": _hosting_guidance(core_bytes, peak_hourly),
        "logging_budget": logging_budget,
        "repro_snapshots_bytes_est": repro_bytes,
        "ram_profile_mb": ram_profile,
        "feature_dims": {a: {"p_h": P_H[a], "p_f": P_F[a]} for a in AGENTS},
        "study_calendar": {
            "first_consent_date": first.isoformat(),
            "last_decision_date": last.isoformat(),
            "calendar_days_inclusive": (last - first).days + 1,
            "dyad_count": len(specs),
            "scheduled_weekly_updates": counts["update"],
        },
        "rounded_trial_storage_gb_est": round(core_bytes / (1024**3), 2),
    }


def _estimate_logging_bytes(event_counts: dict) -> dict:
    """Two short HTTP log lines per call plus learner lines per decision/update (linear growth, before rotation)."""
    n_http = sum(event_counts.get(k, 0) for k in ("register_group", "action", "upload_data", "update", "monitor_alarms"))
    app_bytes = n_http * 2 * _APP_LOG_LINE_BYTES_EST
    rl_bytes = event_counts.get("action", 0) * _RL_LOG_LINE_BYTES_EST + event_counts.get("update", 0) * 4 * _RL_LOG_LINE_BYTES_EST
    total = app_bytes + rl_bytes
    return {
        "http_events_logged": n_http,
        "app_log_bytes_est": app_bytes,
        "rl_log_bytes_est": rl_bytes,
        "total_log_bytes_est": total,
        "total_log_mb_est": round(total / (1024 * 1024), 3),
    }


def _estimate_ram_mb(n_study_rows: int, p_max: int, n_groups: int) -> dict:
    """Rough peak RAM of a weekly update (all rows in memory plus dense p x p algebra per dyad)."""
    base_mb = 280.0
    rows_mb = n_study_rows * p_max * 8 * 3 / (1024 * 1024)
    per_dyad_mb = n_groups * p_max * p_max * 8 * 6 / (1024 * 1024)
    return {
        "baseline_process_mb_est": base_mb,
        "peak_update_mb_est": round(base_mb + rows_mb + per_dyad_mb, 1),
    }


def _hosting_guidance(storage_bytes: int, peak_hourly_actions: int) -> dict:
    storage_mb = storage_bytes / (1024 * 1024)
    return {
        "recommended_db_storage_mb": max(256, round(storage_mb * 10)),
        "recommended_app_instances": 1 if peak_hourly_actions < 500 else 2,
        "notes": [
            "Traffic comes in scheduled bursts (07:30, 07:35, 19:30, Monday 06:00), not continuous load.",
            "The weekly update is small dense algebra (p <= 11 per dyad) plus one EB optimization per agent.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dyads", type=int, default=NUM_DYADS)
    parser.add_argument("--weeks", type=int, default=DEFAULT_NUM_WEEKS)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    print(json.dumps(estimate_trial_resources(num_weeks=args.weeks, num_dyads=args.dyads, seed=args.seed), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
