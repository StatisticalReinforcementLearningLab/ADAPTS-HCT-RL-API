from __future__ import annotations

import datetime
import json
from collections import defaultdict

from tests.simulate_adapts_hct import DEFAULT_NUM_WEEKS, NUM_DYADS, ProtocolTrialSimulator

from app.feature_builder import phi_dims_by_decision_type


FEATURE_DIMENSIONS = phi_dims_by_decision_type()

# Two short lines per HTTP round-trip (method/path/size only — no bodies; see app/__init__.py)
_APP_LOG_LINE_BYTES_EST = 200
_RL_LOG_LINE_BYTES_EST = 600


def estimate_trial_resources(
    base_date: datetime.date | None = None,
    num_weeks: int = DEFAULT_NUM_WEEKS,
    num_dyads: int = NUM_DYADS,
    seed: int = 42,
) -> dict:
    if base_date is None:
        base_date = datetime.date(2025, 1, 5)

    simulator = ProtocolTrialSimulator(
        base_date=base_date,
        num_weeks=num_weeks,
        num_dyads=num_dyads,
        seed=seed,
    )

    counts = defaultdict(int)
    bytes_seen = defaultdict(int)
    observed_groups: dict[str, set[str]] = defaultdict(set)
    eb_snapshot_rows = 0
    matrix_inversions = 0
    local_fit_rows = 0
    study_records_by_type_group = defaultdict(int)

    def deterministic_action(payload: dict) -> int:
        decision_type = payload["decision_type"]
        context = payload["context"]
        if decision_type == "dyad_game":
            return int(context["aya_app_engagement"] + context["cp_app_engagement"] >= 6)
        if decision_type == "cp_message":
            return int(context["cp_app_engagement"] >= 3 or context["cp_missing_rate_7d"] > 0.4)
        return int(context["aya_app_engagement"] >= 3 or context["aya_missing_rate_7d"] > 0.4)

    def size_of(value: dict) -> int:
        return len(json.dumps(value, sort_keys=True).encode("utf-8"))

    def submit_uploads(payloads: list[dict]):
        nonlocal bytes_seen, counts, study_records_by_type_group
        for upload_payload in payloads:
            counts["upload_data"] += 1
            bytes_seen["upload_data"] += size_of(upload_payload)
            study_records_by_type_group[(upload_payload["decision_type"], upload_payload["group_id"])] += 1

    for event in simulator.iter_schedule_events():
        submit_uploads(simulator.pop_due_uploads(event["timestamp"]))

        if event["type"] == "add_group":
            counts["add_group"] += 1
            bytes_seen["add_group"] += size_of(event["payload"])
            continue

        if event["type"] == "update":
            counts["update"] += 1
            bytes_seen["update"] += size_of(event["payload"])
            for decision_type, group_ids in observed_groups.items():
                if not group_ids:
                    continue
                eb_snapshot_rows += 1  # hyper snapshot
                matrix_inversions += len(group_ids) + 1
                for group_id in group_ids:
                    local_fit_rows += 2
                    feature_dim = FEATURE_DIMENSIONS.get(
                        decision_type, FEATURE_DIMENSIONS["aya_message"]
                    )
                    counts[f"{decision_type}_fit_flops"] += (
                        study_records_by_type_group[(decision_type, group_id)] * feature_dim * feature_dim
                    )
            continue

        payload = simulator.build_action_payload(event)
        counts["action"] += 1
        bytes_seen["action"] += size_of(payload)
        action = deterministic_action(payload)
        simulated_response = {
            "action": action,
            "action_prob": 0.5,
            "state": [0.0] * max(1, len(payload["context"])),
        }
        simulator.schedule_upload(payload, simulated_response)
        observed_groups[payload["decision_type"]].add(payload["group_id"])

    submit_uploads(simulator.flush_all_uploads())

    # EB snapshot rows now live in the unified ``model_parameters`` table:
    # one bootstrap row at app init, one "policy" row per /update, plus
    # the local-fit / hyper / posterior snapshot rows that were formerly
    # in the ``empirical_bayes_snapshots`` table.
    row_counts = {
        "groups": counts["add_group"],
        "actions": counts["action"],
        "study_data": counts["upload_data"],
        "model_update_requests": counts["update"],
        "model_parameters": counts["update"] + 1 + eb_snapshot_rows + local_fit_rows,
    }

    average_payload_bytes = {
        name: (bytes_seen[name] / counts[name]) if counts[name] else 0
        for name in ("add_group", "action", "upload_data", "update")
    }

    # EB snapshot rows are ~2048 bytes (theta + covariance JSON); the legacy
    # policy rows are ~128 bytes. Estimate the mix using the snapshot-row count.
    eb_snapshot_total = eb_snapshot_rows + local_fit_rows
    legacy_policy_rows = row_counts["model_parameters"] - eb_snapshot_total
    estimated_storage_bytes = int(
        row_counts["groups"] * average_payload_bytes["add_group"]
        + row_counts["actions"] * average_payload_bytes["action"]
        + row_counts["study_data"] * average_payload_bytes["upload_data"]
        + row_counts["model_update_requests"] * average_payload_bytes["update"]
        + eb_snapshot_total * 2048
        + legacy_policy_rows * 128
    )

    logging_budget = _estimate_logging_bytes(dict(counts))
    repro_snapshot_bytes = _estimate_repro_snapshot_bytes(
        dict(counts), estimated_data_json_bytes=int(estimated_storage_bytes * 0.85)
    )
    ram_profile = _estimate_ram_mb(
        n_study_rows=row_counts["study_data"],
        phi_dims=list(FEATURE_DIMENSIONS.values()),
        n_groups=row_counts["groups"],
    )

    peak_weekly_actions = _peak_weekly_actions(base_date=base_date, num_weeks=num_weeks, num_dyads=num_dyads)
    peak_concurrent_hourly_actions = _peak_hourly_actions(
        base_date=base_date, num_weeks=num_weeks, num_dyads=num_dyads
    )

    study_calendar = _study_calendar_bounds(simulator.dyads)
    study_calendar["scheduled_weekly_updates"] = num_weeks
    core_bytes = (
        estimated_storage_bytes
        + logging_budget["total_log_bytes_est"]
        + repro_snapshot_bytes
    )
    rounded_gb = round(core_bytes / (1024**3), 2)

    return {
        "event_counts": dict(counts),
        "row_counts": row_counts,
        "average_payload_bytes": average_payload_bytes,
        "estimated_storage_bytes": estimated_storage_bytes,
        "estimated_storage_mb": round(estimated_storage_bytes / (1024 * 1024), 3),
        "compute_profile": {
            "matrix_inversions": matrix_inversions,
            "local_fit_snapshot_rows": local_fit_rows,
            "hyper_snapshot_rows": eb_snapshot_rows,
        },
        "traffic_profile": {
            "peak_weekly_actions": peak_weekly_actions,
            "peak_hourly_actions": peak_concurrent_hourly_actions,
        },
        "hosting_guidance": _hosting_guidance(core_bytes, peak_concurrent_hourly_actions),
        "logging_budget": logging_budget,
        "repro_snapshots_bytes_est": repro_snapshot_bytes,
        "ram_profile_mb": ram_profile,
        "feature_phi_dims": dict(FEATURE_DIMENSIONS),
        "study_calendar": study_calendar,
        "rounded_trial_storage_gb_est": rounded_gb,
    }


def _peak_weekly_actions(base_date: datetime.date, num_weeks: int, num_dyads: int) -> int:
    simulator = ProtocolTrialSimulator(base_date=base_date, num_weeks=num_weeks, num_dyads=num_dyads, seed=42)
    weekly_counts = defaultdict(int)
    for event in simulator.iter_schedule_events():
        if event["type"] != "action":
            continue
        week_idx = ((_timestamp_to_date(event["timestamp"]) - base_date).days // 7) + 1
        weekly_counts[week_idx] += 1
    return max(weekly_counts.values(), default=0)


def _peak_hourly_actions(base_date: datetime.date, num_weeks: int, num_dyads: int) -> int:
    simulator = ProtocolTrialSimulator(base_date=base_date, num_weeks=num_weeks, num_dyads=num_dyads, seed=42)
    hourly_counts = defaultdict(int)
    for event in simulator.iter_schedule_events():
        if event["type"] != "action":
            continue
        event_dt = datetime.datetime.fromisoformat(event["timestamp"]).replace(minute=0, second=0, microsecond=0)
        hourly_counts[event_dt.isoformat()] += 1
    return max(hourly_counts.values(), default=0)


def _timestamp_to_date(value: str) -> datetime.date:
    return datetime.datetime.fromisoformat(value).date()


def _study_calendar_bounds(dyads: list) -> dict:
    """
    Calendar span from first consent/recruit day through last dyad's active
    consent end. Weekly /update snapshots in the simulator align with
    ``num_weeks`` (one per scheduled Sunday), typically within one week of
    ceil(span_days/7) when recruitment is staggered by one week per dyad.
    """
    first_recruit = min(d.recruit_date for d in dyads)
    last_consent_end = max(d.consent_end_date for d in dyads)
    span_days = (last_consent_end - first_recruit).days + 1
    calendar_weeks_ceil = (span_days + 6) // 7
    return {
        "first_recruit_date": first_recruit.isoformat(),
        "last_consent_end_date": last_consent_end.isoformat(),
        "calendar_days_inclusive": span_days,
        "calendar_weeks_ceil": calendar_weeks_ceil,
        "dyad_count": len(dyads),
    }


def _estimate_logging_bytes(event_counts: dict) -> dict:
    """
    Rough order-of-magnitude for on-disk logs.

    HTTP lines are short (method/path/status/content-length only; see
    ``app/__init__.py``). Rotating handlers are capped in ``logging_config.py``
    — tune separately from this linear growth estimate.
    """
    n_http = (
        event_counts.get("add_group", 0)
        + event_counts.get("action", 0)
        + event_counts.get("upload_data", 0)
        + event_counts.get("update", 0)
    )
    app_bytes = n_http * 2 * _APP_LOG_LINE_BYTES_EST
    rl_bytes = (
        event_counts.get("action", 0) * _RL_LOG_LINE_BYTES_EST
        + event_counts.get("update", 0) * (4 * _RL_LOG_LINE_BYTES_EST)
    )
    total = app_bytes + rl_bytes
    return {
        "http_events_logged": n_http,
        "app_log_bytes_est": app_bytes,
        "rl_log_bytes_est": rl_bytes,
        "total_log_bytes_est": total,
        "total_log_mb_est": round(total / (1024 * 1024), 3),
        "rotating_handler_note": "Each of app.log and rl.log can rotate up to 100 MiB × 5000 backup files "
        "if log volume is sustained; tune RotatingFileHandler maxBytes/backupCount in logging_config.py.",
    }


def _estimate_repro_snapshot_bytes(event_counts: dict, estimated_data_json_bytes: int) -> int:
    """Each /update writes study_data + actions + groups JSON (~same order as DB rows)."""
    n_updates = event_counts.get("update", 0)
    if n_updates <= 0:
        return 0
    per_update = int(estimated_data_json_bytes * 1.05)
    return per_update * n_updates


def _estimate_ram_mb(n_study_rows: int, phi_dims: list, n_groups: int) -> dict:
    """Very rough peak RAM during a weekly update (load all study rows + dense linear algebra)."""
    d_max = max(phi_dims) if phi_dims else 64
    base_mb = 280.0
    rows_floats_mb = (n_study_rows * d_max * 8) / (1024 * 1024)
    design_mat_mb = rows_floats_mb * 3
    cov_mb = (d_max * d_max * 8 * 8) / (1024 * 1024)
    peak_mb = base_mb + design_mat_mb + cov_mb + (n_groups * 0.05)
    return {
        "baseline_process_mb_est": round(base_mb, 1),
        "peak_update_mb_est": round(peak_mb, 1),
        "assumptions": "Single worker; EB update holds StudyData in memory and builds numpy arrays; "
        "increase if BACKUP_DATABASE zips large CSVs on the same host.",
    }


def _hosting_guidance(storage_bytes: int, peak_hourly_actions: int) -> dict:
    storage_mb = storage_bytes / (1024 * 1024)
    return {
        "recommended_db_storage_mb": max(256, round(storage_mb * 10)),
        "recommended_app_instances": 1 if peak_hourly_actions < 500 else 2,
        "recommended_cpu_class": "small" if peak_hourly_actions < 500 else "medium",
        "notes": [
            "Traffic is dominated by scheduled bursts rather than continuous high concurrency.",
            "Weekly model updates add matrix work but remain modest at the planned 25-dyad scale.",
        ],
    }
