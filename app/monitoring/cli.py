"""
``flask monitor`` commands (Monitoring-Spec §8.3; RL team only, there are no
admin web pages):

    flask monitor loop
    flask monitor run [--item ID] [--since T --until T]
    flask monitor status
    flask monitor show <event_id>
    flask monitor note <event_id> "..." [--by NAME]
    flask monitor report --since D [--until D] [--format text|json]
"""

from __future__ import annotations

import datetime
import getpass
import json

import click
from flask.cli import AppGroup

monitor_cli = AppGroup("monitor", help="On-server monitoring (Monitoring-Spec §8.3).")


def _parse_time(value: str | None, *, end_of_day: bool = False) -> datetime.datetime | None:
    """An ISO date or timestamp on the study clock; a bare date at the end of that day when ``end_of_day``."""
    from app.clock import to_study_local

    if value is None:
        return None
    try:
        dt = to_study_local(value)
    except ValueError as exc:
        raise click.BadParameter(str(exc)) from exc
    if end_of_day and len(value.strip()) == 10:
        dt = datetime.datetime.combine(dt.date(), datetime.time.max)
    return dt


def _dump(obj) -> str:
    from app.monitoring.findings import jsonable

    return json.dumps(jsonable(obj), indent=2, sort_keys=True)


@monitor_cli.command("loop")
@click.option("--max-runs", type=int, default=None, help="Stop after this many runs (testing).")
def loop_cmd(max_runs):
    """Run the monitoring program: one run every MONITOR_RUN_EVERY_SECONDS."""
    from app.monitoring.runner import loop

    n = loop(max_runs=max_runs)
    click.echo(f"Monitoring loop stopped after {n} runs.")


@monitor_cli.command("run")
@click.option("--item", default=None, help="Run one check only (e.g. C7, A4, codes).")
@click.option("--since", default=None, help="Rerun every check over the period starting here (ISO date/time).")
@click.option("--until", default=None, help="End of the period (default: now).")
def run_cmd(item, since, until):
    """Run the checks once, optionally over a past period."""
    from app.monitoring.runner import run_monitoring

    if until is not None and since is None:
        raise click.UsageError("--until needs --since.")
    try:
        result = run_monitoring(now=_parse_time(until, end_of_day=True), since=_parse_time(since), item=item)
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc
    click.echo(
        f"Run #{result['run_id']} ({result['kind']}) over ({result['window_start']}, {result['window_end']}]: "
        f"{len(result['findings'])} findings, {result['n_alarms']} new alarms, {len(result['errors'])} errors."
    )
    for alarm in result["alarms"]:
        click.echo(f"  new {alarm['severity']} {alarm['item_id']} {alarm['event_id']} ({alarm['dedup_key']})")
    for error in result["errors"]:
        click.echo(f"  E7: {error['item']} at {error['at']}: {error['error']}", err=True)


@monitor_cli.command("status")
@click.option("--json", "as_json", is_flag=True, help="Print JSON.")
def status_cmd(as_json):
    """This week's update, alarms of the last 24 h, last run, last host pull."""
    from app.monitoring.report import build_status, format_status
    from app.monitoring.timeutil import monitoring_now

    status = build_status(monitoring_now())
    click.echo(_dump(status) if as_json else format_status(status))


@monitor_cli.command("show")
@click.argument("event_id")
def show_cmd(event_id):
    """One alarm with all its details."""
    from app.models import MonitorAlarm
    from app.monitoring.alarms import alarm_full_record

    row = MonitorAlarm.query.filter_by(event_id=event_id).first()
    if row is None:
        raise click.ClickException(f"no alarm with event_id {event_id}")
    click.echo(_dump(alarm_full_record(row)))


@monitor_cli.command("note")
@click.argument("event_id")
@click.argument("text")
@click.option("--by", default=None, help="Who did it (default: the login name).")
def note_cmd(event_id, text, by):
    """Add a note on what was done (expected for every red alarm)."""
    from app.monitoring.alarms import add_note

    if by is None:
        try:
            by = getpass.getuser()
        except Exception:  # noqa: BLE001
            by = "unknown"
    try:
        row = add_note(event_id, text, by)
    except (LookupError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"Note added to {row.item_id} {row.event_id} by {row.note_by}.")


@monitor_cli.command("report")
@click.option("--since", required=True, help="Start (ISO date or timestamp).")
@click.option("--until", default=None, help="End (ISO date or timestamp; default now).")
@click.option("--format", "fmt", type=click.Choice(["text", "json"]), default="text", show_default=True)
def report_cmd(since, until, fmt):
    """Tables for the paper: per severity, per fallback, per error code."""
    from app.monitoring.report import build_report, format_report
    from app.monitoring.timeutil import monitoring_now

    start = _parse_time(since)
    end = _parse_time(until, end_of_day=True) if until else monitoring_now()
    report = build_report(start, end)
    click.echo(_dump(report) if fmt == "json" else format_report(report))


__all__ = ["monitor_cli"]
