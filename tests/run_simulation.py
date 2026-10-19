#!/usr/bin/env python
"""
Run the ADAPTS-HCT host simulator (``tests/simulate_adapts_hct.py``) and print a summary.

Two modes:

- in-process (default): a fresh ``config.TestingConfig`` app (in-memory
  SQLite, host clock, ``/update`` inline) driven through the Flask test client;
- ``--base-url URL``: an already running API over HTTP. The server should run
  with ``CLOCK_MODE=host`` (e.g. ``config.SimulationConfig``) because the
  simulator compresses calendar time. Against a threaded ``/update`` the
  simulator does not wait for fits to finish; use ``tools/live_trial.py`` for
  a hosted run with monitoring and waits.

Examples::

    python tests/run_simulation.py --dyads 6 --weeks 4
    python tests/run_simulation.py --base-url http://127.0.0.1:5001 --dyads 3 --weeks 2 --prefix run1_
    python tests/run_simulation.py --offline --dyads 25 --weeks 30   # no server: host side only
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.simulate_adapts_hct import (  # noqa: E402
    FlaskClientTransport,
    HostSimulator,
    HttpTransport,
    OfflineTransport,
    SimulationResult,
    TrialPlan,
)


def run(args: argparse.Namespace) -> SimulationResult:
    plan = TrialPlan(
        num_dyads=args.dyads,
        num_weeks=args.weeks,
        start_date=args.start_date,
        seed=args.seed,
        group_prefix=args.prefix,
        strict=args.strict,
        analytics_gap_rate=args.analytics_gap_rate,
    )
    log = print if args.verbose else None
    if args.offline:
        return HostSimulator(OfflineTransport(seed=args.seed), plan, log=log).run()
    if args.base_url:
        return HostSimulator(HttpTransport(args.base_url), plan, log=log).run()

    from app import create_app, db

    app = create_app("config.TestingConfig")
    with app.app_context():
        try:
            return HostSimulator(FlaskClientTransport(app.test_client()), plan, log=log).run()
        finally:
            db.session.remove()


def print_summary(result: SimulationResult) -> None:
    summary = result.summary()
    plan = summary["plan"]
    print(f"Dyads {plan['num_dyads']}, calendar weeks {plan['num_weeks']} from {plan['start_monday']}, seed {plan['seed']}")
    print("-" * 60)
    print(f"Requests: {summary['n_requests']}")
    for endpoint, counts in summary["requests"].items():
        print(f"  {endpoint:16s} {counts}")
    decisions = summary["decisions"]
    print(f"Decisions scheduled by the calendar: {decisions['expected_by_calendar']}  by source: {decisions['by_source']}")
    for agent, weeks in decisions["by_agent_week"].items():
        row = "  ".join(f"w{w}:{c.get('learned', 0)}/{c.get('n', 0)}" for w, c in weeks.items())
        print(f"  {agent:12s} learned/total per week  {row}")
    for agent, weeks in summary["pi_learned"].items():
        if weeks:
            row = "  ".join(f"w{w}:{s['mean']:.3f}" for w, s in weeks.items() if s.get("n"))
            print(f"  {agent:12s} mean learned pi          {row}")
    print(f"Alarms pulled: {summary['alarms_pulled']['n']} {summary['alarms_pulled']['by_item']}")
    print(f"Unexpected statuses: {len(summary['unexpected'])}  host-side violations: {len(summary['violations'])}")
    for record in summary["unexpected"][:5]:
        print(f"  - {record['logical_time']} {record['endpoint']} -> {record['status']} {record.get('message')}")
    for violation in summary["violations"][:5]:
        print(f"  - {violation['logical_time']} {violation['what']}: {violation['issues'][:2]}")
    if decisions["calendar_mismatches"]:
        print(f"Calendar mismatches: {decisions['calendar_mismatches'][:5]}")
    print("-" * 60)
    ok = not summary["unexpected"] and not summary["violations"] and not decisions["calendar_mismatches"]
    print("Done." if ok else "Completed with errors.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the ADAPTS-HCT host simulator")
    parser.add_argument("--base-url", default=None, help="API base URL; default: in-process TestingConfig app")
    parser.add_argument("--offline", action="store_true", help="no server (OfflineTransport)")
    parser.add_argument("--weeks", type=int, default=4, help="calendar weeks to simulate")
    parser.add_argument("--dyads", type=int, default=6, help="number of dyads (one enrolls each week)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start-date", type=dt.date.fromisoformat, default=dt.date(2026, 11, 2))
    parser.add_argument("--prefix", default="", help="group_id prefix (re-runs against the same database)")
    parser.add_argument("--fresh", action="store_true", help="use a timestamp prefix so each run has new groups")
    parser.add_argument("--strict", action="store_true", help="stop at the first unexpected status or violation")
    parser.add_argument("--analytics-gap-rate", type=float, default=0.02)
    parser.add_argument("--json", type=Path, default=None, help="also write the summary JSON here")
    parser.add_argument("-v", "--verbose", action="store_true", help="print every call")
    args = parser.parse_args(argv)
    if args.fresh:
        args.prefix = dt.datetime.now().strftime("%Y%m%d_%H%M%S_")

    result = run(args)
    print_summary(result)
    if args.json:
        args.json.write_text(json.dumps(result.summary(), indent=2, default=str))
    ok = not result.unexpected and not result.violations and not result.calendar_mismatches()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
