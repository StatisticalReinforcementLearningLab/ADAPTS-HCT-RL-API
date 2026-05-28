#!/usr/bin/env python
"""
Run the ADAPTS-HCT simulation against a live RL API server.

Usage:
    python tests/run_simulation.py [--base-url URL] [--weeks N] [--dyads N]

Example:
    # Start the server in another terminal (use port 5001 on macOS - port 5000
    # is often used by AirPlay and returns 403):
    #   flask run --port 5001
    python tests/run_simulation.py --base-url http://127.0.0.1:5001 --weeks 2 --dyads 3

    # With verbose logging (each API call; for action: action, action_prob, context):
    python tests/run_simulation.py -v --weeks 1 --dyads 2

    # Re-run without DB conflicts (use unique groups per run):
    python tests/run_simulation.py --fresh --weeks 14 --dyads 2
"""

import argparse
import datetime
import sys
from pathlib import Path

import requests

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.simulate_adapts_hct import ProtocolTrialSimulator, NUM_DYADS


def run_against_server(
    base_url: str = "http://127.0.0.1:5001",
    num_weeks: int = 2,
    num_dyads: int = 5,
    base_date: datetime.date | None = None,
    verbose: bool = False,
    group_prefix: str = "",
):
    """Run simulation against a live API server."""
    if base_date is None:
        base_date = datetime.date(2025, 1, 5)

    simulator = ProtocolTrialSimulator(
        base_date=base_date,
        num_weeks=num_weeks,
        num_dyads=num_dyads,
        seed=42,
        group_prefix=group_prefix,
    )
    results = {"add_group": 0, "action": 0, "upload_data": 0, "update": 0, "errors": []}
    url = f"{base_url}/api/v1"

    def submit_uploads(payloads: list[dict]):
        for upload_payload in payloads:
            try:
                response = requests.post(f"{url}/upload_data", json=upload_payload, timeout=10)
            except requests.RequestException as exc:
                results["errors"].append({"type": "upload_data", "error": str(exc)})
                continue
            if response.status_code in (200, 201):
                results["upload_data"] += 1
            else:
                results["errors"].append(
                    {"type": "upload_data", "status": response.status_code, "body": response.text[:200]}
                )

    for event in simulator.iter_schedule_events():
        submit_uploads(simulator.pop_due_uploads(event["timestamp"]))
        event_type = event["type"]

        try:
            if event_type == "add_group":
                payload = event["payload"]
                if verbose:
                    print(f"[API] add_group  group_id={payload.get('group_id')}")
                response = requests.post(f"{url}/add_group", json=payload, timeout=10)
                if response.status_code in (200, 201):
                    results["add_group"] += 1
                else:
                    results["errors"].append(
                        {"type": event_type, "status": response.status_code, "body": response.text[:200]}
                    )
                continue

            if event_type == "update":
                payload = event["payload"]
                if verbose:
                    print(f"[API] update  timestamp={payload.get('timestamp')}")
                response = requests.post(f"{url}/update", json=payload, timeout=30)
                if response.status_code in (200, 202):
                    results["update"] += 1
                else:
                    results["errors"].append(
                        {"type": event_type, "status": response.status_code, "body": response.text[:200]}
                    )
                continue

            payload = simulator.build_action_payload(event)
            response = requests.post(f"{url}/action", json=payload, timeout=10)
            if response.status_code in (200, 201):
                results["action"] += 1
                response_json = response.json()
                simulator.schedule_upload(payload, response_json)
                if verbose:
                    print(
                        f"[API] action  group_id={payload.get('group_id')} "
                        f"decision_type={payload.get('decision_type')} "
                        f"decision_idx={payload.get('decision_idx')}  "
                        f"action={response_json.get('action')}  action_prob={response_json.get('action_prob')}"
                    )
            else:
                results["errors"].append(
                    {"type": event_type, "status": response.status_code, "body": response.text[:200]}
                )
        except requests.RequestException as e:
            results["errors"].append({"type": event_type, "error": str(e)})
            if verbose:
                print(f"[API] {event_type}  ERROR: {e}")

    submit_uploads(simulator.flush_all_uploads())

    return results


def main():
    parser = argparse.ArgumentParser(description="Run ADAPTS-HCT simulation against RL API")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:5001",
        help="API base URL (use 5001 on macOS; port 5000 is often used by AirPlay)",
    )
    parser.add_argument("--weeks", type=int, default=2, help="Number of weeks to simulate")
    parser.add_argument("--dyads", type=int, default=5, help="Number of dyads")
    parser.add_argument("-v", "--verbose", action="store_true", help="Log each API call; for action, print action, action_prob, context")
    parser.add_argument(
        "--prefix",
        default="",
        help="Prefix for group IDs (e.g. 'run1_') to avoid conflicts when re-running against same DB",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Use a timestamp prefix so each run uses unique groups (avoids 'already exists' errors)",
    )
    args = parser.parse_args()

    prefix = args.prefix
    if args.fresh:
        prefix = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_"

    print(f"Running simulation: {args.weeks} weeks, {args.dyads} dyads")
    if prefix:
        print(f"Group prefix: {prefix!r}")
    print(f"API: {args.base_url}")
    print("-" * 40)

    # Pre-flight check: 403 often means port 5000 hit AirPlay on macOS
    try:
        r = requests.post(
            f"{args.base_url}/api/v1/add_group",
            json={"group_id": "", "member_list": [], "consent_start_date": "", "consent_end_date": ""},
            timeout=5,
        )
        if r.status_code == 403:
            print(
                "\n403 Forbidden: On macOS, port 5000 is often used by AirPlay. "
                "Run: flask run --port 5001\n"
                "Then: python tests/run_simulation.py --base-url http://127.0.0.1:5001\n"
            )
            sys.exit(1)
    except requests.RequestException as e:
        print(f"\nCannot reach server at {args.base_url}: {e}")
        print("Ensure the server is running: flask run --port 5001\n")
        sys.exit(1)

    results = run_against_server(
        base_url=args.base_url,
        num_weeks=args.weeks,
        num_dyads=args.dyads,
        verbose=args.verbose,
        group_prefix=prefix,
    )

    print(f"add_group: {results['add_group']}")
    print(f"action:    {results['action']}")
    print(f"upload:    {results['upload_data']}")
    print(f"update:    {results['update']}")
    if results["errors"]:
        print(f"Errors:    {len(results['errors'])}")
        for e in results["errors"][:5]:
            print(f"  - {e}")
    else:
        print("Errors:    0")
    print("-" * 40)
    print("Done." if not results["errors"] else "Completed with errors.")


if __name__ == "__main__":
    main()
