"""
Mapping of an API answer to its error code (Monitoring-Spec §6; implementation
contract D17, D19).

``CODE = <endpoint>-<HTTP status or kind>-<short name>`` (§6.1). The code of an
answer depends only on its endpoint, ``status``, ``title`` and which
``message`` it is (§6.1 last rule). :func:`classify` returns:

- ``None`` for a documented success answer (nothing is logged);
- the §6 code for a documented answer that must be logged: every failure,
  the ``/action`` replay ``AC-200-REPLAY`` and the ``/action`` fallbacks
  ``AC-201-NOSTATE`` / ``AC-201-FALLBACK``;
- ``XX-<status>-UNMAPPED`` for an answer that is not in §6 (contract D17; D3
  reports these weekly).

The message patterns are the §6.2-§6.7 wordings. A few answers the routes
give are not worded in §6 yet; they are mapped to the code of their row (the
host action is the same) but flagged ``documented=False`` so that D3 can list
them until §6 is updated (§6.9 rule 4):

- ``RG-400-PARAM``: ``group_id must be a string.``, ``member_list must be a list.``;
- ``AC/UD/UP-400-PARAM``: ``timestamp must be an ISO-8601 timestamp; got ...``
  (and ``timestamp must be a string or datetime object.`` for UD and UP);
- ``AC-500-PERSIST``: ``The decision could not be persisted.`` (the wording of
  the API-Spec §2 example).
"""

from __future__ import annotations

import dataclasses
import re

from app.monitoring.catalog import unmapped_code

API_PREFIX = "/api/v1/"

#: Path (relative to ``/api/v1/``) -> endpoint prefix of §6.1.
ENDPOINT_PREFIXES = {
    "register_group": "RG",
    "add_group": "RG",
    "action": "AC",
    "upload_data": "UD",
    "update": "UP",
    "monitor/alarms": "MA",
    "monitor/events": "MA",
}

SUCCESS = "success"
CODED = "code"
UNMAPPED = "unmapped"


@dataclasses.dataclass(frozen=True)
class Classified:
    """The classification of one answer."""

    code: str | None
    kind: str             # "success" | "code" | "unmapped"
    prefix: str           # RG / AC / UD / UP / MA / XX
    documented: bool = True


@dataclasses.dataclass(frozen=True)
class _Rule:
    code: str | None       # None: a success answer (not logged)
    status: int
    title: str
    patterns: tuple[re.Pattern, ...]
    documented: bool = True


def _rule(code, status, title, *messages, documented=True) -> _Rule:
    return _Rule(code, status, title, tuple(re.compile(m, re.DOTALL) for m in messages), documented)


def _lit(text: str) -> str:
    return re.escape(text)


_TIMESTAMP_ISO = r"timestamp must be an ISO-8601 timestamp; got .*\."
_TIMESTAMP_TYPE = _lit("timestamp must be a string or datetime object.")

# §6.4.1 endings of type-error messages (``<field>: <ending>``).
_TYPE_ENDINGS = (
    _lit("must be 'am' or 'pm'."),
    _lit("must be a positive integer."),
    _lit("must be a non-negative integer."),
    _lit("must be a non-negative number."),
    _lit("must be a number between 0 and 1."),
    _lit("must be 0 or 1."),
    _lit("must be 0, 1, or 'miss'."),
    _lit("must be numeric or 'miss'."),
    _lit("must be an integer from 1 through 5, or 'miss'."),
    _lit("must be a number from 1 through 5, or 'miss'."),
    _lit("must be one of 1, 2, 3, or 4."),
    _lit("must be one of 1, 2, 3, 4, or 'miss'."),
    _lit("must be a number between 0 and 1, or 'miss'."),
    _lit("must be a dictionary with diary items."),
    r"missing diary item '.*'\.",
    r"'.*' must be numeric or 'miss'\.",
    _lit("must be true or false."),
)

_RULES: dict[str, tuple[_Rule, ...]] = {
    # §6.2 POST /api/v1/register_group
    "RG": (
        _rule(None, 201, "Success", _lit("Group registered successfully."), _lit("Group consent window updated.")),
        _rule(
            "RG-400-PARAM", 400, "Invalid Parameter",
            _lit("Request body is required."),
            r"(group_id|member_list|consent_start_date|consent_end_date) is required\.",
            r"(consent_start_date|consent_end_date) must be a date in YYYY-MM-DD format; got .*\.",
        ),
        _rule(
            "RG-400-PARAM", 400, "Invalid Parameter",
            _lit("group_id must be a string."), _lit("member_list must be a list."),
            documented=False,
        ),
        _rule(
            "RG-409-MEMBER", 409, "Member Mismatch",
            r"group_id .* is already registered with members \[.*\]; membership cannot be changed via "
            r"/register_group\.",
        ),
        _rule("RG-500-INTERNAL", 500, "Internal Error", _lit("Internal server error.")),
        _rule("RG-503-DB", 503, "Service Unavailable", _lit("Database write failed; retry with backoff.")),
    ),
    # §6.3 POST /api/v1/action
    "AC": (
        _rule(None, 201, "Success", _lit("Action requested successfully.")),
        _rule("AC-201-NOSTATE", 201, "Success", _lit("Action requested successfully using the no-state fallback.")),
        _rule("AC-201-FALLBACK", 201, "Success", _lit("Action requested successfully using a fallback.")),
        _rule("AC-200-REPLAY", 200, "Duplicate Decision", r"Returning the existing action for \(.*\)\."),
        _rule(
            "AC-400-PARAM", 400, "Invalid Parameter",
            _lit("Request body is required."),
            _lit("group_id is required."), _lit("group_id must be a string."),
            _lit("timestamp is required."), _TIMESTAMP_TYPE,
            _lit("decision_idx is required."), _lit("decision_idx must be an integer."),
            _lit("decision_type is required."), _lit("decision_type must be a string."),
            _lit("Invalid decision_type. Must be 'aya_message', 'cp_message', or 'dyad_game'."),
        ),
        _rule("AC-400-PARAM", 400, "Invalid Parameter", _TIMESTAMP_ISO, documented=False),
        _rule("AC-404-GROUP", 404, "Unknown Group", r"group_id .* is not registered\."),
        _rule("AC-500-PERSIST", 500, "Internal Error", _lit("Failed to persist the action.")),
        _rule("AC-500-PERSIST", 500, "Internal Error", _lit("The decision could not be persisted."), documented=False),
        _rule("AC-503-STORE", 503, "Service Unavailable",
              _lit("The decision store is unavailable; use the host fallback.")),
    ),
    # §6.4 POST /api/v1/upload_data
    "UD": (
        _rule(None, 201, "Success", _lit("Snapshot accepted.")),
        _rule(
            "UD-400-PARAM", 400, "Invalid Parameter",
            _lit("group_id is required."), _lit("group_id must be a string."), _lit("timestamp is required."),
            _lit("data is required."), _lit("data must be a dictionary."),
        ),
        _rule("UD-400-PARAM", 400, "Invalid Parameter", _TIMESTAMP_TYPE, _TIMESTAMP_ISO, documented=False),
        _rule("UD-400-UNKNOWN", 400, "Invalid Parameter", r"unknown field '.*'\."),
        _rule("UD-400-MISSINGFIELD", 400, "Invalid Parameter", r"\S+ is required \(full snapshot\)\."),
        _rule("UD-400-TYPE", 400, "Invalid Parameter", r"\S+: (" + "|".join(_TYPE_ENDINGS) + r")"),
        _rule("UD-400-RANGE", 400, "Invalid Parameter", r"\S+ must be between \S+ and \S+; got .*\."),
        _rule("UD-400-INCONSISTENT", 400, "Invalid Parameter", r"\S+ does not match \S+: expected .*, got .*\."),
        _rule("UD-404-GROUP", 404, "Unknown Group", r"group_id .* is not registered\."),
        _rule("UD-500-INTERNAL", 500, "Internal Error", _lit("Internal server error.")),
        _rule("UD-503-DB", 503, "Service Unavailable", _lit("Database write failed; retry.")),
    ),
    # §6.5 POST /api/v1/update
    "UP": (
        _rule(None, 202, "Update Accepted", _lit("Model update started.")),
        _rule("UP-400-PARAM", 400, "Invalid Parameter", _lit("timestamp is required.")),
        _rule("UP-400-PARAM", 400, "Invalid Parameter", _TIMESTAMP_TYPE, _TIMESTAMP_ISO, documented=False),
        _rule("UP-500-INTERNAL", 500, "Internal Error", _lit("Internal server error.")),
        _rule("UP-503-ENQUEUE", 503, "Service Unavailable", _lit("Could not start the update; retry.")),
    ),
    # §6.6 GET /api/v1/monitor/alarms (and the deprecated /monitor/events)
    "MA": (
        _rule(None, 200, "Success", r"Returning \d+ alarms\."),
        _rule("MA-500-INTERNAL", 500, "Internal Error", _lit("Internal server error.")),
        _rule("MA-503-STORE", 503, "Service Unavailable", _lit("Monitoring store unavailable; retry with backoff.")),
    ),
}

# §6.7 framework-level answers (any path).
_FRAMEWORK = (
    _rule("XX-404-ROUTE", 404, "Not Found", _lit("Unknown route.")),
    _rule("XX-405-METHOD", 405, "Method Not Allowed", _lit("Method not allowed.")),
    _rule("XX-500-UNCAUGHT", 500, "Internal Error", _lit("Internal server error.")),
)


def endpoint_of(path: str) -> str:
    """``/api/v1/upload_data/`` -> ``upload_data`` (``""`` outside the API prefix)."""
    if not isinstance(path, str) or not path.startswith(API_PREFIX):
        return ""
    return path[len(API_PREFIX):].strip("/")


def prefix_of(path: str) -> str:
    """The §6.1 endpoint prefix of a request path (``XX`` when not tied to one endpoint)."""
    return ENDPOINT_PREFIXES.get(endpoint_of(path), "XX")


def _matches(rule: _Rule, status: int, title, message) -> bool:
    if rule.status != status or rule.title != title or not isinstance(message, str):
        return False
    return any(p.fullmatch(message) for p in rule.patterns)


def classify(path: str, status: int, title, message) -> Classified:
    """
    The §6 classification of an answer to ``path`` (contract D17). Endpoint
    rows first (§6.2-§6.6), then the framework rows (§6.7), else
    ``XX-<status>-UNMAPPED``.
    """
    prefix = prefix_of(path)
    status = int(status)
    for rule in _RULES.get(prefix, ()):
        if _matches(rule, status, title, message):
            kind = SUCCESS if rule.code is None else CODED
            return Classified(rule.code, kind, prefix, rule.documented)
    for rule in _FRAMEWORK:
        if _matches(rule, status, title, message):
            return Classified(rule.code, CODED, "XX", True)
    return Classified(unmapped_code(status), UNMAPPED, prefix)


def must_log(classified: Classified) -> bool:
    """§6: every failure, ``AC-200-REPLAY``, the ``AC-201-*`` fallbacks and every unmapped answer are logged."""
    return classified.code is not None


__all__ = [
    "API_PREFIX",
    "ENDPOINT_PREFIXES",
    "Classified",
    "classify",
    "endpoint_of",
    "prefix_of",
    "must_log",
    "SUCCESS",
    "CODED",
    "UNMAPPED",
]
