"""
Persisted data model of the ADAPTS-HCT RL API (API-Spec §5; Monitoring-Spec §9;
implementation contract §2).

All datetimes are stored naive in the study timezone (America/Detroit; contract
D4, ``app.clock``). Host-stamped ``request_timestamp`` columns hold the host's
logical study time; ``created_at``-style columns default to
``app.clock.server_now()``.

The schema is created by ``db.create_all()`` at start-up and migrated in place
by the Alembic chain under ``migrations/versions`` (head
``20261019_01_quarantine_bytes``).
"""

from __future__ import annotations

import csv
import datetime
import base64
import hashlib
import json
import os

from app.clock import server_now
from app.extensions import db


def _now() -> datetime.datetime:
    return server_now()


class Group(db.Model):
    """
    Database table to store groups (dyads) (API-Spec §5.1).

    Warm-up is decided server-side at /action time from the cohort size and
    the dyad's study week (API-Spec §2.2); it is not a per-dyad host flag and
    is not stored here. Every registration is also appended to
    ``group_history`` (Monitoring-Spec §9.6) because a re-registration
    overwrites ``group_info`` here.
    """

    __tablename__ = "groups"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    group_id = db.Column(db.String(255), unique=True, nullable=False)
    # Per-call rid returned by the most recent /register_group for this dyad
    # (updated on idempotent re-registration); API-Spec §5.1. Nullable so
    # rows written before this column existed remain valid.
    rid = db.Column(db.String(255), nullable=True)
    group_info = db.Column(db.JSON, nullable=False)
    created_at = db.Column(db.DateTime, nullable=False)

    def __init__(
        self,
        group_id: str,
        group_info: dict,
        rid: str | None = None,
        created_at: datetime.datetime | None = None,
    ):
        """
        Initialize the Group object.
        """
        if created_at is None:
            created_at = _now()
        self.group_id = group_id
        self.rid = rid
        self.group_info = group_info
        self.created_at = created_at

    def __repr__(self):
        """
        Return a string representation of the Group object.
        """
        return f"<Group group_id={self.group_id} created_at={self.created_at}>"


class GroupHistory(db.Model):
    """
    Append-only history of every successful /register_group call
    (Monitoring-Spec §9.6). Rows are never changed; checks read a dyad's
    consent dates as they were at the time being checked from here, not from
    ``groups`` (which a new registration overwrites).
    """

    __tablename__ = "group_history"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    group_id = db.Column(db.String(255), nullable=False, index=True)
    member_list = db.Column(db.JSON, nullable=False)
    consent_start_date = db.Column(db.String(10), nullable=False)
    consent_end_date = db.Column(db.String(10), nullable=False)
    rid = db.Column(db.String(255), nullable=True)
    recorded_at = db.Column(db.DateTime, nullable=False)

    def __init__(
        self,
        group_id: str,
        member_list: list,
        consent_start_date: str,
        consent_end_date: str,
        rid: str | None = None,
        recorded_at: datetime.datetime | None = None,
    ):
        self.group_id = group_id
        self.member_list = member_list
        self.consent_start_date = str(consent_start_date)
        self.consent_end_date = str(consent_end_date)
        self.rid = rid
        self.recorded_at = recorded_at if recorded_at is not None else _now()

    def __repr__(self):
        return (
            f"<GroupHistory group_id={self.group_id} "
            f"consent={self.consent_start_date}..{self.consent_end_date} rid={self.rid}>"
        )


class DataUpload(db.Model):
    """
    Append-only log of every /upload_data call (API-Spec §5.3).

    Each row is a full flat snapshot of every variable in the field
    dictionary (`data.X` always present, possibly the literal "miss").
    The "current value of field X for dyad Y" is `data.X` from the most
    recent row for Y. /action reads the latest row at decision time; /update
    walks the timeline to derive outcomes and rewards.
    """

    __tablename__ = "data_uploads"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    group_id = db.Column(db.String(255), nullable=False)
    # Per-call rid returned by the /upload_data response that wrote this row
    # (API-Spec §5.3). Nullable so rows written before this column existed
    # remain valid; unique among non-null values.
    rid = db.Column(db.String(255), unique=True, nullable=True)
    data = db.Column(db.JSON, nullable=False)
    request_timestamp = db.Column(db.DateTime, nullable=False)
    created_at = db.Column(db.DateTime, nullable=False)

    def __init__(
        self,
        group_id: str,
        data: dict,
        request_timestamp: datetime.datetime,
        rid: str | None = None,
        created_at: datetime.datetime | None = None,
    ):
        if created_at is None:
            created_at = _now()
        self.group_id = group_id
        self.rid = rid
        self.data = data
        self.request_timestamp = request_timestamp
        self.created_at = created_at

    def __repr__(self):
        return (
            f"<DataUpload group_id={self.group_id} "
            f"request_timestamp={self.request_timestamp}>"
        )


class UploadQuarantine(db.Model):
    """
    Verbatim bodies of rejected (400) /upload_data calls (Monitoring-Spec §7
    F-U1; contract D18). This is data preservation, not an error code: the
    access-controlled store is excluded from the default ``flask export-csv``
    and included only with ``--include-quarantine``.
    """

    __tablename__ = "upload_quarantine"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    rid = db.Column(db.String(255), nullable=False)
    group_id = db.Column(db.String(255), nullable=True)
    # Binary, rather than Text: F-U1 promises byte-for-byte preservation even
    # when a rejected request is not valid UTF-8.
    raw_body = db.Column(db.LargeBinary, nullable=False)
    validation_error = db.Column(db.Text, nullable=False)
    payload_sha256 = db.Column(db.String(64), nullable=False)
    payload_bytes = db.Column(db.Integer, nullable=False)
    received_at = db.Column(db.DateTime, nullable=False)

    def __init__(
        self,
        rid: str,
        raw_body: bytes | str,
        validation_error: str,
        group_id: str | None = None,
        payload_sha256: str | None = None,
        payload_bytes: int | None = None,
        received_at: datetime.datetime | None = None,
    ):
        body_bytes = raw_body.encode("utf-8") if isinstance(raw_body, str) else bytes(raw_body)
        self.rid = rid
        self.group_id = group_id
        self.raw_body = body_bytes
        self.validation_error = validation_error
        self.payload_sha256 = (
            payload_sha256 if payload_sha256 is not None else hashlib.sha256(body_bytes).hexdigest()
        )
        self.payload_bytes = int(payload_bytes) if payload_bytes is not None else len(body_bytes)
        self.received_at = received_at if received_at is not None else _now()

    def __repr__(self):
        return f"<UploadQuarantine rid={self.rid} group_id={self.group_id} bytes={self.payload_bytes}>"


class Action(db.Model):
    """
    One row per minted /action decision (API-Spec §5.2; contract D7, D11-D13).

    ``action_prob`` is Pr(A = 1). ``state`` is the standardized design state
    ``g`` (``NULL`` on scheduled warm-up and when no state could be built);
    ``raw_context`` holds the projected context and ``design_state``
    (``NULL`` for the no-state fallback). ``policy_week`` /
    ``policy_version_id`` / ``model_parameters_id`` identify the policy that
    produced a learned ``action_prob`` and ``mu`` / ``v`` the contrast mean and
    variance it was computed from (Monitoring-Spec §9.5).
    """

    __tablename__ = "actions"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    group_id = db.Column(db.String(255), nullable=False)
    rid = db.Column(db.String(255), unique=True, nullable=False)
    state = db.Column(db.JSON, nullable=True)
    decision_idx = db.Column(db.Integer, nullable=False)
    decision_type = db.Column(db.String(255), nullable=False)
    raw_context = db.Column(db.JSON, nullable=True)
    action = db.Column(db.Integer, nullable=False)
    action_prob = db.Column(db.Float, nullable=False)
    is_warmup = db.Column(db.Boolean, nullable=False, default=False)
    # cohort / week1 / no_state / no_policy / learner_error /
    # invalid_probability (contract D3, D12); NULL on learned decisions.
    warmup_reason = db.Column(db.String(32), nullable=True)
    # Flat model-parameter vector the learner scored to produce action_prob
    # (API-Spec §2.2 / §5.2), stored opaquely. Persisted so the idempotent
    # /action replay can return it verbatim; null on warm-up and for learners
    # that expose no such vector.
    model_param = db.Column(db.JSON, nullable=True)
    random_state = db.Column(db.JSON, nullable=False)
    # Posterior row (model_parameters, snapshot_type="posterior") scored for
    # this decision; NULL when no learner policy was scored (contract D15).
    model_parameters_id = db.Column(
        db.Integer, db.ForeignKey("model_parameters.id"), nullable=True
    )
    policy_week = db.Column(db.Date, nullable=True)
    policy_version_id = db.Column(db.Integer, nullable=True)
    hyperparams_hash = db.Column(db.String(64), nullable=True)
    mu = db.Column(db.Float, nullable=True)
    v = db.Column(db.Float, nullable=True)
    # data_uploads.id of the snapshot the state was built from.
    upload_id = db.Column(db.Integer, nullable=True)
    request_timestamp = db.Column(db.DateTime, nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False)

    __table_args__ = (
        db.UniqueConstraint(
            "group_id",
            "decision_type",
            "decision_idx",
            name="uq_action_group_type_idx",
        ),
    )

    def __init__(
        self,
        group_id: str,
        action: int,
        rid: str,
        state: list | dict | None,
        decision_idx: int,
        decision_type: str,
        raw_context: dict | None,
        action_prob: float,
        random_state: dict,
        model_parameters_id: int | None,
        request_timestamp: datetime.datetime,
        is_warmup: bool = False,
        warmup_reason: str | None = None,
        model_param: list | None = None,
        timestamp: datetime.datetime | None = None,
        *,
        policy_week: datetime.date | None = None,
        policy_version_id: int | None = None,
        hyperparams_hash: str | None = None,
        mu: float | None = None,
        v: float | None = None,
        upload_id: int | None = None,
    ):
        """
        Initialize the Action object.
        """
        if timestamp is None:
            timestamp = _now()
        self.group_id = group_id
        self.action = action
        self.rid = rid
        self.state = state
        self.decision_idx = decision_idx
        self.decision_type = decision_type
        self.raw_context = raw_context
        self.action_prob = action_prob
        self.is_warmup = bool(is_warmup)
        self.warmup_reason = warmup_reason
        self.model_param = model_param
        self.random_state = random_state
        self.model_parameters_id = model_parameters_id
        self.request_timestamp = request_timestamp
        self.timestamp = timestamp
        self.policy_week = policy_week
        self.policy_version_id = policy_version_id
        self.hyperparams_hash = hyperparams_hash
        self.mu = mu
        self.v = v
        self.upload_id = upload_id

    def __repr__(self):
        """
        Return a string representation of the Action object.
        """
        return f"<Action rid={self.rid}, group_id={self.group_id}, action={self.action}, state={self.state}, action_prob={self.action_prob}>"


class ModelParameters(db.Model):
    """
    Unified table for all decision-making-algorithm parameters (API-Spec §5.5;
    contract D15).

    Holds either:
      - A legacy / fixed-probability row (only ``probability_of_action`` set), or
      - A learner snapshot — ``local_fit`` (per dyad), ``hyper`` (pooled, per
        agent) or ``posterior`` (per dyad) — tagged with the study week
        (``study_week_monday``), the ``policy_versions`` row it belongs to, the
        /update rid that wrote it and the hyperparameter fingerprint in force.
        These rows realize Monitoring-Spec §9.5's ``local_fits``,
        ``eb_hyper_updates`` and ``posteriors``.

    Every column except ``id`` and ``timestamp`` is nullable so the same table
    can house both shapes; the writing algorithm decides which fields to set.
    """

    __tablename__ = "model_parameters"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)

    # Legacy / fixed-probability field.
    probability_of_action = db.Column(db.Float, nullable=True)

    # Empirical-Bayes snapshot fields (absorbed from the former
    # ``empirical_bayes_snapshots`` table).
    snapshot_type = db.Column(db.String(64), nullable=True)
    group_id = db.Column(db.String(255), nullable=True)
    decision_type = db.Column(db.String(255), nullable=True)
    agent_decision_index = db.Column(db.Integer, nullable=True)
    sample_size = db.Column(db.Integer, nullable=True, default=0)
    feature_dim = db.Column(db.Integer, nullable=True)
    theta = db.Column(db.JSON, nullable=True)
    covariance = db.Column(db.JSON, nullable=True)
    perturbation = db.Column(db.JSON, nullable=True)
    metadata_json = db.Column(db.JSON, nullable=True)

    # inf_lsvi_eb bookkeeping (contract D15).
    study_week_monday = db.Column(db.Date, nullable=True)
    policy_version_id = db.Column(db.Integer, nullable=True, index=True)
    update_rid = db.Column(db.String(255), nullable=True)
    hyperparams_hash = db.Column(db.String(64), nullable=True)

    timestamp = db.Column(db.DateTime, nullable=False)

    def __init__(
        self,
        probability_of_action: float | None = None,
        snapshot_type: str | None = None,
        group_id: str | None = None,
        decision_type: str | None = None,
        agent_decision_index: int | None = None,
        sample_size: int = 0,
        feature_dim: int | None = None,
        theta: list | None = None,
        covariance: list | None = None,
        perturbation: list | None = None,
        metadata_json: dict | None = None,
        timestamp: datetime.datetime | None = None,
        *,
        study_week_monday: datetime.date | None = None,
        policy_version_id: int | None = None,
        update_rid: str | None = None,
        hyperparams_hash: str | None = None,
    ):
        if timestamp is None:
            timestamp = _now()
        self.probability_of_action = probability_of_action
        self.snapshot_type = snapshot_type
        self.group_id = group_id
        self.decision_type = decision_type
        self.agent_decision_index = agent_decision_index
        self.sample_size = sample_size
        self.feature_dim = feature_dim
        self.theta = theta
        self.covariance = covariance
        self.perturbation = perturbation
        self.metadata_json = metadata_json if metadata_json is not None else {}
        self.timestamp = timestamp
        self.study_week_monday = study_week_monday
        self.policy_version_id = policy_version_id
        self.update_rid = update_rid
        self.hyperparams_hash = hyperparams_hash

    def __repr__(self):
        if self.snapshot_type is not None:
            return (
                f"<ModelParameters snapshot_type={self.snapshot_type} "
                f"decision_type={self.decision_type} group_id={self.group_id} "
                f"agent_decision_index={self.agent_decision_index} "
                f"study_week_monday={self.study_week_monday}>"
            )
        return f"<ModelParameters probability_of_action={self.probability_of_action}>"


class PolicyVersion(db.Model):
    """
    One row per agent per /update run (contract D13, D14; Monitoring-Spec
    §9.5 ``policy_versions``, used by C7 and F-U2).

    ``status`` is ``published`` / ``rejected`` (C3 gate) / ``failed``. A
    published version is eligible for a decision at study time ``t`` in week
    ``W`` iff ``published_at_study <= min(t, Monday(W) 05:45)``, where
    ``published_at_study`` is the update's ``request_timestamp`` plus the run's
    wall-clock duration. ``published_at`` is the server wall time.
    """

    __tablename__ = "policy_versions"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    decision_type = db.Column(db.String(255), nullable=False)
    study_week_monday = db.Column(db.Date, nullable=False)
    update_rid = db.Column(db.String(255), nullable=False)
    status = db.Column(db.String(16), nullable=False)
    reason = db.Column(db.Text, nullable=True)
    published_at = db.Column(db.DateTime, nullable=True)
    published_at_study = db.Column(db.DateTime, nullable=True)
    hyperparams_hash = db.Column(db.String(64), nullable=True)
    n_dyads = db.Column(db.Integer, nullable=False, default=0)
    created_at = db.Column(db.DateTime, nullable=False)

    __table_args__ = (
        db.Index("ix_policy_versions_type_week", "decision_type", "study_week_monday"),
    )

    STATUS_PUBLISHED = "published"
    STATUS_REJECTED = "rejected"
    STATUS_FAILED = "failed"

    def __init__(
        self,
        decision_type: str,
        study_week_monday: datetime.date,
        update_rid: str,
        status: str,
        reason: str | None = None,
        published_at: datetime.datetime | None = None,
        published_at_study: datetime.datetime | None = None,
        hyperparams_hash: str | None = None,
        n_dyads: int = 0,
        created_at: datetime.datetime | None = None,
    ):
        self.decision_type = decision_type
        self.study_week_monday = study_week_monday
        self.update_rid = update_rid
        self.status = status
        self.reason = reason
        self.published_at = published_at
        self.published_at_study = published_at_study
        self.hyperparams_hash = hyperparams_hash
        self.n_dyads = int(n_dyads)
        self.created_at = created_at if created_at is not None else _now()

    def __repr__(self):
        return (
            f"<PolicyVersion id={self.id} {self.decision_type} week={self.study_week_monday} "
            f"status={self.status} rid={self.update_rid}>"
        )


class HyperparameterSet(db.Model):
    """
    Frozen hyperparameter versions (Hyperparameters.md §2.3; Monitoring-Spec
    §9.5, C9; contract D16). Version 1 is written at the first learning
    update; ``flask hyperparams amend`` appends later versions. ``settings``
    is the resolved set including derived values, ``sha256`` its fingerprint.
    """

    __tablename__ = "hyperparameter_sets"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    version = db.Column(db.Integer, unique=True, nullable=False)
    sha256 = db.Column(db.String(64), nullable=False)
    settings = db.Column(db.JSON, nullable=False)
    frozen_at = db.Column(db.DateTime, nullable=False)
    in_force_from = db.Column(db.DateTime, nullable=False)
    reason = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, nullable=False)

    def __init__(
        self,
        version: int,
        sha256: str,
        settings: dict,
        in_force_from: datetime.datetime,
        reason: str | None = None,
        frozen_at: datetime.datetime | None = None,
        created_at: datetime.datetime | None = None,
    ):
        now = _now()
        self.version = int(version)
        self.sha256 = sha256
        self.settings = settings
        self.in_force_from = in_force_from
        self.reason = reason
        self.frozen_at = frozen_at if frozen_at is not None else now
        self.created_at = created_at if created_at is not None else now

    def __repr__(self):
        return f"<HyperparameterSet version={self.version} sha256={self.sha256[:12]}>"


class NoiseInit(db.Model):
    """
    Per-agent noise scale record (Monitoring-Spec §9.5 ``noise_init``, C10).
    σ² is fixed by design (main_new.tex §5.6, eq. (sigma-design)), so this
    records it once, with the bound σ̄² and the resulting τ²₀ values; the
    week-1 statistics columns stay for the record and are NULL when unused.
    """

    __tablename__ = "noise_init"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    decision_type = db.Column(db.String(255), nullable=False)
    sigma_sq = db.Column(db.Float, nullable=False)
    sigma_bar_sq = db.Column(db.Float, nullable=False)
    tau0_act = db.Column(db.Float, nullable=False)
    tau0_mod = db.Column(db.Float, nullable=False)
    source = db.Column(db.String(32), nullable=False, default="design")
    n_week1_rewards = db.Column(db.Integer, nullable=True)
    s2 = db.Column(db.Float, nullable=True)
    clip_used = db.Column(db.Boolean, nullable=False, default=False)
    study_week_monday = db.Column(db.Date, nullable=True)
    created_at = db.Column(db.DateTime, nullable=False)

    def __init__(
        self,
        decision_type: str,
        sigma_sq: float,
        sigma_bar_sq: float,
        tau0_act: float,
        tau0_mod: float,
        source: str = "design",
        n_week1_rewards: int | None = None,
        s2: float | None = None,
        clip_used: bool = False,
        study_week_monday: datetime.date | None = None,
        created_at: datetime.datetime | None = None,
    ):
        self.decision_type = decision_type
        self.sigma_sq = float(sigma_sq)
        self.sigma_bar_sq = float(sigma_bar_sq)
        self.tau0_act = float(tau0_act)
        self.tau0_mod = float(tau0_mod)
        self.source = source
        self.n_week1_rewards = n_week1_rewards
        self.s2 = s2
        self.clip_used = bool(clip_used)
        self.study_week_monday = study_week_monday
        self.created_at = created_at if created_at is not None else _now()


class ThompsonSamplingParams(db.Model):
    """
    Thompson Sampling parameters per (group_id, decision_type).
    Each dyad and decision type has its own independent TS bandit.
    """

    __tablename__ = "thompson_sampling_params"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    group_id = db.Column(db.String(255), nullable=False)
    decision_type = db.Column(db.String(255), nullable=False)
    params = db.Column(db.JSON, nullable=False)  # {action_0: {...}, action_1: {...}}
    updated_at = db.Column(db.DateTime, nullable=False)

    __table_args__ = (db.UniqueConstraint("group_id", "decision_type", name="uq_group_decision"),)

    def __init__(
        self,
        group_id: str,
        decision_type: str,
        params: dict,
        updated_at: datetime.datetime = None,
    ):
        if updated_at is None:
            updated_at = _now()
        self.group_id = group_id
        self.decision_type = decision_type
        self.params = params
        self.updated_at = updated_at


class ModelUpdateRequests(db.Model):
    """
    One row per /update call (API-Spec §5.6).

    ``status`` is ``processing`` / ``completed`` / ``failed``; ``error_message``
    is ``"timeout"`` when F-U4 fires (Monitoring-Spec §7). ``study_week_monday``
    is the week of ``request_timestamp`` (contract D14), ``started_at`` the
    server time the fit began, and ``published_at_study`` the study-logical
    publication time (request timestamp plus run duration; contract D13).
    """

    __tablename__ = "model_update_requests"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    # Per-call rid returned in the 202 response; the key the monitoring
    # algorithm polls (API-Spec §5.6). Renamed from the former `update_id`.
    rid = db.Column(db.String(255), unique=True, nullable=False)
    status = db.Column(db.String(50), nullable=False)
    request_timestamp = db.Column(db.DateTime, nullable=False)
    created_at = db.Column(db.DateTime, nullable=False)
    completed_at = db.Column(db.DateTime, nullable=True)
    error_message = db.Column(db.String(1024), nullable=True)
    study_week_monday = db.Column(db.Date, nullable=True)
    started_at = db.Column(db.DateTime, nullable=True)
    published_at_study = db.Column(db.DateTime, nullable=True)

    def __init__(
        self,
        rid: str,
        request_timestamp: datetime.datetime,
        status: str = "processing",
        created_at: datetime.datetime | None = None,
        *,
        study_week_monday: datetime.date | None = None,
        started_at: datetime.datetime | None = None,
        published_at_study: datetime.datetime | None = None,
    ):
        """
        Initialize the ModelUpdateRequests object.

        The new design has no callback: completion is observed by reading
        `status` / `completed_at` (API-Spec §2.4), not POSTed anywhere.
        """
        if created_at is None:
            created_at = _now()
        self.rid = rid
        self.request_timestamp = request_timestamp
        self.status = status
        self.created_at = created_at
        self.study_week_monday = study_week_monday
        self.started_at = started_at
        self.published_at_study = published_at_study

    def __repr__(self):
        """
        Return a string representation of the ModelUpdateRequests object.
        """
        return f"<ModelUpdateRequests rid={self.rid}, status={self.status}>"


class StudyData(db.Model):
    """
    Update-derived (action, outcome) pairs (API-Spec §5.4).

    One row per (action, derived outcome), written during /update: the action
    is located in `actions`, the outcome fields are read from later
    `data_uploads` rows on the timeline, and the scalar reward is
    computed and stored. Idempotent across /update re-runs — an existing row
    is updated in place if its outcome window has since filled in.

    ``reward`` is the normalized reward of contract D6; the native value is in
    ``outcome["reward_native"]``. ``state`` is NULL when the action had none.
    """

    __tablename__ = "study_data"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    group_id = db.Column(db.String(255), nullable=False)
    decision_idx = db.Column(db.Integer, nullable=False)
    decision_type = db.Column(db.String(255), nullable=False)
    action = db.Column(db.Integer, nullable=False)
    action_prob = db.Column(db.Float, nullable=False)
    state = db.Column(db.JSON, nullable=True)
    raw_context = db.Column(db.JSON, nullable=False)
    outcome = db.Column(db.JSON, nullable=False)
    reward = db.Column(db.Float, nullable=True)
    request_timestamp = db.Column(db.DateTime, nullable=False)
    derived_at = db.Column(db.DateTime, nullable=True)
    created_at = db.Column(db.DateTime, nullable=False)

    __table_args__ = (
        db.UniqueConstraint(
            "group_id",
            "decision_type",
            "decision_idx",
            name="uq_study_group_type_idx",
        ),
    )

    def __init__(
        self,
        group_id: str,
        decision_idx: int,
        decision_type: str,
        action: int,
        action_prob: float,
        state: list | None,
        raw_context: dict,
        outcome: dict,
        reward: float | None,
        request_timestamp: datetime.datetime,
        derived_at: datetime.datetime | None = None,
        created_at: datetime.datetime | None = None,
    ):
        """
        Initialize the StudyData object.
        """
        if created_at is None:
            created_at = _now()
        self.group_id = group_id
        self.decision_idx = decision_idx
        self.decision_type = decision_type
        self.action = action
        self.action_prob = action_prob
        self.state = state
        self.raw_context = raw_context
        self.outcome = outcome
        self.reward = reward
        self.request_timestamp = request_timestamp
        self.derived_at = derived_at
        self.created_at = created_at

    def __repr__(self):
        """
        Return a string representation of the StudyData object.
        """
        return f"<StudyData group_id={self.group_id}, raw_context={self.raw_context}, action={self.action}, reward={self.reward}>"


class StandardizationBaseline(db.Model):
    """
    Per-dyad week-1 baselines used to standardize continuous state variables
    before they enter the legacy learners (API-Spec §5.8). The inf_lsvi_eb
    learner standardizes on fixed design ranges instead (main_new.tex §5.4)
    and does not read this table.

    One row per (group_id, decision_type, variable_name). Populated once per
    dyad/agent at the first weekly /update after the dyad has accumulated
    week-1 data, and never modified thereafter.
    """

    __tablename__ = "standardization_baselines"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    group_id = db.Column(db.String(255), nullable=False)
    decision_type = db.Column(db.String(255), nullable=False)
    variable_name = db.Column(db.String(255), nullable=False)
    mu = db.Column(db.Float, nullable=False)
    sigma = db.Column(db.Float, nullable=False)
    sample_size = db.Column(db.Integer, nullable=False, default=0)
    created_at = db.Column(db.DateTime, nullable=False)

    __table_args__ = (
        db.UniqueConstraint(
            "group_id",
            "decision_type",
            "variable_name",
            name="uq_baseline_group_dt_var",
        ),
    )

    def __init__(
        self,
        group_id: str,
        decision_type: str,
        variable_name: str,
        mu: float,
        sigma: float,
        sample_size: int = 0,
        created_at: datetime.datetime | None = None,
    ):
        if created_at is None:
            created_at = _now()
        self.group_id = group_id
        self.decision_type = decision_type
        self.variable_name = variable_name
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.sample_size = int(sample_size)
        self.created_at = created_at


class UpdateReproducibilitySnapshot(db.Model):
    """
    Points to an on-disk full copy of data_uploads, actions (decision states),
    and groups taken immediately before a model update completes. Under the
    update-derived design study_data is itself produced during /update, so the
    snapshot copies the upstream data_uploads instead (API-Spec §5.9).
    """

    __tablename__ = "update_reproducibility_snapshots"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    # The /update call's rid; matches ModelUpdateRequests.rid (API-Spec §5.9).
    # Renamed from the former `update_id`.
    rid = db.Column(db.String(255), nullable=False)
    model_parameters_id = db.Column(db.Integer, nullable=True)
    snapshot_dir = db.Column(db.String(2048), nullable=False)
    data_uploads_count = db.Column(db.Integer, nullable=False, default=0)
    actions_count = db.Column(db.Integer, nullable=False, default=0)
    groups_count = db.Column(db.Integer, nullable=False, default=0)
    total_bytes = db.Column(db.BigInteger, nullable=False, default=0)
    created_at = db.Column(db.DateTime, nullable=False)

    def __init__(
        self,
        rid: str,
        snapshot_dir: str,
        model_parameters_id: int | None = None,
        data_uploads_count: int = 0,
        actions_count: int = 0,
        groups_count: int = 0,
        total_bytes: int = 0,
        created_at: datetime.datetime | None = None,
    ):
        if created_at is None:
            created_at = _now()
        self.rid = rid
        self.model_parameters_id = model_parameters_id
        self.snapshot_dir = snapshot_dir
        self.data_uploads_count = data_uploads_count
        self.actions_count = actions_count
        self.groups_count = groups_count
        self.total_bytes = total_bytes
        self.created_at = created_at


# ---------------------------------------------------------------- monitoring


class MonitorAlarm(db.Model):
    """
    One row per alarm raised by monitoring (API-Spec §5.10; Monitoring-Spec
    §9.1). Rows are never deleted and never changed except to add a note.
    ``dedup_key`` (unique when set) lets scheduled checks and reruns avoid
    duplicate alarms for the same finding.
    """

    __tablename__ = "monitor_alarms"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    event_id = db.Column(db.String(64), unique=True, nullable=False)
    item_id = db.Column(db.String(64), nullable=False)
    severity = db.Column(db.String(16), nullable=False)
    owner = db.Column(db.String(16), nullable=False)
    raised_at = db.Column(db.DateTime, nullable=False, index=True)
    source = db.Column(db.String(32), nullable=False)
    run_id = db.Column(db.Integer, nullable=True)
    group_id = db.Column(db.String(255), nullable=True)
    decision_type = db.Column(db.String(255), nullable=True)
    decision_idx = db.Column(db.Integer, nullable=True)
    rid = db.Column(db.String(255), nullable=True)
    message = db.Column(db.Text, nullable=False)
    value = db.Column(db.Float, nullable=True)
    threshold = db.Column(db.Float, nullable=True)
    detail = db.Column(db.JSON, nullable=False)
    fallback_used = db.Column(db.String(255), nullable=True)
    note = db.Column(db.Text, nullable=True)
    note_by = db.Column(db.String(255), nullable=True)
    noted_at = db.Column(db.DateTime, nullable=True)
    check_list_version = db.Column(db.String(16), nullable=False)
    dedup_key = db.Column(db.String(255), unique=True, nullable=True)
    created_at = db.Column(db.DateTime, nullable=False)

    def __init__(
        self,
        event_id: str,
        item_id: str,
        severity: str,
        owner: str,
        source: str,
        message: str,
        check_list_version: str,
        raised_at: datetime.datetime | None = None,
        detail: dict | None = None,
        run_id: int | None = None,
        group_id: str | None = None,
        decision_type: str | None = None,
        decision_idx: int | None = None,
        rid: str | None = None,
        value: float | None = None,
        threshold: float | None = None,
        fallback_used: str | None = None,
        note: str | None = None,
        note_by: str | None = None,
        noted_at: datetime.datetime | None = None,
        dedup_key: str | None = None,
        created_at: datetime.datetime | None = None,
    ):
        now = _now()
        self.event_id = event_id
        self.item_id = item_id
        self.severity = severity
        self.owner = owner
        self.source = source
        self.message = message
        self.check_list_version = check_list_version
        self.raised_at = raised_at if raised_at is not None else now
        self.detail = detail if detail is not None else {}
        self.run_id = run_id
        self.group_id = group_id
        self.decision_type = decision_type
        self.decision_idx = decision_idx
        self.rid = rid
        self.value = value
        self.threshold = threshold
        self.fallback_used = fallback_used
        self.note = note
        self.note_by = note_by
        self.noted_at = noted_at
        self.dedup_key = dedup_key
        self.created_at = created_at if created_at is not None else now

    def __repr__(self):
        return f"<MonitorAlarm {self.item_id} {self.severity}/{self.owner} event_id={self.event_id}>"


class ApiErrorEvent(db.Model):
    """
    The error-code log (Monitoring-Spec §6.9, §9.2): one row per non-success
    answer, written only by the after-response hook. Stores identifiers and
    a SHA-256 fingerprint of the request body, never the body itself.
    ``rid`` is nonempty for every API or framework-level answer; it is
    nullable only so a non-envelope answer can still be logged (D3 finding).
    """

    __tablename__ = "api_error_events"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    event_id = db.Column(db.String(64), unique=True, nullable=False)
    code = db.Column(db.String(64), nullable=False)
    endpoint = db.Column(db.String(255), nullable=False)
    http_status = db.Column(db.Integer, nullable=False)
    title = db.Column(db.String(255), nullable=True)
    message = db.Column(db.Text, nullable=True)
    rid = db.Column(db.String(255), nullable=True)
    group_id = db.Column(db.String(255), nullable=True)
    decision_type = db.Column(db.String(255), nullable=True)
    decision_idx = db.Column(db.Integer, nullable=True)
    payload_sha256 = db.Column(db.String(64), nullable=True)
    payload_bytes = db.Column(db.Integer, nullable=True)
    occurred_at = db.Column(db.DateTime, nullable=False, index=True)

    def __init__(
        self,
        event_id: str,
        code: str,
        endpoint: str,
        http_status: int,
        title: str | None = None,
        message: str | None = None,
        rid: str | None = None,
        group_id: str | None = None,
        decision_type: str | None = None,
        decision_idx: int | None = None,
        payload_sha256: str | None = None,
        payload_bytes: int | None = None,
        occurred_at: datetime.datetime | None = None,
    ):
        self.event_id = event_id
        self.code = code
        self.endpoint = endpoint
        self.http_status = int(http_status)
        self.title = title
        self.message = message
        self.rid = rid
        self.group_id = group_id
        self.decision_type = decision_type
        self.decision_idx = decision_idx
        self.payload_sha256 = payload_sha256
        self.payload_bytes = payload_bytes
        self.occurred_at = occurred_at if occurred_at is not None else _now()

    def __repr__(self):
        return f"<ApiErrorEvent {self.code} {self.endpoint} rid={self.rid}>"


class MonitorRun(db.Model):
    """
    One row per monitoring run (Monitoring-Spec §9.3); ``id`` is the
    ``run_id``. ``kind`` is ``regular`` / ``check`` / ``rerun``; ``status``
    ``ok`` / ``error`` / ``skipped``. ``logical_now`` is the study-clock
    ``now`` the run was evaluated at (contract D20).
    """

    __tablename__ = "monitor_runs"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    kind = db.Column(db.String(16), nullable=False)
    detail = db.Column(db.JSON, nullable=True)
    item_id = db.Column(db.String(64), nullable=True)
    window_start = db.Column(db.DateTime, nullable=True)
    window_end = db.Column(db.DateTime, nullable=True)
    started_at = db.Column(db.DateTime, nullable=False)
    finished_at = db.Column(db.DateTime, nullable=True)
    status = db.Column(db.String(16), nullable=False)
    n_alarms = db.Column(db.Integer, nullable=False, default=0)
    error = db.Column(db.Text, nullable=True)
    logical_now = db.Column(db.DateTime, nullable=True)

    def __init__(
        self,
        kind: str,
        status: str,
        started_at: datetime.datetime | None = None,
        finished_at: datetime.datetime | None = None,
        window_start: datetime.datetime | None = None,
        window_end: datetime.datetime | None = None,
        item_id: str | None = None,
        detail: dict | None = None,
        n_alarms: int = 0,
        error: str | None = None,
        logical_now: datetime.datetime | None = None,
    ):
        self.kind = kind
        self.status = status
        self.started_at = started_at if started_at is not None else _now()
        self.finished_at = finished_at
        self.window_start = window_start
        self.window_end = window_end
        self.item_id = item_id
        self.detail = detail
        self.n_alarms = int(n_alarms)
        self.error = error
        self.logical_now = logical_now

    @property
    def run_id(self) -> int | None:
        return self.id

    def __repr__(self):
        return f"<MonitorRun run_id={self.id} kind={self.kind} status={self.status}>"


class UpdateWeek(db.Model):
    """
    One row per weekly update (Monitoring-Spec §9.4): every /update rid the
    host sent for the week, in order, and the week's readiness state
    (``waiting`` / ``running`` / ``ready`` / ``failed`` / ``stuck``).
    """

    __tablename__ = "update_weeks"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    study_week_monday = db.Column(db.Date, unique=True, nullable=False)
    rids = db.Column(db.JSON, nullable=False)
    status = db.Column(db.String(16), nullable=False, default="waiting")
    first_call_at = db.Column(db.DateTime, nullable=True)
    finished_at = db.Column(db.DateTime, nullable=True)
    ready_at = db.Column(db.DateTime, nullable=True)
    missing_closing_uploads = db.Column(db.JSON, nullable=True)
    c7_raised = db.Column(db.Boolean, nullable=False, default=False)

    def __init__(
        self,
        study_week_monday: datetime.date,
        rids: list | None = None,
        status: str = "waiting",
        first_call_at: datetime.datetime | None = None,
        finished_at: datetime.datetime | None = None,
        ready_at: datetime.datetime | None = None,
        missing_closing_uploads: list | None = None,
        c7_raised: bool = False,
    ):
        self.study_week_monday = study_week_monday
        self.rids = list(rids) if rids is not None else []
        self.status = status
        self.first_call_at = first_call_at
        self.finished_at = finished_at
        self.ready_at = ready_at
        self.missing_closing_uploads = missing_closing_uploads
        self.c7_raised = bool(c7_raised)

    def __repr__(self):
        return f"<UpdateWeek {self.study_week_monday} status={self.status} rids={len(self.rids or [])}>"


class MonitorPull(db.Model):
    """
    One row per GET /monitor/alarms call, keyed by its rid (API-Spec §5.11).
    A gap in this table is the API-side view of a missed host pull (E6).
    """

    __tablename__ = "monitor_pulls"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    rid = db.Column(db.String(255), unique=True, nullable=False)
    window_start = db.Column(db.DateTime, nullable=True)
    returned_count = db.Column(db.Integer, nullable=True)
    status = db.Column(db.Integer, nullable=False)
    created_at = db.Column(db.DateTime, nullable=False)

    def __init__(
        self,
        rid: str,
        status: int,
        window_start: datetime.datetime | None = None,
        returned_count: int | None = None,
        created_at: datetime.datetime | None = None,
    ):
        self.rid = rid
        self.status = int(status)
        self.window_start = window_start
        self.returned_count = returned_count
        self.created_at = created_at if created_at is not None else _now()

    def __repr__(self):
        return f"<MonitorPull rid={self.rid} status={self.status} returned={self.returned_count}>"


# -------------------------------------------------------------------- export

#: Models excluded from default exports and backups (contract D18): rejected
#: upload bodies may hold unvalidated participant data.
QUARANTINED_MODELS = (UploadQuarantine,)

MANIFEST_FILENAME = "MANIFEST.json"


def export_models(include_quarantine: bool = False) -> list[type]:
    """
    Every mapped model, sorted by table name (contract §2: ``export-csv`` and
    ``backup_tables`` iterate over every model). Models defined outside this
    module (e.g. under ``app.monitoring``) are included once imported.
    ``upload_quarantine`` is included only with ``include_quarantine``.
    """
    classes = {mapper.class_ for mapper in db.Model.registry.mappers}
    if not include_quarantine:
        classes -= set(QUARANTINED_MODELS)
    return sorted(classes, key=lambda cls: cls.__tablename__)


def _csv_cell(value):
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    if isinstance(value, (datetime.datetime, datetime.date)):
        return value.isoformat()
    if isinstance(value, (bytes, bytearray, memoryview)):
        # CSV has no binary cell type. Keep quarantine exports lossless and
        # self-describing instead of relying on a locale-dependent decode.
        return "base64:" + base64.b64encode(bytes(value)).decode("ascii")
    return value


def dump_tables_to_csv(directory: str, *, include_quarantine: bool = False) -> dict[str, dict]:
    """
    Write one ``<table>.csv`` per model into ``directory`` (created if needed)
    and return the manifest ``{filename: {"sha256": ..., "rows": n}}``. JSON
    columns are written as compact sorted-key JSON, datetimes as ISO-8601,
    NULL as the empty string. Rows are ordered by primary key so repeated
    exports of the same data are byte-identical. Requires an app context.
    """
    os.makedirs(directory, exist_ok=True)
    manifest: dict[str, dict] = {}
    for model in export_models(include_quarantine=include_quarantine):
        table = model.__table__
        filename = f"{table.name}.csv"
        path = os.path.join(directory, filename)
        columns = [col.key for col in table.columns]
        query = db.session.query(model)
        pk = list(table.primary_key.columns)
        if pk:
            query = query.order_by(*pk)
        rows = query.all()
        with open(path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(columns)
            for row in rows:
                writer.writerow([_csv_cell(getattr(row, col)) for col in columns])
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        manifest[filename] = {"sha256": digest.hexdigest(), "rows": len(rows)}
    return manifest
