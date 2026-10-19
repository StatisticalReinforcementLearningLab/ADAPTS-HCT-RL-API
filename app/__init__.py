import subprocess
import logging
import os
import pickle
import sys

from flask import Flask, request
from sqlalchemy import event, text
from werkzeug.exceptions import HTTPException

from app.extensions import db, migrate
from app.logging_config import setup_logging
from app.algorithms.flat_prob import FlatProbRLAlgorithm
from app.algorithms.empirical_bayes import ThreeAgentEmpiricalBayesAlgorithm
from app.algorithms.eb_gradient import ThreeAgentEmpiricalBayesGradientAlgorithm
from app.algorithms.inf_lsvi_eb import InfLsviEbAlgorithm
from app.algorithms.inf_lsvi_local import ThreeAgentInfLsviAlgorithm
from app.algorithms.inf_lsvi_pool import ThreeAgentInfLsviPooledAlgorithm
from app.algorithms.hybrid_rel_pool import HybridRelPoolAlgorithm
from app.algorithms.thompson_sampling import ThompsonSamplingAlgorithm
from app.algorithms.random_baseline import RandomBaselineAlgorithm
from app.algorithms.always_send import AlwaysSendAlgorithm
from app.algorithms.always_none import AlwaysNoneAlgorithm
from app.deterministic_sampler import DeterministicSampleStream
from app.hyperparams import (
    HyperparameterError,
    check_startup_against_frozen,
    default_hyperparams,
    latest_frozen,
    load_hyperparams,
    register_cli as register_hyperparams_cli,
)
from app.models import Action, ModelParameters

DEFAULT_ALGORITHM = "inf_lsvi_eb"

# Learners that draw from the deterministic sample buffer; their sampler
# cursor is restored from the actions table on start-up (API-Spec §6).
SAMPLER_ALGORITHMS = {
    "inf_lsvi_eb": InfLsviEbAlgorithm,
    "empirical_bayes": ThreeAgentEmpiricalBayesAlgorithm,
    "eb_gradient": ThreeAgentEmpiricalBayesGradientAlgorithm,
    "inf_lsvi": ThreeAgentInfLsviAlgorithm,
    "inf_lsvi_pool": ThreeAgentInfLsviPooledAlgorithm,
    "hybrid_rel_pool": HybridRelPoolAlgorithm,
}


def create_app(config_class="config.Config"):
    """
    Factory function to create and configure the Flask app.

    Start-up refuses (raises) when the hyperparameter file is invalid
    (Hyperparameters.md §2.1), when it differs from the frozen set
    (Hyperparameters.md §2.3; Monitoring-Spec C9), or when monitoring cannot
    load (Monitoring-Spec M1).
    """
    # Set up logging
    setup_logging()
    logger = logging.getLogger()
    logger.info("Starting Flask application...")

    app = Flask(__name__)
    app.config.from_object(config_class)
    from app.clock import CLOCK_MODES

    if str(app.config.get("CLOCK_MODE", "wall")).lower() not in CLOCK_MODES:
        raise ValueError(f"CLOCK_MODE must be one of {CLOCK_MODES}, got {app.config.get('CLOCK_MODE')!r}.")
    cli_command = _invoked_cli_command()
    exempt = cli_command in set(app.config.get("HYPERPARAMS_CHECK_EXEMPT_COMMANDS") or ())

    _prepare_sqlite(app)

    # Initialize database and migration extensions
    db.init_app(app)
    migrate.init_app(app, db)
    _install_sqlite_pragmas(app)

    # Hyperparameters (Hyperparameters.md §2; contract D16). An invalid file
    # stops start-up, except for `flask hyperparams ...`, which must be able
    # to report the problem.
    app.hyperparams_error = None
    try:
        app.hyperparams = load_hyperparams(
            app.config.get("HYPERPARAMS_PATH"),
            overrides=app.config.get("HYPERPARAMS_OVERRIDES"),
        )
    except HyperparameterError as exc:
        if cli_command != "hyperparams":
            raise
        logger.error("Invalid hyperparameters (continuing for `flask hyperparams`): %s", exc)
        app.hyperparams_error = str(exc)
        app.hyperparams = default_hyperparams()
    # Deprecated mirrors for code that still reads the old warm-up keys.
    app.config["WARMUP_COHORT_MIN_DYADS"] = app.hyperparams.n_warm
    app.config["WARMUP_WEEKS"] = app.hyperparams.t_warm
    logger.info(
        "Hyperparameters resolved from %s (sha256=%s)",
        app.hyperparams.source_path or "design defaults",
        app.hyperparams.sha256,
    )

    # Create RL algorithm instance
    algo_name = app.config.get("RL_ALGORITHM", DEFAULT_ALGORITHM)
    if algo_name == "thompson_sampling":
        app.rl_algorithm = ThompsonSamplingAlgorithm(seed=app.config.get("RL_ALGORITHM_SEED"), app=app)
    elif algo_name == "random_baseline":
        app.rl_algorithm = RandomBaselineAlgorithm(seed=app.config.get("RL_ALGORITHM_SEED"))
    elif algo_name == "always_send":
        app.rl_algorithm = AlwaysSendAlgorithm(seed=app.config.get("RL_ALGORITHM_SEED"))
    elif algo_name == "always_none":
        app.rl_algorithm = AlwaysNoneAlgorithm(seed=app.config.get("RL_ALGORITHM_SEED"))
    elif algo_name in SAMPLER_ALGORITHMS:
        sampler = _load_or_init_sample_buffer(app)
        app.sampler = sampler
        app.rl_algorithm = SAMPLER_ALGORITHMS[algo_name](
            seed=app.config.get("RL_ALGORITHM_SEED"),
            app=app,
            sampler=sampler,
        )
    else:
        app.rl_algorithm = FlatProbRLAlgorithm(seed=app.config.get("RL_ALGORITHM_SEED"))

    # Register blueprints
    from app.routes.group import group_blueprint
    from app.routes.action import action_blueprint
    from app.routes.data import data_blueprint
    from app.routes.update import update_blueprint

    app.register_blueprint(group_blueprint, url_prefix="/api/v1")
    app.register_blueprint(action_blueprint, url_prefix="/api/v1")
    app.register_blueprint(data_blueprint, url_prefix="/api/v1")
    app.register_blueprint(update_blueprint, url_prefix="/api/v1")

    @app.get("/healthz")
    def healthz():
        """Lightweight VM/container readiness check; never returns study data."""
        db.session.execute(text("SELECT 1"))
        return {"status": "ok"}, 200

    # Monitoring (Monitoring-Spec §2, §8): blueprint, CLI group, after-response
    # code logger. Deliberately unguarded: the API refuses to start if
    # monitoring cannot load (M1).
    from app import monitoring

    monitoring.init_app(app)

    # HTTP audit: method/path/sizes only — never log request or response bodies (PHI).
    @app.before_request
    def log_request_info():
        logging.info(
            "HTTP request: method=%s path=%s content_length=%s",
            request.method,
            request.path,
            request.content_length,
        )

    @app.after_request
    def log_response_info(response):
        cl = response.calculate_content_length()
        logging.info(
            "HTTP response: status=%s content_length=%s",
            response.status,
            cl if cl is not None else "?",
        )
        return response

    _register_error_handlers(app, logger)

    with app.app_context():
        # Create tables for models
        db.create_all()
        # An older deployed schema may not yet have every column mapped by the
        # current models.  Let ``flask db upgrade`` reach Alembic without
        # querying those models first; normal application startup initializes
        # and validates state after the migration has completed.
        if cli_command != "db":
            initialize_model_parameters(app)
        # Refuse to start if the resolved set differs from the frozen one
        # (C9 alarm first). Exempt CLI commands inspect or repair the set, or
        # only read the database, so they only warn (and raise no alarm).
        if exempt:
            frozen = latest_frozen()
            if frozen is not None and frozen.sha256 != app.hyperparams.sha256:
                logger.warning(
                    "Resolved hyperparameters (sha256=%s) differ from frozen version %d "
                    "(sha256=%s); tolerated for `flask %s` only.",
                    app.hyperparams.sha256,
                    frozen.version,
                    frozen.sha256,
                    cli_command,
                )
        else:
            check_startup_against_frozen(app.hyperparams)
        # Restore sampler cursor to the furthest recorded position so a server
        # restart resumes the stream where it stopped (rather than
        # re-consuming primitives that were already used).
        if cli_command != "db" and algo_name in SAMPLER_ALGORITHMS:
            _restore_sampler_cursor(app)

    # Register CLI commands
    register_cli_commands(app)
    register_hyperparams_cli(app)

    return app


# ------------------------------------------------------------------ helpers

_FAULT_NEUTRAL = {
    "FAULT_ACTION_CRASH": frozenset(),
    "FAULT_FORCE_PI": None,
    "FAULT_SAMPLER_SHIFT": 0,
    "FAULT_UPDATE_HANG_SECONDS": 0.0,
}


def fault_setting(name: str, app=None):
    """
    A fault-injection setting (contract D21): ``app.config[name]`` when
    ``FAULT_INJECTION_ENABLED`` is True, else its neutral value
    (empty set / ``None`` / ``0``). ``app`` defaults to the current app.
    """
    if name not in _FAULT_NEUTRAL:
        raise KeyError(f"unknown fault-injection setting {name!r}")
    if app is None:
        from flask import current_app

        app = current_app
    if not app.config.get("FAULT_INJECTION_ENABLED", False):
        return _FAULT_NEUTRAL[name]
    value = app.config.get(name)
    if value is None or value == "":
        return _FAULT_NEUTRAL[name]
    if name == "FAULT_ACTION_CRASH":
        if isinstance(value, str):
            value = value.split(",")
        return frozenset(str(v).strip() for v in value if str(v).strip())
    if name == "FAULT_SAMPLER_SHIFT":
        return int(value)
    return float(value)


def _invoked_cli_command() -> str | None:
    """
    The ``flask`` sub-command this process runs (``"hyperparams"``,
    ``"db"``, ...), or ``None`` when the process is not the flask CLI.
    """
    argv = list(sys.argv or [])
    if not argv:
        return None
    prog = os.path.basename(argv[0] or "")
    is_flask = prog in ("flask", "flask.exe") or (
        prog == "__main__.py" and os.path.basename(os.path.dirname(argv[0])) == "flask"
    )
    if not is_flask:
        return None
    args = iter(argv[1:])
    for arg in args:
        if arg in ("--app", "-A", "--env-file", "-e"):
            next(args, None)
        elif arg.startswith("-"):
            continue
        else:
            return arg
    return None


def _sqlite_file_path(uri: str | None) -> str | None:
    """Filesystem path of a file-backed SQLite URI, ``None`` for other databases and in-memory SQLite."""
    if not uri or not uri.startswith("sqlite"):
        return None
    path = uri.split("://", 1)[1] if "://" in uri else ""
    if path.startswith("/"):
        path = path[1:]
    path = path.split("?", 1)[0]
    if path in ("", ":memory:") or path.startswith("file::memory:"):
        return None
    return path


def _prepare_sqlite(app) -> None:
    """For a file-backed SQLite DB: create its directory and add a driver busy timeout."""
    path = _sqlite_file_path(app.config.get("SQLALCHEMY_DATABASE_URI"))
    if path is None:
        return
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    options = dict(app.config.get("SQLALCHEMY_ENGINE_OPTIONS") or {})
    connect_args = dict(options.get("connect_args") or {})
    connect_args.setdefault("timeout", int(app.config.get("SQLITE_BUSY_TIMEOUT_MS", 30_000)) / 1000.0)
    connect_args.setdefault("check_same_thread", False)
    options["connect_args"] = connect_args
    app.config["SQLALCHEMY_ENGINE_OPTIONS"] = options


def _install_sqlite_pragmas(app) -> None:
    """
    WAL journaling and a busy timeout on every connection of a file-backed
    SQLite DB, so the API server and the monitoring process can share it
    (Monitoring-Spec §2.3).
    """
    if _sqlite_file_path(app.config.get("SQLALCHEMY_DATABASE_URI")) is None:
        return
    busy_ms = int(app.config.get("SQLITE_BUSY_TIMEOUT_MS", 30_000))
    use_wal = bool(app.config.get("SQLITE_WAL", True))

    def _on_connect(dbapi_connection, _record):
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute(f"PRAGMA busy_timeout = {busy_ms}")
            if use_wal:
                cursor.execute("PRAGMA journal_mode = WAL")
                cursor.execute("PRAGMA synchronous = NORMAL")
        finally:
            cursor.close()

    with app.app_context():
        event.listen(db.engine, "connect", _on_connect)


def _register_error_handlers(app, logger) -> None:
    """
    Framework-level answers in the common envelope with a fresh nonempty rid
    (contract D19; API-Spec §2 preamble; Monitoring-Spec §6.7):
    404 ``XX-404-ROUTE``, 405 ``XX-405-METHOD``, 500 ``XX-500-UNCAUGHT``.
    """
    from app.routes.envelope import envelope, new_rid

    @app.errorhandler(404)
    def handle_not_found(_e):
        return envelope(404, "Not Found", "Unknown route.", new_rid())

    @app.errorhandler(405)
    def handle_method_not_allowed(_e):
        return envelope(405, "Method Not Allowed", "Method not allowed.", new_rid())

    @app.errorhandler(Exception)
    def handle_exception(e):
        """Last-resort handler: every other exception gets the common envelope."""
        if isinstance(e, HTTPException) and e.code is not None and e.code < 500:
            # Other client errors raised by the framework (e.g. a body that is
            # not JSON) keep their HTTP status; D3 flags any unmapped answer.
            return envelope(e.code, e.name, e.description or e.name, new_rid())
        logger.error("Unhandled Exception: %s", str(e), exc_info=True)
        try:
            db.session.rollback()
        except Exception:  # noqa: BLE001 - the answer must still go out
            logger.exception("Session rollback failed in the last-resort handler.")
        return envelope(500, "Internal Error", "Internal server error.", new_rid())


def _load_or_init_sample_buffer(app) -> DeterministicSampleStream:
    """Load the pre-sampled buffer from SAMPLE_BUFFER_PATH, generating a fresh
    one if it doesn't exist and SAMPLE_BUFFER_AUTO_INIT is True.

    Special case: if SAMPLE_BUFFER_PATH is falsy, generate a fresh in-memory
    buffer from SAMPLE_BUFFER_SEED without persisting to disk. Useful for
    tests that want per-boot determinism without cross-test contamination.

    Freshly generated buffers also store the allocation's Monte Carlo bank for
    ``allocation.mc_seed`` / ``allocation.mc_samples`` (contract D10).
    """
    hp = app.hyperparams
    buf_path = app.config.get("SAMPLE_BUFFER_PATH")
    if not buf_path:
        seed = int(app.config.get("SAMPLE_BUFFER_SEED", 0))
        n_normals = int(app.config.get("SAMPLE_BUFFER_NORMALS", 500_000))
        n_uniforms = int(app.config.get("SAMPLE_BUFFER_UNIFORMS", 50_000))
        sampler = DeterministicSampleStream.fresh(
            n_normals=n_normals,
            n_uniforms=n_uniforms,
            seed=seed,
            mc_seed=hp.mc_seed,
            mc_samples=hp.mc_samples,
        )
        app.logger.info(
            "In-memory deterministic sample buffer "
            "(n_normals=%d, n_uniforms=%d, seed=%d)",
            n_normals,
            n_uniforms,
            seed,
        )
        return sampler
    # numpy adds the .npz suffix when saving; also accept that form.
    candidate_paths = [buf_path]
    if not buf_path.endswith(".npz"):
        candidate_paths.append(buf_path + ".npz")
    existing = next((p for p in candidate_paths if os.path.exists(p)), None)

    if existing:
        sampler = DeterministicSampleStream.load(existing)
        app.logger.info(
            "Loaded deterministic sample buffer from %s "
            "(n_normals=%d, n_uniforms=%d, cursor=%s, seed=%s, mc_bank=%s)",
            existing,
            sampler.n_normals,
            sampler.n_uniforms,
            sampler.cursor(),
            sampler.seed,
            sampler.mc_bank_info,
        )
        return sampler

    if not app.config.get("SAMPLE_BUFFER_AUTO_INIT", False):
        raise FileNotFoundError(
            f"SAMPLE_BUFFER_PATH={buf_path!r} does not exist and "
            "SAMPLE_BUFFER_AUTO_INIT is False. Run `flask init-buffer` or "
            "create the file manually."
        )

    seed = int(app.config.get("SAMPLE_BUFFER_SEED", 0))
    n_normals = int(app.config.get("SAMPLE_BUFFER_NORMALS", 5_000_000))
    n_uniforms = int(app.config.get("SAMPLE_BUFFER_UNIFORMS", 200_000))
    sampler = DeterministicSampleStream.fresh(
        n_normals=n_normals,
        n_uniforms=n_uniforms,
        seed=seed,
        mc_seed=hp.mc_seed,
        mc_samples=hp.mc_samples,
    )
    saved = sampler.save(buf_path)
    app.logger.info(
        "Auto-initialized deterministic sample buffer at %s "
        "(n_normals=%d, n_uniforms=%d, seed=%d, mc_bank=%s)",
        saved,
        n_normals,
        n_uniforms,
        seed,
        sampler.mc_bank_info,
    )
    return sampler


def _restore_sampler_cursor(app) -> None:
    """On startup, advance the in-memory sampler to the furthest cursor
    stamped on any Action (component-wise maximum of
    ``random_state.sampler_cursor_end``) so a crash/restart never
    re-consumes primitives, whatever order rows were committed in."""
    sampler: DeterministicSampleStream = getattr(app, "sampler", None)
    if sampler is None:
        return

    furthest: dict[str, int] | None = None
    source_rid = None
    rows = Action.query.with_entities(Action.rid, Action.random_state).filter(
        Action.random_state.isnot(None)
    )
    for rid, rs in rows:
        end = rs.get("sampler_cursor_end") if isinstance(rs, dict) else None
        if not isinstance(end, dict):
            continue
        normal, uniform = int(end.get("normal", 0)), int(end.get("uniform", 0))
        if furthest is None:
            furthest = {"normal": normal, "uniform": uniform}
            source_rid = rid
            continue
        if (normal, uniform) > (furthest["normal"], furthest["uniform"]):
            source_rid = rid
        furthest = {"normal": max(furthest["normal"], normal), "uniform": max(furthest["uniform"], uniform)}
    if furthest is None:
        return
    try:
        sampler.restore(furthest)
        app.logger.info(
            "Restored sampler cursor from actions (furthest rid=%s) cursor=%s",
            source_rid,
            furthest,
        )
    except Exception as exc:  # pragma: no cover - defensive
        app.logger.warning(
            "Failed to restore sampler cursor from actions: %s", exc
        )


def initialize_model_parameters(app):
    """
    Initialize the ModelParameters table with the legacy policy row if empty
    (contract D15: still written at init; inf_lsvi_eb does not score it).
    """
    if not ModelParameters.query.first():
        # Load priors from config or pickle file
        pickle_file = app.config["PRIORS_PICKLE_FILE"]
        priors = app.config["MODEL_PRIORS"]

        if pickle_file:
            try:
                with open(pickle_file, "rb") as f:
                    priors = pickle.load(f)
                app.logger.info("Loaded priors from pickle file: %s", pickle_file)
            except Exception as e:
                app.logger.error("Failed to load priors from pickle file: %s", str(e))
                raise e

        # Initialize the ModelParameters table
        default_params = ModelParameters(probability_of_action=priors["probability_of_action"])
        db.session.add(default_params)
        db.session.commit()
        app.logger.info("Initialized ModelParameters with priors: %s", priors)


def export_all_tables(directory: str, include_quarantine: bool = False) -> dict:
    """
    Write every table to ``<directory>/<table>.csv`` plus
    ``<directory>/MANIFEST.json`` = ``{"generated_at", "include_quarantine",
    "files": {filename: {"sha256", "rows"}}}`` (Monitoring-Spec D1; contract
    §2). ``upload_quarantine`` only with ``include_quarantine`` (D18).
    Returns the manifest. Requires an app context.
    """
    import json

    from app.clock import format_detroit, server_now
    from app.models import MANIFEST_FILENAME, dump_tables_to_csv

    files = dump_tables_to_csv(directory, include_quarantine=include_quarantine)
    manifest = {
        "generated_at": format_detroit(server_now()),
        "include_quarantine": bool(include_quarantine),
        "files": files,
    }
    with open(os.path.join(directory, MANIFEST_FILENAME), "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    return manifest


def register_cli_commands(app):
    """
    Registers custom CLI commands with the Flask app.
    """
    import click

    @app.cli.command("export-csv")
    @click.option("--out", "out_dir", default="exports", show_default=True, help="Output directory.")
    @click.option(
        "--include-quarantine",
        is_flag=True,
        default=False,
        help="Also export upload_quarantine (rejected upload bodies; access-controlled).",
    )
    def export_csv(out_dir, include_quarantine):
        """
        Export every database table to CSV files plus MANIFEST.json.
        """
        manifest = export_all_tables(out_dir, include_quarantine=include_quarantine)
        for filename, entry in sorted(manifest["files"].items()):
            print(f"  {os.path.join(out_dir, filename)} ({entry['rows']} rows)")
        print(f"Exported to {out_dir}/ (manifest: MANIFEST.json)")

    @app.cli.command("reset-db")
    def reset_db():
        """
        Drops all tables and recreates them using migrations.
        """
        print("Dropping all tables...")
        db.drop_all()
        db.session.commit()

        print("Recreating all tables...")
        subprocess.run(["flask", "db", "upgrade"], check=True)
        print("Database reset complete.")

    @app.cli.command("init-buffer")
    def init_buffer():
        """
        Generate a fresh deterministic sample buffer and write it to the
        path configured by SAMPLE_BUFFER_PATH.

        This is a pure "pre-study" step: it pre-samples a long sequence of
        Standard Gaussian floats and Uniform [0, 1) floats from a seed, and
        stores the allocation's Monte Carlo bank (M = allocation.mc_samples
        draws under allocation.mc_seed; Hyperparameters.md §5.4). At
        runtime, whenever the algorithm needs a random draw it pulls the
        next primitive(s) from this file instead of generating fresh ones.

        Given the same buffer file and the same sequence of server events,
        the algorithm produces byte-identical outputs every time.

        Config keys consulted:
          SAMPLE_BUFFER_PATH, SAMPLE_BUFFER_SEED,
          SAMPLE_BUFFER_NORMALS, SAMPLE_BUFFER_UNIFORMS.
        """
        path = app.config.get("SAMPLE_BUFFER_PATH")
        if not path:
            click.echo("SAMPLE_BUFFER_PATH is not set in config.", err=True)
            raise click.Abort()

        seed = int(app.config.get("SAMPLE_BUFFER_SEED", 0))
        n_normals = int(app.config.get("SAMPLE_BUFFER_NORMALS", 5_000_000))
        n_uniforms = int(app.config.get("SAMPLE_BUFFER_UNIFORMS", 200_000))
        hp = app.hyperparams

        if any(os.path.exists(p) for p in (path, path + ".npz")):
            click.echo(
                f"Refusing to overwrite existing buffer at {path}. "
                "Delete it first if you really want to regenerate.",
                err=True,
            )
            raise click.Abort()

        sampler = DeterministicSampleStream.fresh(
            n_normals=n_normals,
            n_uniforms=n_uniforms,
            seed=seed,
            mc_seed=hp.mc_seed,
            mc_samples=hp.mc_samples,
        )
        saved = sampler.save(path)
        click.echo(
            f"Wrote buffer to {saved}  "
            f"(n_normals={n_normals}, n_uniforms={n_uniforms}, seed={seed}, "
            f"mc_seed={hp.mc_seed}, mc_samples={hp.mc_samples})"
        )

    @app.cli.command("upgrade-schema")
    def upgrade_schema():
        """
        Idempotent schema-upgrade for in-place ADAPTS-HCT changes that arrived
        between releases.

        Creates any new tables (e.g. data_uploads, standardization_baselines)
        and idempotently adds the new flat-contract columns
        (actions.is_warmup / actions.warmup_reason, study_data.derived_at).
        Column drops (groups.warmup, model_update_requests.callback_url) and
        renames are handled by `flask db upgrade` (Alembic), not here; so are
        the inf_lsvi_eb / monitoring changes of migration 20261016_01.

        Safe to run repeatedly. Works against PostgreSQL and SQLite.
        """
        from sqlalchemy import inspect, text

        inspector = inspect(db.engine)
        dialect = db.engine.dialect.name
        bool_default = "FALSE" if dialect != "sqlite" else "0"

        new_columns = {
            "actions": [
                ("is_warmup", f"BOOLEAN NOT NULL DEFAULT {bool_default}"),
                ("warmup_reason", "VARCHAR(32)"),
            ],
            "study_data": [
                ("derived_at", "TIMESTAMP"),
            ],
        }
        for table_name, cols in new_columns.items():
            if table_name not in inspector.get_table_names():
                continue
            existing_cols = {c["name"] for c in inspector.get_columns(table_name)}
            for col_name, col_type in cols:
                if col_name in existing_cols:
                    print(f"  - {table_name}.{col_name} already present")
                    continue
                with db.engine.begin() as conn:
                    conn.execute(
                        text(f"ALTER TABLE {table_name} ADD COLUMN {col_name} {col_type}")
                    )
                print(f"  + added {table_name}.{col_name} ({dialect})")

        # Create any tables in the model metadata but not in the DB
        # (e.g. data_uploads, standardization_baselines).
        before = set(inspector.get_table_names())
        db.create_all()
        after = set(inspect(db.engine).get_table_names())
        new_tables = sorted(after - before)
        if new_tables:
            for tbl in new_tables:
                print(f"  + created table {tbl}")
        else:
            print("  - no new tables to create")

        print("Schema upgrade complete.")
