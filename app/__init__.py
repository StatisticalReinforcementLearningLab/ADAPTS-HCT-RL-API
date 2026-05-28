import subprocess
import logging
import os
import pickle
from flask import Flask, request, jsonify
from app.extensions import db, migrate
from app.logging_config import setup_logging
from app.algorithms.flat_prob import FlatProbRLAlgorithm
from app.algorithms.empirical_bayes import ThreeAgentEmpiricalBayesAlgorithm
from app.algorithms.eb_gradient import ThreeAgentEmpiricalBayesGradientAlgorithm
from app.algorithms.inf_lsvi_local import ThreeAgentInfLsviAlgorithm
from app.algorithms.inf_lsvi_pool import ThreeAgentInfLsviPooledAlgorithm
from app.algorithms.hybrid_rel_pool import HybridRelPoolAlgorithm
from app.algorithms.thompson_sampling import ThompsonSamplingAlgorithm
from app.algorithms.random_baseline import RandomBaselineAlgorithm
from app.algorithms.always_send import AlwaysSendAlgorithm
from app.algorithms.always_none import AlwaysNoneAlgorithm
from app.deterministic_sampler import DeterministicSampleStream
from app.models import Action, ModelParameters


def create_app(config_class="config.Config"):
    """
    Factory function to create and configure the Flask app.
    """
    # Set up logging
    setup_logging()
    logger = logging.getLogger()
    logger.info("Starting Flask application...")

    app = Flask(__name__)
    app.config.from_object(config_class)

    # Initialize database and migration extensions
    db.init_app(app)
    migrate.init_app(app, db)

    # Create RL algorithm instance
    algo_name = app.config.get("RL_ALGORITHM", "flat_prob")
    if algo_name == "thompson_sampling":
        app.rl_algorithm = ThompsonSamplingAlgorithm(seed=app.config.get("RL_ALGORITHM_SEED"), app=app)
    elif algo_name == "random_baseline":
        app.rl_algorithm = RandomBaselineAlgorithm(seed=app.config.get("RL_ALGORITHM_SEED"))
    elif algo_name == "always_send":
        app.rl_algorithm = AlwaysSendAlgorithm(seed=app.config.get("RL_ALGORITHM_SEED"))
    elif algo_name == "always_none":
        app.rl_algorithm = AlwaysNoneAlgorithm(seed=app.config.get("RL_ALGORITHM_SEED"))
    elif algo_name == "empirical_bayes":
        sampler = _load_or_init_sample_buffer(app)
        app.sampler = sampler
        app.rl_algorithm = ThreeAgentEmpiricalBayesAlgorithm(
            seed=app.config.get("RL_ALGORITHM_SEED"),
            app=app,
            sampler=sampler,
        )
    elif algo_name == "eb_gradient":
        sampler = _load_or_init_sample_buffer(app)
        app.sampler = sampler
        app.rl_algorithm = ThreeAgentEmpiricalBayesGradientAlgorithm(
            seed=app.config.get("RL_ALGORITHM_SEED"),
            app=app,
            sampler=sampler,
        )
    elif algo_name == "inf_lsvi":
        sampler = _load_or_init_sample_buffer(app)
        app.sampler = sampler
        app.rl_algorithm = ThreeAgentInfLsviAlgorithm(
            seed=app.config.get("RL_ALGORITHM_SEED"),
            app=app,
            sampler=sampler,
        )
    elif algo_name == "inf_lsvi_pool":
        sampler = _load_or_init_sample_buffer(app)
        app.sampler = sampler
        app.rl_algorithm = ThreeAgentInfLsviPooledAlgorithm(
            seed=app.config.get("RL_ALGORITHM_SEED"),
            app=app,
            sampler=sampler,
        )
    elif algo_name == "hybrid_rel_pool":
        sampler = _load_or_init_sample_buffer(app)
        app.sampler = sampler
        app.rl_algorithm = HybridRelPoolAlgorithm(
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

    # Monitoring (Section 6 of main.tex): blueprint + CLI commands.
    # The Monitoring_Algorithm package must be on PYTHONPATH (or copied into
    # this repo). All checks share this app's SQLAlchemy `db` instance.
    try:
        from monitoring import MonitorEvent  # noqa: F401  (registers ORM)
        from monitoring.blueprint import monitor_bp
        from monitoring.cli import register_cli
        # Flask 3.x: register_blueprint's url_prefix overrides the blueprint's own,
        # so we pass the full "/api/v1/monitor" prefix explicitly.
        app.register_blueprint(monitor_bp, url_prefix="/api/v1/monitor")
        register_cli(app)
        with app.app_context():
            db.create_all()  # ensure monitor_events table exists
        logger.info("Monitoring blueprint registered at /api/v1/monitor")
    except Exception as exc:
        logger.warning("monitoring registration failed: %r", exc)

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
    
    # Global error handler
    @app.errorhandler(Exception)
    def handle_exception(e):
        """
        Catch all unhandled exceptions, log them, and return a 500 error.
        """
        logger.error("Unhandled Exception: %s", str(e), exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

    with app.app_context():
        # Create tables for models
        db.create_all()
        initialize_model_parameters(app)
        # Restore sampler cursor to the last recorded position from the most
        # recent Action so a server restart resumes the stream where it
        # stopped (rather than re-consuming primitives that were already used).
        if algo_name in ("empirical_bayes", "eb_gradient", "inf_lsvi",
                         "inf_lsvi_pool", "hybrid_rel_pool"):
            _restore_sampler_cursor(app)

    # Register CLI commands
    register_cli_commands(app)

    return app


def _load_or_init_sample_buffer(app) -> DeterministicSampleStream:
    """Load the pre-sampled buffer from SAMPLE_BUFFER_PATH, generating a fresh
    one if it doesn't exist and SAMPLE_BUFFER_AUTO_INIT is True.

    Special case: if SAMPLE_BUFFER_PATH is falsy, generate a fresh in-memory
    buffer from SAMPLE_BUFFER_SEED without persisting to disk. Useful for
    tests that want per-boot determinism without cross-test contamination.
    """
    buf_path = app.config.get("SAMPLE_BUFFER_PATH")
    if not buf_path:
        seed = int(app.config.get("SAMPLE_BUFFER_SEED", 0))
        n_normals = int(app.config.get("SAMPLE_BUFFER_NORMALS", 500_000))
        n_uniforms = int(app.config.get("SAMPLE_BUFFER_UNIFORMS", 5_000))
        sampler = DeterministicSampleStream.fresh(
            n_normals=n_normals, n_uniforms=n_uniforms, seed=seed
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
            "(n_normals=%d, n_uniforms=%d, cursor=%s, seed=%s)",
            existing,
            sampler.n_normals,
            sampler.n_uniforms,
            sampler.cursor(),
            sampler.seed,
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
    n_uniforms = int(app.config.get("SAMPLE_BUFFER_UNIFORMS", 10_000))
    sampler = DeterministicSampleStream.fresh(
        n_normals=n_normals, n_uniforms=n_uniforms, seed=seed
    )
    saved = sampler.save(buf_path)
    app.logger.info(
        "Auto-initialized deterministic sample buffer at %s "
        "(n_normals=%d, n_uniforms=%d, seed=%d)",
        saved,
        n_normals,
        n_uniforms,
        seed,
    )
    return sampler


def _restore_sampler_cursor(app) -> None:
    """On startup, scan the most recent Action row for a cursor stamp and
    advance the in-memory sampler to that point so a crash/restart doesn't
    re-consume primitives."""
    sampler: DeterministicSampleStream = getattr(app, "sampler", None)
    if sampler is None:
        return

    latest = (
        Action.query.filter(Action.random_state.isnot(None))
        .order_by(Action.timestamp.desc())
        .first()
    )
    if latest is None:
        return

    rs = latest.random_state or {}
    end = rs.get("sampler_cursor_end") if isinstance(rs, dict) else None
    if not end:
        return
    try:
        sampler.restore(end)
        app.logger.info(
            "Restored sampler cursor from Action rid=%s cursor=%s",
            latest.rid,
            end,
        )
    except Exception as exc:  # pragma: no cover - defensive
        app.logger.warning(
            "Failed to restore sampler cursor from latest Action: %s", exc
        )

def initialize_model_parameters(app):
    """
    Initialize the ModelParameters table with default priors if empty.
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

def register_cli_commands(app):
    """
    Registers custom CLI commands with the Flask app.
    """

    @app.cli.command("export-csv")
    def export_csv():
        """
        Export all database tables to CSV files in the exports/ directory.
        """
        import csv
        import os
        from app.models import (
            Group,
            Action,
            StudyData,
            ModelUpdateRequests,
            ModelParameters,
            ThompsonSamplingParams,
            EmpiricalBayesSnapshot,
            UpdateReproducibilitySnapshot,
        )

        export_dir = "exports"
        os.makedirs(export_dir, exist_ok=True)
        models = [
            Group,
            Action,
            StudyData,
            ModelUpdateRequests,
            ModelParameters,
            ThompsonSamplingParams,
            EmpiricalBayesSnapshot,
            UpdateReproducibilitySnapshot,
        ]

        for model in models:
            table_name = model.__tablename__
            file_path = os.path.join(export_dir, f"{table_name}.csv")
            rows = db.session.query(model).all()
            columns = [col.key for col in model.__table__.columns]
            with open(file_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(columns)
                for row in rows:
                    writer.writerow([getattr(row, col) for col in columns])
            print(f"  {file_path} ({len(rows)} rows)")
        print(f"Exported to {export_dir}/")

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
        Standard Gaussian floats and Uniform [0, 1) floats from a seed. At
        runtime, whenever the algorithm needs a random draw it pulls the
        next primitive(s) from this file instead of generating fresh ones.

        Given the same buffer file and the same sequence of server events,
        the algorithm produces byte-identical outputs every time.

        Config keys consulted:
          SAMPLE_BUFFER_PATH, SAMPLE_BUFFER_SEED,
          SAMPLE_BUFFER_NORMALS, SAMPLE_BUFFER_UNIFORMS.
        """
        import click

        path = app.config.get("SAMPLE_BUFFER_PATH")
        if not path:
            click.echo("SAMPLE_BUFFER_PATH is not set in config.", err=True)
            raise click.Abort()

        seed = int(app.config.get("SAMPLE_BUFFER_SEED", 0))
        n_normals = int(app.config.get("SAMPLE_BUFFER_NORMALS", 5_000_000))
        n_uniforms = int(app.config.get("SAMPLE_BUFFER_UNIFORMS", 10_000))

        if any(os.path.exists(p) for p in (path, path + ".npz")):
            click.echo(
                f"Refusing to overwrite existing buffer at {path}. "
                "Delete it first if you really want to regenerate.",
                err=True,
            )
            raise click.Abort()

        sampler = DeterministicSampleStream.fresh(
            n_normals=n_normals, n_uniforms=n_uniforms, seed=seed
        )
        saved = sampler.save(path)
        click.echo(
            f"Wrote buffer to {saved}  "
            f"(n_normals={n_normals}, n_uniforms={n_uniforms}, seed={seed})"
        )

    @app.cli.command("upgrade-schema")
    def upgrade_schema():
        """
        Idempotent schema-upgrade for in-place ADAPTS-HCT changes that arrived
        between releases.

        Adds:
        - groups.warmup boolean column (default False) — needed for the warmup
          override on the first 5 dyads.
        - standardization_baselines table — populated lazily by the algorithm.

        Safe to run repeatedly. Works against PostgreSQL and SQLite.
        """
        from sqlalchemy import inspect, text

        inspector = inspect(db.engine)

        # 1. Add groups.warmup if it doesn't already exist.
        if "groups" in inspector.get_table_names():
            existing_cols = {c["name"] for c in inspector.get_columns("groups")}
            if "warmup" not in existing_cols:
                dialect = db.engine.dialect.name
                if dialect == "postgresql":
                    stmt = text(
                        "ALTER TABLE groups "
                        "ADD COLUMN warmup BOOLEAN NOT NULL DEFAULT FALSE"
                    )
                elif dialect == "sqlite":
                    stmt = text(
                        "ALTER TABLE groups "
                        "ADD COLUMN warmup BOOLEAN NOT NULL DEFAULT 0"
                    )
                else:
                    # Generic fallback; may need tweaking for exotic dialects.
                    stmt = text(
                        "ALTER TABLE groups "
                        "ADD COLUMN warmup BOOLEAN NOT NULL DEFAULT FALSE"
                    )
                with db.engine.begin() as conn:
                    conn.execute(stmt)
                print(f"  + added groups.warmup ({dialect})")
            else:
                print("  - groups.warmup already present")
        else:
            print("  ! groups table not present yet; will be created by create_all")

        # 2. Create any tables that exist in the model metadata but not in the
        #    DB (e.g. standardization_baselines, future tables).
        before = set(inspector.get_table_names())
        db.create_all()
        # Re-inspect after create_all
        after = set(inspect(db.engine).get_table_names())
        new_tables = sorted(after - before)
        if new_tables:
            for tbl in new_tables:
                print(f"  + created table {tbl}")
        else:
            print("  - no new tables to create")

        print("Schema upgrade complete.")
