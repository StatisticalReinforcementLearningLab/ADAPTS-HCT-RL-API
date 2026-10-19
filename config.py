import json
import os
import tempfile

from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------- env helpers
#
# Operational settings may be overridden by environment variables of the same
# name (Monitoring-Spec §10: "config.py (or environment variables)"). Values
# are read once, when this module is imported.


def _env_str(name, default):
    value = os.getenv(name)
    return default if value is None or value == "" else value


def _env_int(name, default):
    value = os.getenv(name)
    return default if value is None or value.strip() == "" else int(value)


def _env_float(name, default):
    value = os.getenv(name)
    return default if value is None or value.strip() == "" else float(value)


def _env_json(name, default):
    value = os.getenv(name)
    return default if value is None or value.strip() == "" else json.loads(value)


def _env_set(name):
    """Comma-separated list -> frozenset (empty when unset)."""
    value = os.getenv(name) or ""
    return frozenset(item.strip() for item in value.split(",") if item.strip())


def _env_optional_float(name):
    value = os.getenv(name)
    return None if value is None or value.strip() == "" else float(value)


def _default_run_dir():
    """Simulation run directory: $ADAPTS_RUN_DIR, else <tmp>/adapts_sim_run (not created here)."""
    return os.getenv("ADAPTS_RUN_DIR") or os.path.join(tempfile.gettempdir(), "adapts_sim_run")


class Config:
    # General Configuration
    DEBUG = True
    TESTING = False

    # Database Configuration
    # PostgreSQL connection string format: postgresql://username:password@host:port/database_name
    #
    # For Mac users:
    # 1. If using a dedicated PostgreSQL user:
    #    "postgresql://justin_user:your_password@localhost:5432/justin_rl_db"
    #
    # 2. If using your Mac username (no password for local connections):
    #    "postgresql://your_mac_username@localhost:5432/justin_rl_db"
    #
    # 3. Using environment variable (recommended for security):
    #    Set SQLALCHEMY_DATABASE_URI in your shell: export SQLALCHEMY_DATABASE_URI="postgresql://..."
    #    This will automatically be used instead of the default value below
    #
    # Default port for PostgreSQL is 5432. Change if your PostgreSQL uses a different port.
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_DATABASE_URI = os.getenv('SQLALCHEMY_DATABASE_URI') or "postgresql://zipingxu@localhost:5432/justin_rl_db"
    SQLALCHEMY_ENGINE_OPTIONS = {
        "pool_recycle": 3600,  # Recycle connections after 3600 seconds (1 hour)
    }
    # For a file-backed SQLite database the API server and the separate
    # monitoring process share one file (Monitoring-Spec §2.3): create_app
    # enables WAL journaling and this busy timeout on every connection.
    SQLITE_WAL = True
    SQLITE_BUSY_TIMEOUT_MS = 30_000
    RL_ALGORITHM_SEED = 42  # Seed for RL Algorithm random state

    # ---- Learner (implementation contract D1) ----
    # "inf_lsvi_eb" (main_new.tex, the deployed design) is the default. Legacy
    # learners stay selectable: "flat_prob", "thompson_sampling",
    # "empirical_bayes", "eb_gradient", "inf_lsvi", "inf_lsvi_pool",
    # "hybrid_rel_pool", "random_baseline", "always_send", "always_none".
    RL_ALGORITHM = os.getenv("RL_ALGORITHM", "inf_lsvi_eb")

    # ---- Hyperparameters (Hyperparameters.md §2; contract D16) ----
    # The learner hyperparameters live in hyperparameters.yaml, not here.
    # HYPERPARAMS_PATH: explicit file; None means $ADAPTS_HYPERPARAMS, else the
    # repo's hyperparameters.yaml (absent file = design defaults).
    # HYPERPARAMS_OVERRIDES: nested or dotted-key overrides applied on top of
    # the file (tests and simulations only; they enter the fingerprint).
    HYPERPARAMS_PATH = None
    HYPERPARAMS_OVERRIDES = None
    # CLI commands that may run when the file differs from the frozen set
    # (they inspect or repair it, or only read the database).
    HYPERPARAMS_CHECK_EXEMPT_COMMANDS = ("hyperparams", "db", "export-csv", "monitor")

    # ---- Server-side warm-up (API-Spec §2.2; contract D2/D3) ----
    # The warm-up sizes are hyperparameters now (warmup.n_warm / warmup.t_warm
    # in hyperparameters.yaml). create_app mirrors them into the deprecated
    # WARMUP_COHORT_MIN_DYADS / WARMUP_WEEKS keys for code that still reads
    # them; do not set those keys here.

    # ---- Study clock (contract D4; Monitoring-Spec §3) ----
    # "wall": study time is the server's Detroit wall clock (production).
    # "host": study time is the latest host request timestamp (simulations
    # that compress the calendar).
    CLOCK_MODE = _env_str("CLOCK_MODE", "wall")

    # /upload_data slot boundary: an upload stamped before this local hour is
    # the "am" upload of its day, at or after it the "pm" upload.
    UPLOAD_SLOT_NOON_HOUR = 12

    # ---- Fault injection (contract D21; testing and simulation only) ----
    # Ignored unless FAULT_INJECTION_ENABLED, which is never True in Config or
    # ProductionConfig. The FAULT_* values may come from env vars of the same
    # names (comma list / numbers).
    FAULT_INJECTION_ENABLED = False
    FAULT_ACTION_CRASH = _env_set("FAULT_ACTION_CRASH")  # decision types whose learned path raises
    FAULT_FORCE_PI = _env_optional_float("FAULT_FORCE_PI")  # overwrite the learned pi before A13
    FAULT_SAMPLER_SHIFT = _env_int("FAULT_SAMPLER_SHIFT", 0)  # draw u at cursor+shift, record cursor
    FAULT_UPDATE_HANG_SECONDS = _env_float("FAULT_UPDATE_HANG_SECONDS", 0.0)  # sleep inside the update thread

    # ---- Monitoring (Monitoring-Spec §10) ----
    MONITOR_STUDY_TZ = _env_str("MONITOR_STUDY_TZ", "America/Detroit")
    MONITOR_RUN_EVERY_SECONDS = _env_int("MONITOR_RUN_EVERY_SECONDS", 300)
    MONITOR_UPDATE_EXPECTED_BY = _env_str("MONITOR_UPDATE_EXPECTED_BY", "MON 03:30")  # C11
    MONITOR_WEEK_READY_BY = _env_str("MONITOR_WEEK_READY_BY", "MON 05:45")  # C7; policy deadline (D13)
    MONITOR_UPDATE_TIMEOUT_MIN = _env_int("MONITOR_UPDATE_TIMEOUT_MIN", 30)  # F-U4
    # Expected /action calls per active dyad per day (API-Spec §3; Monitoring-Spec §3):
    # weekdays use Python's numbering (Monday = 0); partial_week says whether
    # the agent also decides during the dyad's partial week (study week 0).
    MONITOR_DECISION_CALENDAR = _env_json(
        "MONITOR_DECISION_CALENDAR",
        {
            "aya_message": {"per_day": 2, "weekdays": [0, 1, 2, 3, 4, 5, 6], "partial_week": True},
            "cp_message": {"per_day": 1, "weekdays": [0, 1, 2, 3, 4, 5, 6], "partial_week": True},
            "dyad_game": {"per_day": 1, "weekdays": [0], "partial_week": False},
        },
    )
    MONITOR_MESSAGE_FLOOR = _env_float("MONITOR_MESSAGE_FLOOR", 0.20)  # A7
    MONITOR_MESSAGE_CAP = _env_float("MONITOR_MESSAGE_CAP", 0.80)  # A6
    MONITOR_GAME_STREAK_WEEKS = _env_int("MONITOR_GAME_STREAK_WEEKS", 3)  # A17
    MONITOR_HOST_PULL_TIMES = _env_json("MONITOR_HOST_PULL_TIMES", ["05:50", "17:50"])  # E6
    MONITOR_KEEP_DAYS = _env_json("MONITOR_KEEP_DAYS", None)  # no automatic deletion

    # /update runs on a background thread unless this is True (tests run it
    # inline because a thread cannot share the in-memory SQLite connection).
    UPDATE_RUN_INLINE = False

    # Prior Configuration (legacy learners only; not used by inf_lsvi_eb)
    # If you specify a pickle file, it should be a dictionary with the same keys
    # as the entries in the ModelParameters table. Otherwise, see the next setting
    # to specify the individual keys in text format.
    PRIORS_PICKLE_FILE = None  # Path to the pickled priors file

    # If you don't want to use a pickle file, you can specify the priors directly
    # using the following format. The keys should match the entries in the
    # ModelParameters table defined in app/routes/models.py.
    MODEL_PRIORS = {
        "probability_of_action": 0.5,
    }

    # Backup database before processing an update request
    BACKUP_DATABASE = True

    # Before each /update job runs the learner, write study_data + actions + groups to disk
    SAVE_UPDATE_REPRO_SNAPSHOTS = True
    REPRO_SNAPSHOT_ROOT = "repro_snapshots"

    # ---- EB-Gradient smooth allocation (legacy; Hyperparameters.md §7) ----
    # Superseded by allocation.* in hyperparameters.yaml for inf_lsvi_eb.
    # ρ(x) = L_min + (L_max - L_min) / (1 + c * exp(-b * x))**k. The
    # action-selection probability π is the Monte Carlo expectation of
    # ρ(m + sqrt(v) z) over a pre-sampled bank of M N(0,1) draws (seed +
    # bank size below). See Prior_Construction_Note.tex §Generalized-logistic
    # smooth allocation. Defaults match MiWaves Figure 3.
    SMOOTH_ALLOC_LMIN = 0.2
    SMOOTH_ALLOC_LMAX = 0.8
    SMOOTH_ALLOC_C = 5.0
    SMOOTH_ALLOC_B = 20.0
    SMOOTH_ALLOC_K = 1.0
    SMOOTH_ALLOC_MC_SAMPLES = 500
    SMOOTH_ALLOC_MC_SEED = 12345

    # ---- EB-Gradient inverse-Gamma prior on diag(Σ_0) (legacy) ----
    # τ_d² ~ InvGamma(ν₀/2, ν₀·τ₀²/2). τ₀² is deliberately large so the
    # cold pool is uninformative; ν₀ is the pseudo-dyad count that controls
    # how stubbornly the prior insists. Superseded by eb.nu0 and the derived
    # tau0 of hyperparameters.yaml for inf_lsvi_eb.
    EB_PRIOR_TAU0_SQ = 10.0
    EB_PRIOR_NU0 = 2.0

    # ---- Deterministic sampling (reproducibility; API-Spec §6, contract D11) ----
    # All randomness is consumed from a pre-sampled "stream" of standard
    # normals + uniforms stored on disk, plus the fixed Monte Carlo bank of the
    # allocation (allocation.mc_seed / mc_samples). Generate one with
    # `flask init-buffer` before the study starts.
    SAMPLE_BUFFER_PATH = "buffers/study_buffer.npz"
    # If the file does not exist on app startup, one will be auto-generated
    # using the seed and sizes below. Use `flask init-buffer` for finer
    # control (e.g. study-specific seeds).
    SAMPLE_BUFFER_AUTO_INIT = True
    SAMPLE_BUFFER_SEED = 20260421
    # inf_lsvi_eb consumes exactly one uniform per minted decision and no
    # normals (contract D11): 25 dyads x ~300 decisions ~ 7,500 uniforms, so
    # 200,000 leaves a wide margin. Normals are only used by legacy learners.
    SAMPLE_BUFFER_NORMALS = 5_000_000
    SAMPLE_BUFFER_UNIFORMS = 200_000


class DevelopmentConfig(Config):
    DEBUG = True


class TestingConfig(Config):
    TESTING = True
    BACKUP_DATABASE = False
    SAVE_UPDATE_REPRO_SNAPSHOTS = False
    REPRO_SNAPSHOT_ROOT = os.path.join(tempfile.gettempdir(), "adapts_test_repro_snapshots")
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"
    SQLALCHEMY_ENGINE_OPTIONS = {}
    RL_ALGORITHM = "inf_lsvi_eb"
    CLOCK_MODE = "host"
    UPDATE_RUN_INLINE = True
    # Hermetic fault-injection defaults (tests set app.config["FAULT_*"]
    # explicitly; env vars from the developer's shell are not inherited).
    FAULT_INJECTION_ENABLED = True
    FAULT_ACTION_CRASH = frozenset()
    FAULT_FORCE_PI = None
    FAULT_SAMPLER_SHIFT = 0
    FAULT_UPDATE_HANG_SECONDS = 0.0
    # Tests use an in-memory buffer (no path): each app boot gets a fresh
    # buffer generated from the seed below, so tests don't accumulate
    # cross-run cursor state.
    SAMPLE_BUFFER_PATH = None
    SAMPLE_BUFFER_AUTO_INIT = True
    SAMPLE_BUFFER_SEED = 7
    SAMPLE_BUFFER_NORMALS = 500_000
    SAMPLE_BUFFER_UNIFORMS = 50_000


class SimulationConfig(Config):
    """
    Live-server simulations and fault scenarios (contract §4): a file-backed
    SQLite database shared with the monitoring process, the host's logical
    clock, fault injection honoured, /update on a real background thread, and
    every file the run writes under one run directory ($ADAPTS_RUN_DIR, else
    <tmp>/adapts_sim_run).
    """

    DEBUG = False
    TESTING = False
    RUN_DIR = _default_run_dir()
    SQLALCHEMY_DATABASE_URI = os.getenv("SQLALCHEMY_DATABASE_URI") or (
        "sqlite:///" + os.path.join(RUN_DIR, "adapts_sim.db")
    )
    SQLALCHEMY_ENGINE_OPTIONS = {}
    RL_ALGORITHM = os.getenv("RL_ALGORITHM", "inf_lsvi_eb")
    CLOCK_MODE = "host"
    FAULT_INJECTION_ENABLED = True
    UPDATE_RUN_INLINE = False
    BACKUP_DATABASE = False
    SAVE_UPDATE_REPRO_SNAPSHOTS = True
    REPRO_SNAPSHOT_ROOT = os.path.join(RUN_DIR, "repro_snapshots")
    SAMPLE_BUFFER_PATH = os.path.join(RUN_DIR, "study_buffer.npz")
    SAMPLE_BUFFER_AUTO_INIT = True
    SAMPLE_BUFFER_NORMALS = 500_000
    SAMPLE_BUFFER_UNIFORMS = 200_000


class ProductionConfig(Config):
    DEBUG = False
    FAULT_INJECTION_ENABLED = False
