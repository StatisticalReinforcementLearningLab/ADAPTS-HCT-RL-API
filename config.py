import os
from dotenv import load_dotenv

load_dotenv()


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
    #    Set DATABASE_URL in your shell: export DATABASE_URL="postgresql://..."
    #    This will automatically be used instead of the default value below
    #
    # Default port for PostgreSQL is 5432. Change if your PostgreSQL uses a different port.
    # SQLALCHEMY_DATABASE_URI = os.getenv(
        # "DATABASE_URL", "postgresql://myuser:mypassword@localhost:5432/mydatabase"
    # )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_DATABASE_URI = os.getenv('SQLALCHEMY_DATABASE_URI') or "postgresql://zipingxu@localhost:5432/justin_rl_db"
    SQLALCHEMY_ENGINE_OPTIONS = {
        "pool_recycle": 3600,  # Recycle connections after 3600 seconds (1 hour)
    }
    RL_ALGORITHM_SEED = 42  # Seed for RL Algorithm random state

    # ---- Server-side warm-up (API-Spec §3.2) ----
    # A decision is purely randomized (Bernoulli(0.5), drawn from the sample
    # buffer) iff the cohort still has fewer than WARMUP_COHORT_MIN_DYADS
    # registered dyads, OR the dyad has had fewer than
    # WARMUP_WEEK1_CP_DECISIONS cp_message decisions (its first active week;
    # cp_message fires once per active day, so its count is a day clock shared
    # by all three agents). The host has no say in warm-up.
    WARMUP_COHORT_MIN_DYADS = 5
    WARMUP_WEEK1_CP_DECISIONS = 6

    # Prior Configuration
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

    # RL algorithm: "flat_prob", "thompson_sampling", "empirical_bayes",
    # or "eb_gradient" (MAP marginal-likelihood EB + generalized-logistic
    # smooth allocation; see Prior_Construction_Note.tex).
    RL_ALGORITHM = os.getenv("RL_ALGORITHM", "empirical_bayes")

    # ---- EB-Gradient smooth allocation (generalized logistic) ----
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

    # ---- EB-Gradient inverse-Gamma prior on diag(Σ_0) ----
    # τ_d² ~ InvGamma(ν₀/2, ν₀·τ₀²/2). τ₀² is deliberately large so the
    # cold pool is uninformative; ν₀ is the pseudo-dyad count that controls
    # how stubbornly the prior insists. At N ≲ ν₀ the prior dominates and
    # the EB posterior collapses to the per-dyad Inf-LSVI fit; at N ≫ ν₀
    # the data dominates and we recover the ML estimator. See
    # Study_Design/main.tex Eq. (eb-gradient-prior).
    EB_PRIOR_TAU0_SQ = 10.0
    EB_PRIOR_NU0 = 2.0

    # ---- Deterministic sampling (reproducibility) ----
    # When the empirical_bayes algorithm is active, all randomness is consumed
    # from a pre-sampled "stream" of standard normals + uniforms stored on
    # disk. Generate one with `flask init-buffer --output <path> --seed <s>`
    # before the study starts, then point SAMPLE_BUFFER_PATH at the file.
    SAMPLE_BUFFER_PATH = "buffers/study_buffer.npz"
    # If the file does not exist on app startup, one will be auto-generated
    # using the seed and sizes below. Use `flask init-buffer` for finer
    # control (e.g. study-specific seeds).
    SAMPLE_BUFFER_AUTO_INIT = True
    SAMPLE_BUFFER_SEED = 20260421
    # Sizing for a typical full ADAPTS-HCT trial (25 dyads × 14 weeks).
    # Per-action consumption is ~ phi_dim normals (closed-form action prob =
    # zero MC samples). Per-update consumption is ~ phi_dim normals per
    # (agent, dyad). Defaults are generous — see RL_API_Simulation_Design.md
    # for a back-of-envelope budget.
    SAMPLE_BUFFER_NORMALS = 5_000_000
    SAMPLE_BUFFER_UNIFORMS = 10_000

class DevelopmentConfig(Config):
    DEBUG = True


class TestingConfig(Config):
    TESTING = True
    BACKUP_DATABASE = False
    SAVE_UPDATE_REPRO_SNAPSHOTS = False
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"
    # Tests use an in-memory buffer (no path): each app boot gets a fresh
    # buffer generated from the seed below, so tests don't accumulate
    # cross-run cursor state.
    SAMPLE_BUFFER_PATH = None
    SAMPLE_BUFFER_AUTO_INIT = True
    SAMPLE_BUFFER_SEED = 7
    SAMPLE_BUFFER_NORMALS = 500_000
    SAMPLE_BUFFER_UNIFORMS = 5_000


class ProductionConfig(Config):
    DEBUG = False
