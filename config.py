import os


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
    SQLALCHEMY_DATABASE_URI = "postgresql://zipingxu@localhost:5432/justin_rl_db"
    SQLALCHEMY_ENGINE_OPTIONS = {
        "pool_recycle": 3600,  # Recycle connections after 3600 seconds (1 hour)
    }
    RL_ALGORITHM_SEED = 42  # Seed for RL Algorithm random state

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

class DevelopmentConfig(Config):
    DEBUG = True


class TestingConfig(Config):
    TESTING = True
    BACKUP_DATABASE = False
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"


class ProductionConfig(Config):
    DEBUG = False
