# Configuration for Flask application
import os

class Config:
    """Base configuration for the Flask app.

    Uses SQLite for simplicity. The database file will be created in the
    project root as ``app.db``.
    """

    # Secret key for session and CSRF protection
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key")

    # SQLite database URI
    SQLALCHEMY_DATABASE_URI = "sqlite:///app.db"
    # Disable the event system to save resources
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    # Enable CSRF protection via Flask-WTF
    WTF_CSRF_ENABLED = True
