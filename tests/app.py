# app.py
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import LoginManager

# Import Config from config.py
from config import Config

# Create the SQLAlchemy db instance
# It will be bound to the app in create_app

db = SQLAlchemy()

# Create the Flask-Migrate instance
migrate = Migrate()

# Login manager
login_manager = LoginManager()
login_manager.login_view = 'auth.login'


def create_app(config_object=Config):
    """Application factory.

    Parameters
    ----------
    config_object : object
        A configuration object or class. Defaults to :class:`Config`.
    """
    app = Flask(__name__)
    app.config.from_object(config_object)

    # Initialize extensions
    db.init_app(app)
    migrate.init_app(app, db)
    login_manager.init_app(app)

    # Import models so that they are registered with SQLAlchemy
    import models

    # Register blueprints
    from auth import bp as auth_bp
    from posts import bp as posts_bp
    from comments import bp as comments_bp
    app.register_blueprint(auth_bp)
    app.register_blueprint(posts_bp)
    app.register_blueprint(comments_bp)

    # User loader for Flask-Login
    @login_manager.user_loader
    def load_user(user_id):
        from models import User
        return User.query.get(int(user_id))

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
