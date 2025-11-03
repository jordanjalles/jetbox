# models.py
# Import db from app module
from app import db
from datetime import datetime
from flask_login import UserMixin

class User(UserMixin, db.Model):
    """Simple user model."""
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    posts = db.relationship("Post", backref="author", lazy="dynamic")
    comments = db.relationship("Comment", backref="author", lazy="dynamic")

    def __repr__(self):
        return f"<User {self.username}>"

class Post(db.Model):
    """Blog post model."""
    __tablename__ = "posts"

    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    body = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)

    comments = db.relationship("Comment", backref="post", lazy="dynamic")

    def __repr__(self):
        return f"<Post {self.title}>"

class Comment(db.Model):
    """Comment on a post."""
    __tablename__ = "comments"

    id = db.Column(db.Integer, primary_key=True)
    body = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    post_id = db.Column(db.Integer, db.ForeignKey("posts.id"), nullable=False)

    def __repr__(self):
        return f"<Comment {self.id} by {self.user_id}>"
