# tests/test_app.py
import os
import tempfile
import pytest
from app import create_app, db
from models import User, Post, Comment

@pytest.fixture
def app():
    # Use in-memory SQLite for tests
    class TestConfig:
        SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
        SQLALCHEMY_TRACK_MODIFICATIONS = False
        TESTING = True
        SECRET_KEY = 'test'
    app = create_app(TestConfig)
    with app.app_context():
        db.create_all()
    yield app
    # teardown
    with app.app_context():
        db.drop_all()

@pytest.fixture
def client(app):
    return app.test_client()

def register(client, username, email, password):
    return client.post('/auth/register', data=dict(
        username=username,
        email=email,
        password=password
    ), follow_redirects=True)

def login(client, username, password):
    return client.post('/auth/login', data=dict(
        username=username,
        password=password
    ), follow_redirects=True)

def test_user_registration_and_login(client):
    # Register
    rv = register(client, 'alice', 'alice@example.com', 'secret')
    assert b'Registration successful' in rv.data
    # Login
    rv = login(client, 'alice', 'secret')
    assert b'Logged in successfully' in rv.data

def test_post_creation(client):
    register(client, 'bob', 'bob@example.com', 'pass')
    login(client, 'bob', 'pass')
    rv = client.post('/posts/create', data=dict(
        title='Test Post',
        body='This is a test.'
    ), follow_redirects=True)
    assert b'Post created' in rv.data
    assert b'Test Post' in rv.data

def test_comment_creation(client):
    register(client, 'carol', 'carol@example.com', 'pwd')
    login(client, 'carol', 'pwd')
    # Create a post first
    client.post('/posts/create', data=dict(
        title='Post for comment',
        body='Content'
    ), follow_redirects=True)
    # Get post id from database
    with client.application.app_context():
        post = Post.query.first()
        post_id = post.id
    # Add comment
    rv = client.post('/comments/create', data=dict(
        post_id=post_id,
        body='Nice post!'
    ), follow_redirects=True)
    assert b'Comment added' in rv.data
    assert b'Nice post!' in rv.data

