"""A simple Flask REST API with CRUD endpoints for a User model.

This module implements an in‑memory user store and exposes the following
endpoints:

* POST   /users          – Create a new user
* GET    /users          – List all users
* GET    /users/<int:id> – Retrieve a single user
* PUT    /users/<int:id> – Update a user
* DELETE /users/<int:id> – Delete a user

All responses are JSON and errors are returned with an appropriate HTTP
status code and a JSON body containing an ``error`` key.

The module can be run directly (``python crud_api.py``) which starts a
debug server on port 5000.
"""

from __future__ import annotations

from flask import Flask, jsonify, request, abort
from werkzeug.exceptions import HTTPException

app = Flask(__name__)

# In‑memory store: id -> user dict
users: dict[int, dict[str, str]] = {}
next_id = 1

# ---------- Helper functions ----------

def _get_next_id() -> int:
    global next_id
    current = next_id
    next_id += 1
    return current


def _validate_user_payload(data: dict) -> None:
    if not isinstance(data, dict):
        raise ValueError("Payload must be a JSON object")
    if "name" not in data or not isinstance(data["name"], str):
        raise ValueError("'name' is required and must be a string")
    if "email" not in data or not isinstance(data["email"], str):
        raise ValueError("'email' is required and must be a string")

# ---------- Error handlers ----------
@app.errorhandler(HTTPException)
def handle_http_exception(e: HTTPException):
    response = e.get_response()
    response.data = jsonify({"error": e.description}).data
    response.content_type = "application/json"
    return response

@app.errorhandler(ValueError)
def handle_value_error(e: ValueError):
    return jsonify({"error": str(e)}), 400

@app.errorhandler(404)
def handle_404(e):
    return jsonify({"error": "Not found"}), 404

# ---------- CRUD endpoints ----------
@app.route("/users", methods=["POST"])
def create_user():
    data = request.get_json(force=True)
    _validate_user_payload(data)
    user_id = _get_next_id()
    user = {"id": user_id, "name": data["name"], "email": data["email"]}
    users[user_id] = user
    return jsonify(user), 201

@app.route("/users", methods=["GET"])
def list_users():
    return jsonify(list(users.values()))

@app.route("/users/<int:user_id>", methods=["GET"])
def get_user(user_id: int):
    user = users.get(user_id)
    if not user:
        abort(404)
    return jsonify(user)

@app.route("/users/<int:user_id>", methods=["PUT"])
def update_user(user_id: int):
    user = users.get(user_id)
    if not user:
        abort(404)
    data = request.get_json(force=True)
    _validate_user_payload(data)
    user.update({"name": data["name"], "email": data["email"]})
    return jsonify(user)

@app.route("/users/<int:user_id>", methods=["DELETE"])
def delete_user(user_id: int):
    if user_id not in users:
        abort(404)
    del users[user_id]
    return jsonify({"message": "User deleted"})

# ---------- Run block ----------
if __name__ == "__main__":
    app.run(debug=True)
