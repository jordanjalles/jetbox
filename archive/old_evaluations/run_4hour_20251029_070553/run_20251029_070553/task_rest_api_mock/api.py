from flask import Flask, request, jsonify

app = Flask(__name__)

# In-memory storage for users
users = []

@app.route('/users', methods=['GET'])
def get_users():
    """Return the list of users."""
    return jsonify(users), 200

@app.route('/users', methods=['POST'])
def create_user():
    """Create a new user with JSON payload containing at least a 'name' field."""
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400
    data = request.get_json()
    if 'name' not in data:
        return jsonify({'error': "Missing 'name' field"}), 400
    # Simple user representation
    user = {
        'id': len(users) + 1,
        'name': data['name']
    }
    users.append(user)
    return jsonify(user), 201

if __name__ == '__main__':
    app.run(debug=True)
