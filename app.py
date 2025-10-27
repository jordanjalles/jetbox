from flask import Flask, request, jsonify, abort

app = Flask(__name__)

# In-memory store
_items = {}
_next_id = 1

# Helper to validate JSON body
def _require_json():
    if not request.is_json:
        abort(400, description="Bad Request")
    data = request.get_json()
    if not isinstance(data, dict):
        abort(400, description="Bad Request")
    return data

@app.route('/items', methods=['POST'])
def create_item():
    global _next_id
    data = _require_json()
    name = data.get('name')
    value = data.get('value')
    if name is None or value is None:
        abort(400, description="Bad Request")
    item_id = _next_id
    _next_id += 1
    item = {'id': item_id, 'name': name, 'value': value}
    _items[item_id] = item
    return jsonify(item), 201

@app.route('/items', methods=['GET'])
def list_items():
    return jsonify(list(_items.values()))

@app.route('/items/<int:item_id>', methods=['GET'])
def get_item(item_id):
    item = _items.get(item_id)
    if not item:
        abort(404, description="Not Found")
    return jsonify(item)

@app.route('/items/<int:item_id>', methods=['PUT'])
def update_item(item_id):
    item = _items.get(item_id)
    if not item:
        abort(404, description="Not Found")
    data = _require_json()
    name = data.get('name')
    value = data.get('value')
    if name is None or value is None:
        abort(400, description="Bad Request")
    item.update({'name': name, 'value': value})
    return jsonify(item)

@app.route('/items/<int:item_id>', methods=['DELETE'])
def delete_item(item_id):
    if item_id not in _items:
        abort(404, description="Not Found")
    del _items[item_id]
    return '', 204

# Error handlers
@app.errorhandler(400)
def bad_request(e):
    return jsonify(error="Bad Request"), 400

@app.errorhandler(404)
def not_found(e):
    return jsonify(error="Not Found"), 404

# Expose internal state for tests
__all__ = ['app', '_items', '_next_id']
