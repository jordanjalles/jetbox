import asyncio
import socketio
from .ot import OTHandler
from .db import DocumentDB
from .redis_presence import PresenceManager

# Async Socket.IO server
sio = socketio.AsyncServer(async_mode='asgi')
app = socketio.ASGIApp(sio)

# In-memory document state
class DocumentState:
    def __init__(self, doc_id, content=""):
        self.doc_id = doc_id
        self.content = content
        self.version = 0
        self.pending_ops = []  # list of ops waiting to be applied

# Global state: doc_id -> DocumentState
documents = {}

# Initialize DB and Presence
db = DocumentDB()
presence = PresenceManager()

async def get_document_state(doc_id):
    if doc_id not in documents:
        # Load from DB
        content = await db.load_document(doc_id)
        documents[doc_id] = DocumentState(doc_id, content)
    return documents[doc_id]

@sio.event
async def connect(sid, environ):
    print(f"Client connected: {sid}")

@sio.event
async def disconnect(sid):
    print(f"Client disconnected: {sid}")
    # Remove from all documents presence
    for doc_id in list(documents.keys()):
        await presence.remove_user(doc_id, sid)

@sio.event
async def join_document(sid, data):
    """data: {doc_id: str, user_id: str}"""
    doc_id = data.get("doc_id")
    user_id = data.get("user_id")
    if not doc_id or not user_id:
        return
    await presence.add_user(doc_id, user_id)
    await sio.enter_room(sid, doc_id)
    state = await get_document_state(doc_id)
    # Send current document state
    await sio.emit("document_state", {"content": state.content, "version": state.version}, room=sid)

@sio.event
async def leave_document(sid, data):
    doc_id = data.get("doc_id")
    user_id = data.get("user_id")
    if not doc_id or not user_id:
        return
    await presence.remove_user(doc_id, user_id)
    await sio.leave_room(sid, doc_id)

@sio.event
async def edit_operation(sid, data):
    """data: {doc_id, op, version}"""
    doc_id = data.get("doc_id")
    op = data.get("op")
    client_version = data.get("version")
    if not doc_id or not op or client_version is None:
        return
    state = await get_document_state(doc_id)
    # Transform operation against pending ops
    transformed_op = OTHandler.transform(op, state.pending_ops, client_version, state.version)
    # Apply to state
    state.content = OTHandler.apply(state.content, transformed_op)
    state.version += 1
    # Append to pending ops
    state.pending_ops.append(transformed_op)
    # Persist to DB
    await db.save_document(doc_id, state.content)
    # Broadcast to others
    await sio.emit("remote_operation", {"op": transformed_op, "version": state.version}, room=doc_id, skip_sid=sid)

# Periodically clear pending ops older than a threshold
async def cleanup_pending_ops():
    while True:
        await asyncio.sleep(60)
        for state in documents.values():
            state.pending_ops = []

# Start cleanup task
asyncio.create_task(cleanup_pending_ops())

# Expose app
__all__ = ["app", "sio"]
# Initialize DB and Presence on startup
async def startup():
    await db.init()
    await presence.init()

asyncio.create_task(startup())
