import asyncpg

class DocumentDB:
    def __init__(self, dsn=None):
        self.dsn = dsn or "postgresql://postgres:postgres@localhost:5432/collab"
        self.pool = None

    async def init(self):
        self.pool = await asyncpg.create_pool(dsn=self.dsn)

    async def load_document(self, doc_id):
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("SELECT content FROM documents WHERE id=$1", doc_id)
            return row["content"] if row else ""

    async def save_document(self, doc_id, content):
        async with self.pool.acquire() as conn:
            await conn.execute("INSERT INTO documents(id, content) VALUES($1, $2) ON CONFLICT(id) DO UPDATE SET content=$2", doc_id, content)
