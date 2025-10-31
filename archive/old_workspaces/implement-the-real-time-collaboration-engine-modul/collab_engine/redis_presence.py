import aioredis

class PresenceManager:
    def __init__(self, redis_url="redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis = None

    async def init(self):
        self.redis = await aioredis.from_url(self.redis_url, decode_responses=True)

    async def add_user(self, doc_id, user_id):
        await self.redis.sadd(f"doc:{doc_id}:users", user_id)

    async def remove_user(self, doc_id, user_id):
        await self.redis.srem(f"doc:{doc_id}:users", user_id)

    async def get_users(self, doc_id):
        return await self.redis.smembers(f"doc:{doc_id}:users")
