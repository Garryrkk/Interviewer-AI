import os
from typing import Any
import asyncio

# PostgreSQL
import asyncpg

# MySQL
import aiomysql

# Redis
import aioredis

# MongoDB GridFS
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorGridFSBucket

# Vector DB example (Chroma)
from chromadb import AsyncClient


# ---------------------------
# 1️⃣ SQL Database
# ---------------------------
async def create_database_connection() -> Any:
    db_type = os.getenv("DB_TYPE", "postgresql").lower()

    if db_type == "postgresql":
        dsn = os.getenv(
            "DATABASE_URL",
            "postgresql://user:password@localhost:5432/mydb"
        )
        pool = await asyncpg.create_pool(dsn)
        return pool

    elif db_type == "mysql":
        pool = await aiomysql.create_pool(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", 3306)),
            user=os.getenv("DB_USER", "user"),
            password=os.getenv("DB_PASSWORD", "password"),
            db=os.getenv("DB_NAME", "mydb"),
            minsize=int(os.getenv("MYSQL_POOL_MIN", 1)),
            maxsize=int(os.getenv("MYSQL_POOL_MAX", 10)),
            autocommit=True,
            charset="utf8mb4"
        )
        return pool

    else:
        raise ValueError(f"Unsupported DB_TYPE: {db_type}")


# ---------------------------
# 2️⃣ Redis Cache
# ---------------------------
UPSTASH_REDIS_URL = os.getenv(
    "REDIS_URL",
    "redis://default:ASb1AAImcDIxYjZmNGFjNTkyNTQ0ZmJhYmE0NTdhMTQzMDE3OGY1MHAyOTk3Mw@divine-pelican-9973.upstash.io:6379"
)

async def create_redis_connection() -> aioredis.Redis:
    redis = aioredis.from_url(
        UPSTASH_REDIS_URL,
        encoding="utf-8",
        decode_responses=True
    )
    await redis.ping()
    print("✅ Connected to Redis!")
    return redis


# ---------------------------
# 3️⃣ File Storage (MongoDB + GridFS)
# ---------------------------
MONGO_URL = os.getenv(
    "MONGO_URL",
    "mongodb+srv://garimakalra471:happy12345678@cluster0.zyq6f.mongodb.net/mydatabase"
)

async def create_file_storage_connection() -> AsyncIOMotorGridFSBucket:
    client = AsyncIOMotorClient(MONGO_URL)
    db = client.get_database()  # uses the DB in URL
    fs_bucket = AsyncIOMotorGridFSBucket(db, bucket_name="files")
    print("✅ Connected to MongoDB GridFS!")
    return fs_bucket


# ---------------------------
# 4️⃣ Vector Database (ChromaDB example)
# ---------------------------
async def create_vector_database_connection() -> Any:
    client = AsyncClient()
    await client.start()
    print("✅ Connected to Vector Database!")
    return client


# ---------------------------
# Example Usage
# ---------------------------
async def main():
    db_pool = await create_database_connection()
    redis = await create_redis_connection()
    fs_bucket = await create_file_storage_connection()
    vector_db = await create_vector_database_connection()

    # Test Redis
    await redis.set("foo", "bar")
    val = await redis.get("foo")
    print("Redis value:", val)

    await redis.close()
    if db_pool:  # close MySQL pool
        db_pool.close()
        await db_pool.wait_closed()
    if vector_db:
        await vector_db.stop()

# Run the example
if __name__ == "__main__":
    asyncio.run(main())
