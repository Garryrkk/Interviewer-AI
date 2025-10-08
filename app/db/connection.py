import os
from typing import Any
import asyncio
from redis import asyncio as redis  # ✅ new import
from redis.asyncio.client import Redis  # for type-hinting if you want it
# PostgreSQL
import asyncpg

# MySQL
import aiomysql

# MongoDB GridFS
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorGridFSBucket

# Vector DB example (Chroma)
from chromadb.config import Settings
import chromadb


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
UPSTASH_REDIS_URL = "your-upstash-url"  # e.g. "redis://default:pass@host:port"

async def create_redis_connection() -> Redis:
    r = redis.from_url(
        UPSTASH_REDIS_URL,
        encoding="utf-8",
        decode_responses=True
    )
    await r.ping()
    print("✅ Connected to Redis!")
    return r


# ---------------------------
# 3️⃣ File Storage (MongoDB + GridFS)
# ---------------------------
MONGO_URL = os.getenv(
    "MONGO_URL",
    "mongodb+srv://garimakalra471:happy12345678@cluster0.zyq6f.mongodb.net/mydatabase"
)

async def create_file_storage_connection() -> AsyncIOMotorGridFSBucket:
    client = AsyncIOMotorClient(MONGO_URL)
    db = client.get_database()
    fs_bucket = AsyncIOMotorGridFSBucket(db, bucket_name="files")
    print("✅ Connected to MongoDB GridFS!")
    return fs_bucket


# ---------------------------
# 4️⃣ Vector Database (ChromaDB example)
# ---------------------------
async def create_vector_database_connection() -> Any:
    def _make_client():
        return chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory="chroma_data"
        ))

    client = await asyncio.to_thread(_make_client)

    def _test():
        col = client.get_or_create_collection("test_collection")
        col.add(documents=["hello chroma"], ids=["1"])
        return col.count()

    count = await asyncio.to_thread(_test)
    print(f"✅ Connected to Vector Database! Docs in test_collection: {count}")
    return client


# ---------------------------
# Example Usage
# ---------------------------
async def main():
    db_pool = await create_database_connection()
    redis_conn = await create_redis_connection()
    fs_bucket = await create_file_storage_connection()
    vector_db = await create_vector_database_connection()

    # Test Redis
    await redis_conn.set("foo", "bar")
    val = await redis_conn.get("foo")
    print("Redis value:", val)

    await redis_conn.close()
    if db_pool:
        db_pool.close()
        await db_pool.wait_closed()
    # vector_db is a sync client; no .stop() needed

if __name__ == "__main__":
    asyncio.run(main())
