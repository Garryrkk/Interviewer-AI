import os
from typing import Any

# PostgreSQL
import asyncpg

# MySQL
import aiomysql

async def create_database_connection() -> Any:
    """
    Create an async database connection pool.
    Supports PostgreSQL or MySQL depending on DB_TYPE env variable.
    """
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
            autocommit=True,
        )
        return pool

    else:
        raise ValueError(f"Unsupported DB_TYPE: {db_type}")
