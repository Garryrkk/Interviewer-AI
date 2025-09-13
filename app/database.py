# backend/database.py
import os
import logging
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import SQLAlchemyError
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

Base = declarative_base()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://postgres:password@localhost:5432/interview_ai")

engine = create_async_engine(
    DATABASE_URL,
    echo=False,          
    future=True,
)

AsyncSessionLocal = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False
)


async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except SQLAlchemyError as e:
            logger.error(f"Database session error: {e}")
            raise
        finally:
            await session.close()


async def init_db():
   
    try:
        async with engine.begin() as conn:
            logger.info("Creating database tables if not exist...")
            await conn.run_sync(Base.metadata.create_all)
            logger.info("Database tables ready!")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


async def close_db():

    try:
        await engine.dispose()
        logger.info("Database engine disposed successfully")
    except Exception as e:
        logger.error(f"Error disposing database engine: {e}")

