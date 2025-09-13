import pytest
import asyncio
from httpx import AsyncClient
from sqlalchemy_utils import create_database, drop_database, database_exists
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from app.main import app
from app.core.database import Base, get_db
from app.core.config import settings

TEST_DATABASE_URL = settings.DATABASE_URL.replace(
    "_prod", "_test"
) if "prod" in settings.DATABASE_URL else settings.DATABASE_URL + "_test"

engine_test = create_async_engine(TEST_DATABASE_URL, future=True, echo=False)
TestingSessionLocal = sessionmaker(
    engine_test, class_=AsyncSession, expire_on_commit=False
)

async def override_get_db():
    async with TestingSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

@pytest.fixture(scope="session")
def event_loop():
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session", autouse=True)
async def setup_test_database():
    sync_db_url = TEST_DATABASE_URL.replace("postgresql+asyncpg", "postgresql")
    
    if not database_exists(sync_db_url):
        create_database(sync_db_url)
    
    async with engine_test.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    
    app.dependency_overrides[get_db] = override_get_db
    
    yield
    
    async with engine_test.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    
    await engine_test.dispose()
    
    if database_exists(sync_db_url):
        drop_database(sync_db_url)

@pytest.fixture(scope="function")
async def db_session():
    async with TestingSessionLocal() as session:
        try:
            yield session
        finally:
            await session.rollback()
            await session.close()

@pytest.fixture(scope="function")
async def client():
    async with AsyncClient(app=app, base_url="http://test") as c:
        yield c

@pytest.fixture
def auth_headers():
    if settings.REQUIRE_AUTH and hasattr(settings, 'API_KEY'):
        return {"Authorization": f"Bearer {settings.API_KEY}"}
    return {}

@pytest.fixture
def invalid_auth_headers():
    return {"Authorization": "Bearer invalid_key_12345"}