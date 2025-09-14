# users/service.py
from passlib.context import CryptContext
from datetime import datetime, timedelta
import uuid
import jwt

from app.models.user import User
from app.database import async_session_maker
from sqlalchemy.future import select
from sqlalchemy.exc import IntegrityError

from app.core.config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class UserService:
    @staticmethod
    def hash_password(password: str) -> str:
        return pwd_context.hash(password)

    @staticmethod
    def verify_password(password: str, hashed: str) -> bool:
        return pwd_context.verify(password, hashed)

    @staticmethod
    def create_access_token(data: dict, expires_delta: int = 60):
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(minutes=expires_delta)
        to_encode.update({"exp": expire})
        return jwt.encode(to_encode, settings.SECRET_KEY, algorithm="HS256")

    @staticmethod
    async def create_user(username: str, email: str, password: str):
        async with async_session_maker() as session:
            user = User(
                id=uuid.uuid4(),
                username=username,
                email=email,
                hashed_password=UserService.hash_password(password),
                is_active=True,
            )
            session.add(user)
            try:
                await session.commit()
                await session.refresh(user)
                return user
            except IntegrityError:
                await session.rollback()
                raise ValueError("User with this email or username already exists")

    @staticmethod
    async def authenticate_user(email: str, password: str):
        async with async_session_maker() as session:
            result = await session.execute(select(User).where(User.email == email))
            user = result.scalars().first()
            if not user or not UserService.verify_password(password, user.hashed_password):
                return None
            return user
