from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import selectinload
from passlib.context import CryptContext

from src.auth.models import Base, Role, User
from src.common.exceptions import AuthenticationError
from src.common.logging import get_logger
from src.config import settings

logger = get_logger(__name__)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

engine = create_async_engine(settings.database_url, echo=False)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def init_db() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Seed default roles
    async with async_session() as session:
        ib_roles = {
            "admin": "Senior management with full access",
            "trading": "Front office trading desks",
            "risk": "Risk management and analytics",
            "compliance": "Compliance and legal",
            "research": "Equity and fixed income research (Chinese Wall restricted)",
            "wealth_management": "Wealth management and client advisory",
            "operations": "Back office operations and settlement",
            "auditor": "External audit and regulatory examination",
            "viewer": "Read-only access to public filings",
        }
        for role_name, description in ib_roles.items():
            result = await session.execute(select(Role).where(Role.name == role_name))
            if result.scalar_one_or_none() is None:
                session.add(Role(name=role_name, description=description))
        await session.commit()

    logger.info("database_initialized")


async def create_user(username: str, password: str, role_names: list[str]) -> User:
    async with async_session() as session:
        # Check if user exists
        result = await session.execute(select(User).where(User.username == username))
        if result.scalar_one_or_none() is not None:
            raise AuthenticationError(f"User '{username}' already exists")

        # Get roles
        roles = []
        for role_name in role_names:
            result = await session.execute(select(Role).where(Role.name == role_name))
            role = result.scalar_one_or_none()
            if role is None:
                raise AuthenticationError(f"Role '{role_name}' does not exist")
            roles.append(role)

        user = User(
            username=username,
            password_hash=pwd_context.hash(password),
            roles=roles,
        )
        session.add(user)
        await session.commit()
        await session.refresh(user)

        logger.info("user_created", username=username, roles=role_names)
        return user


async def authenticate_user(username: str, password: str) -> User:
    async with async_session() as session:
        result = await session.execute(
            select(User).options(selectinload(User.roles)).where(User.username == username)
        )
        user = result.scalar_one_or_none()

        if user is None or not pwd_context.verify(password, user.password_hash):
            raise AuthenticationError("Invalid username or password")

        return user


async def get_user_by_id(user_id: int) -> User | None:
    async with async_session() as session:
        result = await session.execute(
            select(User).options(selectinload(User.roles)).where(User.id == user_id)
        )
        return result.scalar_one_or_none()


async def list_users() -> list[User]:
    async with async_session() as session:
        result = await session.execute(
            select(User).options(selectinload(User.roles))
        )
        return list(result.scalars().all())
