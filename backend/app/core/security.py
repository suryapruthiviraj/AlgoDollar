from __future__ import annotations

import base64
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import bcrypt
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jwt import PyJWTError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.database.session import get_async_session

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/token")


# --------------------------------------------------------------------------- #
#  Password hashing                                                            #
# --------------------------------------------------------------------------- #
#
# `bcrypt` is used directly. This previously went through `passlib`'s
# CryptContext, which was BROKEN: passlib 1.7.4 (last released 2020, effectively
# unmaintained) reads `bcrypt.__about__.__version__`, an attribute bcrypt
# removed in 4.1. With bcrypt 5.x the failed version probe puts passlib on a
# path where EVERY hash() call raises
#     ValueError: password cannot be longer than 72 bytes
# regardless of the actual password length. A 28-character password failed.
# So password hashing did not work at all, and no test covered it.
#
# bcrypt's 72-byte input limit is real, not a passlib artifact. Two ways to
# handle it are wrong:
#   * truncating to 72 bytes makes every password sharing a 72-byte prefix
#     equivalent — a silent authentication weakness;
#   * rejecting long input turns a strong passphrase into an error.
# So input is SHA-256'd first and base64'd, mapping any length to a fixed
# 44 bytes. base64 (not raw digest) because a raw digest can contain NUL,
# which truncates the C string bcrypt hashes. This is what Django and Passlib's
# own `bcrypt_sha256` do.

def _prehash(plain: str) -> bytes:
    return base64.b64encode(hashlib.sha256(plain.encode("utf-8")).digest())


def hash_password(plain: str) -> str:
    return bcrypt.hashpw(_prehash(plain), bcrypt.gensalt()).decode("ascii")


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(_prehash(plain), hashed.encode("ascii"))
    except (ValueError, TypeError):
        # A malformed or truncated stored hash means "cannot authenticate",
        # never an unhandled 500 — and never a pass.
        return False


def create_access_token(
    subject: str | int,
    extra: Optional[dict[str, Any]] = None,
    expires_delta: Optional[timedelta] = None,
) -> str:
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=settings.access_token_expire_minutes)
    )
    payload: dict[str, Any] = {
        "sub": str(subject),
        "exp": expire,
        "iat": datetime.now(timezone.utc),
    }
    if extra:
        payload.update(extra)
    return jwt.encode(payload, settings.secret_key, algorithm=settings.algorithm)


def decode_access_token(token: str) -> dict[str, Any]:
    try:
        # `algorithms` is passed explicitly and is never derived from the
        # token's own header — accepting the header's `alg` is how the classic
        # "alg: none" / HS256-signed-with-the-RSA-public-key forgeries work.
        return jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
    # PyJWTError is the base of every PyJWT failure: bad signature, expired,
    # malformed, wrong algorithm. Catching the base class means a validation
    # failure can never fall through this handler and reach the caller as a
    # 500 that leaks a stack trace.
    except PyJWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    session: AsyncSession = Depends(get_async_session),
):
    from sqlalchemy import select

    from app.database.models import User

    payload = decode_access_token(token)
    user_id: str | None = payload.get("sub")
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
            headers={"WWW-Authenticate": "Bearer"},
        )

    result = await session.execute(select(User).where(User.id == int(user_id)))
    user = result.scalar_one_or_none()
    if user is None or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


async def get_current_active_user(
    current_user=Depends(get_current_user),
):
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user
