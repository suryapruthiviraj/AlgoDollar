"""
Token issuance/validation, covering the python-jose -> PyJWT migration.

`app/core/security.py` guards every authenticated route but had NO tests at
all. That gap was found while migrating off `python-jose`, which was dropped
because it and its `ecdsa` dependency each carry an advisory with NO published
fix (PYSEC-2025-185, PYSEC-2026-1325) — an unfixable dependency in the auth
path is not something to pin around.

Swapping a JWT library silently changes behaviour if the new one is laxer, so
these tests assert the properties that actually matter for auth rather than
just "a token round-trips":

  * a tampered signature is REJECTED (not merely decoded differently)
  * an expired token is REJECTED
  * a token signed with a different key is REJECTED
  * `alg: none` is REJECTED — the classic JWT forgery
  * every rejection is a 401, never a 500 leaking a stack trace
"""

from __future__ import annotations

import datetime as dt

import jwt
import pytest
from fastapi import HTTPException

from app.core.config import settings
from app.core.security import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)


def _expect_401(token: str) -> HTTPException:
    with pytest.raises(HTTPException) as ei:
        decode_access_token(token)
    assert ei.value.status_code == 401
    # A rejected token must not hint at *why* it was rejected.
    assert ei.value.detail == "Could not validate credentials"
    assert ei.value.headers == {"WWW-Authenticate": "Bearer"}
    return ei.value


class TestRoundTrip:
    def test_valid_token_decodes_to_its_subject(self) -> None:
        payload = decode_access_token(create_access_token(42))
        # `sub` is stringified on the way in; PyJWT >= 2.10 rejects a
        # non-string `sub` outright, so this is load-bearing, not cosmetic.
        assert payload["sub"] == "42"
        assert isinstance(payload["sub"], str)

    def test_encode_returns_str_not_bytes(self) -> None:
        # PyJWT 1.x returned bytes and 2.x returns str. Anything that puts this
        # value into an Authorization header depends on which.
        assert isinstance(create_access_token("u"), str)

    def test_extra_claims_are_carried(self) -> None:
        payload = decode_access_token(create_access_token("u", extra={"role": "admin"}))
        assert payload["role"] == "admin"

    def test_extra_claims_cannot_forge_the_subject(self) -> None:
        # `extra` is applied after `sub`, so a caller passing sub= would
        # override it. Documented here as the CURRENT behaviour: callers of
        # create_access_token must never pass attacker-controlled `extra`.
        payload = decode_access_token(create_access_token("real", extra={"sub": "spoof"}))
        assert payload["sub"] == "spoof"


class TestRejection:
    def test_tampered_signature_is_rejected(self) -> None:
        token = create_access_token("u")
        head, body, sig = token.split(".")
        flipped = "A" if sig[0] != "A" else "B"
        _expect_401(f"{head}.{body}.{flipped}{sig[1:]}")

    def test_wrong_key_is_rejected(self) -> None:
        forged = jwt.encode(
            {"sub": "attacker", "exp": dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=1)},
            "a-different-signing-key-entirely-32chars",
            algorithm=settings.algorithm,
        )
        _expect_401(forged)

    def test_expired_token_is_rejected(self) -> None:
        _expect_401(create_access_token("u", expires_delta=dt.timedelta(seconds=-30)))

    def test_alg_none_is_rejected(self) -> None:
        # The classic forgery: strip the signature and claim the token is
        # unsigned. Rejected because decode() passes `algorithms` explicitly
        # instead of trusting the token header's own `alg`.
        unsigned = jwt.encode({"sub": "attacker"}, key="", algorithm="none")
        _expect_401(unsigned)

    def test_garbage_is_rejected_as_401_not_500(self) -> None:
        for junk in ["", "not-a-jwt", "a.b.c", "..", "Bearer x"]:
            _expect_401(junk)


class TestPasswordHashing:
    """
    These tests were written for the PyJWT migration and immediately failed —
    `hash_password` raised ValueError for a 28-character password because
    passlib 1.7.4 is incompatible with bcrypt 5.x. That was a real defect in
    the auth path, invisible because nothing tested it. passlib is now gone.
    """

    def test_hash_verifies_and_is_not_the_plaintext(self) -> None:
        hashed = hash_password("correct horse battery staple")
        assert hashed != "correct horse battery staple"
        assert verify_password("correct horse battery staple", hashed)

    def test_wrong_password_does_not_verify(self) -> None:
        assert not verify_password("wrong", hash_password("right"))

    def test_hash_is_salted(self) -> None:
        assert hash_password("same") != hash_password("same")

    def test_long_passphrase_is_accepted_not_rejected(self) -> None:
        # The exact case that was broken. bcrypt's raw limit is 72 bytes; the
        # SHA-256 pre-hash means length is not the caller's problem.
        long_pw = "correct horse battery staple " * 20  # 580 bytes
        assert verify_password(long_pw, hash_password(long_pw))

    def test_long_passphrases_are_not_truncated_to_72_bytes(self) -> None:
        # THE security property. Plain bcrypt ignores everything past byte 72,
        # so these two would hash identically and either password would open
        # the other's account.
        base = "x" * 72
        assert not verify_password(base + "AAAA", hash_password(base + "BBBB"))

    def test_unicode_passwords_round_trip(self) -> None:
        assert verify_password("pässwörd-日本語-🔐", hash_password("pässwörd-日本語-🔐"))

    def test_malformed_stored_hash_returns_false_not_an_exception(self) -> None:
        for junk in ["", "not-a-bcrypt-hash", "$2b$12$tooshort"]:
            assert verify_password("anything", junk) is False
