import sys
from pathlib import Path
from starlette.requests import HTTPConnection

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

import auth  # noqa: E402


def test_get_user_from_token_returns_user(monkeypatch) -> None:
    expected_user = auth.UserInDB(username="alice", hashed_password="x", disabled=False, must_change_password=False)
    monkeypatch.setattr(auth, "get_user", lambda username: expected_user if username == "alice" else None)

    token = auth.create_access_token({"sub": "alice"})

    assert auth.get_user_from_token(token) == expected_user


def test_get_user_from_token_rejects_invalid_token() -> None:
    assert auth.get_user_from_token("not-a-jwt") is None


def test_get_current_connection_user_accepts_bearer_header(monkeypatch) -> None:
    expected_user = auth.UserInDB(username="alice", hashed_password="x", disabled=False, must_change_password=False)
    monkeypatch.setattr(auth, "get_user", lambda username: expected_user if username == "alice" else None)

    token = auth.create_access_token({"sub": "alice"})
    connection = HTTPConnection(
        {
            "type": "http",
            "headers": [(b"authorization", f"Bearer {token}".encode("utf-8"))],
            "query_string": b"",
        }
    )

    import asyncio

    assert asyncio.run(auth.get_current_connection_user(connection)) == expected_user


def test_get_current_connection_user_accepts_query_token(monkeypatch) -> None:
    expected_user = auth.UserInDB(username="alice", hashed_password="x", disabled=False, must_change_password=False)
    monkeypatch.setattr(auth, "get_user", lambda username: expected_user if username == "alice" else None)

    token = auth.create_access_token({"sub": "alice"})
    connection = HTTPConnection(
        {
            "type": "websocket",
            "headers": [],
            "query_string": f"token={token}".encode("utf-8"),
        }
    )

    import asyncio

    assert asyncio.run(auth.get_current_connection_user(connection)) == expected_user
