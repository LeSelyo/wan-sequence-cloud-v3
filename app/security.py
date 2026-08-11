from __future__ import annotations

import hmac
import os

from fastapi import Header, HTTPException


async def require_api_token(authorization: str | None = Header(default=None)) -> None:
    expected = os.getenv("API_TOKEN")
    if not expected:
        return
    scheme, separator, supplied = (authorization or "").partition(" ")
    if separator != " " or scheme.lower() != "bearer" or not hmac.compare_digest(supplied, expected):
        raise HTTPException(status_code=401, detail="invalid or missing bearer token")

