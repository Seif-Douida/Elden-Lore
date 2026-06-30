"""
backend/api/auth.py

User identity dependency. Today it returns a fixed dev user so the conversation
layer is user-aware from the start; later, swap the body of get_current_user to
verify a Supabase JWT from the Authorization header — no schema or endpoint
changes needed (every query already filters by this user_id).
"""

from __future__ import annotations

import uuid

# A stable dev user so conversations group consistently in development.
DEV_USER_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")


async def get_current_user() -> uuid.UUID:
    # TODO (pre-deploy): replace with Supabase JWT verification, e.g.
    #   token = request.headers["authorization"].removeprefix("Bearer ")
    #   payload = verify_supabase_jwt(token)
    #   return uuid.UUID(payload["sub"])
    return DEV_USER_ID