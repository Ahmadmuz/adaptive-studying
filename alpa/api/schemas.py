from __future__ import annotations

from pydantic import BaseModel, Field


class UserIn(BaseModel):
    name: str = Field(min_length=1, max_length=120)


class UserOut(BaseModel):
    id: int
    name: str


class SessionOut(BaseModel):
    id: int
    user_id: int


class AttemptIn(BaseModel):
    intervention_id: int
    item_id: int
    choice_index: int | None = Field(default=None, ge=0)
    response_text: str | None = Field(default=None, max_length=2000)
    latency_ms: float | None = Field(default=None, ge=0)
    hint_level: int = Field(default=0, ge=0)
    paste_detected: bool = False
    confidence: float | None = Field(default=None, ge=0, le=1)
