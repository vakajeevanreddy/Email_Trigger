from pydantic import BaseModel
from typing import Optional

class Observation(BaseModel):
    email_id: int
    email_text: str
    step_count: int

class Action(BaseModel):
    category: Optional[str] = None
    action_type: Optional[str] = None
    response: Optional[str] = None

class Reward(BaseModel):
    value: float
    reason: str