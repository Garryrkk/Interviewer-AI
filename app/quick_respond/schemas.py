# schemas.py
from typing import List, Optional
from pydantic import BaseModel, Field

class QuickRequest(BaseModel):
    text: str = Field(..., description="User input text to generate a quick response for")
    simplify: Optional[bool] = Field(False, description="If true, produce a simpler/lower-reading-level reply")
    max_tokens: Optional[int] = Field(80, description="Optional max tokens / length hint for the model")

class QuickResponse(BaseModel):
    reply: str = Field(..., description="AI quick reply text")
    suggestions: Optional[List[str]] = Field(None, description="Optional short tips or variants")
    source: str = Field("mock", description="Source used (mock | openai | custom)")
