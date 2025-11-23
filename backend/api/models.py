# backend/api/models.py

from pydantic import BaseModel
from typing import List, Optional, Dict, Any


class PromptRequest(BaseModel):
    message: str

class PromptResponse(BaseModel):
    answer: str
    routing: Dict[str, Any]
    sources: Optional[List[Dict[str, Any]]] = None