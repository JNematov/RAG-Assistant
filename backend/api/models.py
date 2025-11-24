# backend/api/models.py

from pydantic import BaseModel
from typing import List, Optional, Dict, Any

# Defines the structure for incoming API requests
class PromptRequest(BaseModel):
    message: str
# Defines the structure for outgoing API responses, ensuring consistent, validated output (e.g., serialized to JSON automatically)
class PromptResponse(BaseModel):
    answer: str
    routing: Dict[str, Any]
    sources: Optional[List[Dict[str, Any]]] = None