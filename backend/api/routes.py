from fastapi import APIRouter, HTTPException
from backend.api.models import PromptRequest, PromptResponse
from backend.core.orchestrator import handle_prompt_async  # async version

router = APIRouter()

@router.post("/prompt", response_model=PromptResponse)
async def prompt_endpoint(req: PromptRequest):
    try:
        result = await handle_prompt_async(req.message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return PromptResponse(
        answer=result["answer"],
        routing=result.get("routing", {}),
        sources=result.get("sources", []),
    )