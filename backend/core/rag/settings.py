# backend/core/rag/settings.py
import os
"""
Central place for model and Ollama configuration.

We keep routing and QA/summarization models separate so we can:
- use a TINY, fast model for routing (classification)
- use a stronger but still reasonably fast model for QA/summaries
"""

# Fast, small model for ROUTING ONLY (intent + source + operation)
# Pull with: ollama pull qwen2.5:1.5b
ROUTER_MODEL = "qwen2.5:1.5b"

# Main model for QA-style answers
# Pull with: ollama pull qwen2.5:7b
DEFAULT_QUERY_MODEL = "qwen2.5:3b"

# Model used for batch summarization of many chunks
DEFAULT_BATCH_MODEL = "qwen2.5:3b"
# Model used for the final merged summary (can be same as query model)
DEFAULT_FINAL_MODEL = "qwen2.5:1.5b"
# Ollama HTTP settings
OLLAMA_API_URL = "http://localhost:11434/api/generate"

# Be conservative with timeouts so you see errors instead of hanging forever.
# You can increase this if you run on a slower machine.
OLLAMA_TIMEOUT_SECONDS = 90

# -------------------------------------------------------------------
#  Groq routing configuration (optional)
# -------------------------------------------------------------------

# Set USE_GROQ_ROUTER=1 in your environment to enable Groq for routing:
#   export USE_GROQ_ROUTER=1
USE_GROQ_ROUTER = os.getenv("USE_GROQ_ROUTER", "0") == "1"

# Your Groq API key (set this in your shell environment, do NOT hard-code)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Default Groq model for routing. Llama 3 8B is fast and strong for classification.
GROQ_ROUTER_MODEL = os.getenv("GROQ_ROUTER_MODEL", "llama3-70b-8192")

# Groq's OpenAI-compatible endpoint
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"