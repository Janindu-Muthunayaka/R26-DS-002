"""
FastAPI Server for Layer 7 Personalization (Component 4).

This runs in its own environment because it requires `transformers==5.7.0` and 
`numpy==2.4.4`, which are incompatible with the main reader pipeline.

Usage:
    python layers/l7_personalization/server.py --port 8101
"""

import argparse
import sys
from pathlib import Path

# Add system/ to sys.path so we can import modules
_SYSTEM = Path(__file__).resolve().parent.parent.parent
if str(_SYSTEM) not in sys.path:
    sys.path.insert(0, str(_SYSTEM))

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from layers.l7_personalization.personalization.main_flow import handle_voice_command

app = FastAPI(title="L7 Personalization Service")

class InterpretRequest(BaseModel):
    text: str
    user_id: Optional[str] = "user_001"
    retrieved_chunk_id: Optional[str] = None

@app.post("/interpret")
def interpret(req: InterpretRequest):
    """
    Accepts Sinhala text, translates it, detects the intent, and applies 
    personalization styling based on user history.
    """
    try:
        if not req.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty.")
            
        result = handle_voice_command(req.text, req.user_id)
        
        # The main pipeline (l0_voice) only needs the final prompt dictionary
        final_prompt = result.get("final_prompt", {})
        
        # Ensure we add the retrieved_chunk_id back into the response if provided
        if req.retrieved_chunk_id:
            final_prompt["retrieved_chunk_id"] = req.retrieved_chunk_id
            
        return final_prompt
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8101, help="Port to run the personalization service on")
    args = parser.parse_args()
    
    print(f"Starting L7 Personalization service on port {args.port}...")
    uvicorn.run(app, host="127.0.0.1", port=args.port)
