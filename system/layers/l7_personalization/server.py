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

# Add system/ and system/layers/l7_personalization to sys.path so we can import modules
_SYSTEM = Path(__file__).resolve().parent.parent.parent
_L7 = Path(__file__).resolve().parent
if str(_SYSTEM) not in sys.path:
    sys.path.insert(0, str(_SYSTEM))
if str(_L7) not in sys.path:
    sys.path.insert(0, str(_L7))

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any

# Import under the SAME module names main_flow uses (`personalization.*`,
# from the L7 folder on sys.path). Importing style_model as
# `layers.l7_personalization.personalization.style_model` instead loads a
# second copy with its own in-memory weights: /profile then served weights
# frozen at startup while main_flow's copy learned and saved.
try:
    from personalization.main_flow import handle_voice_command
    from personalization import style_model, users as user_registry
    from personalization.logger import db as l7_db, export_logs_to_csv
    from delivery.planner import plan_speech
    from delivery.voice_profile import get_profile as get_voice_delivery_profile
except ImportError:
    from layers.l7_personalization.personalization.main_flow import handle_voice_command
    from layers.l7_personalization.personalization import style_model, users as user_registry
    from layers.l7_personalization.personalization.logger import db as l7_db, export_logs_to_csv
    from layers.l7_personalization.delivery.planner import plan_speech
    from layers.l7_personalization.delivery.voice_profile import get_profile as get_voice_delivery_profile
get_user_summary = style_model.get_user_summary

app = FastAPI(title="L7 Personalization Service")

class InterpretRequest(BaseModel):
    text: str
    user_id: Optional[str] = "user_001"
    retrieved_chunk_id: Optional[str] = None
    dry_run: Optional[bool] = False   # decide style only: no logging / learning
    # From the main server, which holds what was last spoken this session:
    # {"has_last": bool, "next_turn": bool, "age_s": float}
    last_spoken_ctx: Optional[Dict[str, Any]] = None

class NewUserRequest(BaseModel):
    base_weights: Dict[str, float]    # Simple / Detailed / StepByStep (fractions or %)
    description: Optional[str] = ""
    name: Optional[str] = ""
    tag: Optional[str] = ""

class DeliverRequest(BaseModel):
    answer_si: str
    style: Optional[str] = "Detailed"
    user_id: Optional[str] = "user_001"
    replay: Optional[bool] = False

@app.post("/interpret")
def interpret(req: InterpretRequest):
    """
    Accepts Sinhala text, translates it, detects the intent, and applies 
    personalization styling based on user history.
    """
    try:
        if not req.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty.")
            
        result = handle_voice_command(req.text, req.user_id, retrieved_chunk_id=req.retrieved_chunk_id,
                                      dry_run=bool(req.dry_run),
                                      last_spoken_ctx=req.last_spoken_ctx)
        
        # The main pipeline (l0_voice) only needs the final prompt dictionary
        final_prompt = result.get("final_prompt", {})
        
        # Merge personalization stage details for frontend logging/diagnostics
        pers_stage = result.get("personalization_stage", {})
        final_prompt["style_source"] = pers_stage.get("style_source")
        final_prompt["user_profile"] = pers_stage.get("user_profile")
        final_prompt["learned"] = pers_stage.get("learned")
        
        # Ensure default personalization fields are present even for system commands
        final_prompt.setdefault("style_class", pers_stage.get("style_class") or "Detailed")
        final_prompt.setdefault("prompt_modifier", "Provide a thorough, in-depth explanation with full context, reasoning, and supporting detail. Do not shorten or oversimplify.")
        final_prompt.setdefault("personalization_flags", {})
        
        # Ensure speed rate (0.8 - 1.3) is always provided in the response
        user_speed = pers_stage.get("speed") or final_prompt.get("speed") or 1.0
        final_prompt["speed"] = round(float(user_speed), 2)
        final_prompt["speech_rate"] = round(float(user_speed), 2)

        # Ensure we add the retrieved_chunk_id back into the response if provided
        if req.retrieved_chunk_id:
            final_prompt["retrieved_chunk_id"] = req.retrieved_chunk_id
            
        final_prompt["sinhala_input"] = req.text

        # Attach frame if available
        if "frame" in result:
            final_prompt["frame"] = result["frame"]

        return final_prompt

    except HTTPException:
        # "Text cannot be empty" is the caller's mistake (400), not a crash.
        # The catch-all below turned it into a 500 with a stack trace.
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/profile/{user_id}")
def get_profile(user_id: str):
    """
    Returns the user's personalization profile history and model weights.
    """
    try:
        return get_user_summary(user_id)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# ── User registry ─────────────────────────────────────────────────────────

def _user_view(u):
    """Registry record plus the live learned weights."""
    return {**u, "profile": get_user_summary(u["user_id"])}


def _purge_user_rows(user_id):
    """Drop a deleted user's rows from every L7 table (logs, voice profile...)."""
    from tinydb import Query
    for name in l7_db.tables():
        if name != "user_profiles":
            l7_db.table(name).remove(Query().user_id == user_id)
    style_model.delete_user_state(user_id)


def _rename_user_rows(old_id, new_id):
    """Move a renumbered user's rows in every L7 table to the new id."""
    from tinydb import Query
    for name in l7_db.tables():
        if name != "user_profiles":
            l7_db.table(name).update({"user_id": new_id}, Query().user_id == old_id)
    style_model.rename_user_state(old_id, new_id)


@app.get("/users")
def list_users():
    return {"users": [_user_view(u) for u in user_registry.list_users()]}


@app.post("/users")
def add_user(req: NewUserRequest):
    try:
        u = user_registry.add_user(req.base_weights, req.description, req.name, req.tag)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    style_model.reset_user(u["user_id"])          # start them at their base split
    return {"user": _user_view(u)}


@app.post("/users/{user_id}/delete")
def delete_user(user_id: str):
    try:
        renames = user_registry.delete_user(user_id, _purge_user_rows, _rename_user_rows)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except KeyError:
        raise HTTPException(status_code=404, detail=f"no user {user_id}")
    try:
        export_logs_to_csv()
    except Exception:
        pass
    return {"deleted": user_id, "renamed": [{"from": a, "to": b} for a, b in renames],
            "users": [_user_view(u) for u in user_registry.list_users()]}


@app.post("/users/{user_id}/reset")
def reset_user(user_id: str):
    if user_registry.get_user(user_id) is None:
        raise HTTPException(status_code=404, detail=f"no user {user_id}")
    style_model.reset_user(user_id)
    return {"user": _user_view(user_registry.get_user(user_id))}


@app.post("/deliver")
def deliver(req: DeliverRequest):
    """
    Adaptive voice delivery: turns a Sinhala answer into a timed, segmented speech plan.
    """
    try:
        profile = get_voice_delivery_profile(req.user_id)
        plan = plan_speech(req.answer_si, req.style, profile=profile, replay=req.replay)
        return plan
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/delivery-profile/{user_id}")
def get_delivery_profile_endpoint(user_id: str):
    """
    Returns the user's adaptive voice delivery profile (rate, pauses, chunk size, streak).
    """
    try:
        return get_voice_delivery_profile(user_id)
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
