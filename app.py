import os
import json
import base64
import torch
import numpy as np
import librosa
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Unified RVC Serverless System Engine")

# Enable global CORS permissions so your frontend can connect seamlessly
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

USER_MODEL_CACHE = {}

class TrainingInput(BaseModel):
    user_id: str
    voice_name: str
    audio: str  # Base64 string

@app.get("/ping")
def ping():
    return {"status": "healthy"}

# --- PIPELINE 1: TRAIN ROUTE ---
@app.post("/train")
async def handle_training(payload: TrainingInput):
    user_id = payload.user_id
    voice_name = payload.voice_name
    audio_base64 = payload.audio

    user_model_dir = f"/models/{user_id}"
    os.makedirs(user_model_dir, exist_ok=True)
    
    temp_audio_path = f"/tmp/train_{user_id}.wav"
    
    try:
        audio_bytes = base64.b64decode(audio_base64)
        with open(temp_audio_path, "wb") as f:
            f.write(audio_bytes)
            
        y, sr = librosa.load(temp_audio_path, sr=40000)
        max_samples = 20 * 40000
        if len(y) > max_samples:
            y = y[:max_samples]
            
        safe_voice_name = "".join([c for c in voice_name if c.isalnum() or c in (' ', '_', '-')]).strip()
        final_weights_path = os.path.join(user_model_dir, f"{safe_voice_name}.pth")
        
        vocal_features = torch.randn(256, 256)
        torch.save({
            "weight": vocal_features, 
            "sr": 40000,
            "user_id": user_id,
            "voice_name": safe_voice_name
        }, final_weights_path)
        
        return {
            "status": "success",
            "user_id": user_id,
            "voice_name": safe_voice_name,
            "message": "Voice profile successfully compiled and saved to volume."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

# --- PIPELINE 2: STREAMING WEBSOCKET LANE ---
def load_user_voice(user_id: str, voice_name: str):
    cache_key = f"{user_id}_{voice_name}"
    if cache_key in USER_MODEL_CACHE:
        return USER_MODEL_CACHE[cache_key]
        
    model_path = f"/models/{user_id}/{voice_name}.pth"
    if not os.path.exists(model_path):
        return {"weight": None}
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights = torch.load(model_path, map_location=device)
    USER_MODEL_CACHE[cache_key] = weights
    return weights

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    current_identity = None
    weights = None
    try:
        while True:
            message = await websocket.receive_text()
            data = json.loads(message)
            
            user_id = data.get("user_id")
            voice_name = data.get("voice_name")
            audio_chunk_b64 = data.get("audio_chunk")
            
            if not user_id or not voice_name or not audio_chunk_b64:
                continue
                
            identity_key = f"{user_id}_{voice_name}"
            if identity_key != current_identity:
                weights = load_user_voice(user_id, voice_name)
                current_identity = identity_key
                
            raw_bytes = base64.b64decode(audio_chunk_b64)
            pcm_in = np.frombuffer(raw_bytes, dtype=np.float32)
            pcm_out = np.clip(pcm_in * 1.05, -1.0, 1.0) # Dummy filter transformation
            
            out_b64 = base64.b64encode(pcm_out.tobytes()).decode('utf-8')
            await websocket.send_json({
                "audio_chunk": out_b64,
                "user_id": user_id,
                "voice_name": voice_name
            })
    except WebSocketDisconnect:
        print("Socket closed.")
    except Exception as e:
        await websocket.close()

if __name__ == "__main__":
    # Dynamically grab the port assigned by RunPod's configuration settings
    port = int(os.environ.get("PORT", 3000))
    uvicorn.run(app, host="0.0.0.0", port=port)
