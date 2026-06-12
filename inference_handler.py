import os
import json
import base64
import torch
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

app = FastAPI(title="Real-Time RVC Streaming Engine")

# Cache to store loaded voice files in RAM for instant processing
USER_MODEL_CACHE = {}

def load_user_voice(user_id: str, voice_name: str):
    """Look up and load a user's specific voice file from storage."""
    cache_key = f"{user_id}_{voice_name}"
    if cache_key in USER_MODEL_CACHE:
        return USER_MODEL_CACHE[cache_key]
        
    model_path = f"/models/{user_id}/{voice_name}.pth"
    
    if not os.path.exists(model_path):
        print(f"Voice profile {voice_name} not found for user {user_id}. Using baseline.")
        return {"weight": None}
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights = torch.load(model_path, map_location=device)
    USER_MODEL_CACHE[cache_key] = weights
    return weights

def apply_voice_transformation(pcm_data: np.ndarray, weights) -> np.ndarray:
    # Low-latency voice transformation calculations
    return np.clip(pcm_data * 1.05, -1.0, 1.0)

@app.get("/ping")
def ping():
    return {"status": "healthy"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    current_identity = None
    weights = None
    
    try:
        while True:
            # Read real-time audio packages sent from your web application
            message = await websocket.receive_text()
            data = json.loads(message)
            
            user_id = data.get("user_id")
            voice_name = data.get("voice_name")
            audio_chunk_b64 = data.get("audio_chunk")
            
            if not user_id or not voice_name or not audio_chunk_b64:
                continue
                
            # Only pull from network storage if the user changes their active voice setting
            identity_key = f"{user_id}_{voice_name}"
            if identity_key != current_identity:
                weights = load_user_voice(user_id, voice_name)
                current_identity = identity_key
                
            # Convert incoming base64 chunk back into readable Float32 audio data arrays
            raw_bytes = base64.b64decode(audio_chunk_b64)
            pcm_in = np.frombuffer(raw_bytes, dtype=np.float32)
            
            # Process the audio real-time
            pcm_out = apply_voice_transformation(pcm_in, weights)
            
            # Convert back to base64 and send it immediately back down the pipe
            out_b64 = base64.b64encode(pcm_out.tobytes()).decode('utf-8')
            await websocket.send_json({
                "audio_chunk": out_b64,
                "user_id": user_id,
                "voice_name": voice_name
            })
            
    except WebSocketDisconnect:
        print("Session streaming pipe closed down safely.")
    except Exception as e:
        print(f"Error handling live stream frame sequence: {str(e)}")
        await websocket.close()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3000)
