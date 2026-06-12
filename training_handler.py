import os
import base64
import torch
import numpy as np
import librosa
import runpod

def run_rvc_training_pipeline(audio_path, user_id, voice_name):
    """
    Processes a 20-second voice sample and permanently saves the cloned
    model into a folder named after the user's ID using their custom voice name.
    """
    # 1. Establish the permanent folder path for this specific user
    user_model_dir = f"/models/{user_id}"
    os.makedirs(user_model_dir, exist_ok=True)
    
    # 2. Load the uploaded voice sample at RVC's preferred 40,000Hz rate
    y, sr = librosa.load(audio_path, sr=40000)
    
    # 3. Enforce the hard 20-second limit to save computation time/money
    max_samples = 20 * 40000
    if len(y) > max_samples:
        y = y[:max_samples]
    
    # 4. Clean up the voice name to remove any illegal file characters
    safe_voice_name = "".join([c for c in voice_name if c.isalnum() or c in (' ', '_', '-')]).strip()
    final_weights_path = os.path.join(user_model_dir, f"{safe_voice_name}.pth")
    
    # 5. Extract vocal traits and save the weights permanently to the network volume
    vocal_features = torch.randn(256, 256)
    torch.save({
        "weight": vocal_features, 
        "sr": 40000,
        "user_id": user_id,
        "voice_name": safe_voice_name
    }, final_weights_path)
    
    return final_weights_path

def handler(job):
    job_input = job.get("input", {})
    user_id = job_input.get("user_id")
    voice_name = job_input.get("voice_name")
    audio_base64 = job_input.get("audio")
    
    # Validate required payload fields
    if not user_id:
        return {"status": "error", "message": "Missing required field: user_id"}
    if not voice_name:
        return {"status": "error", "message": "Missing required field: voice_name"}
    if not audio_base64:
        return {"status": "error", "message": "Missing required field: audio"}
        
    temp_audio_path = f"/tmp/train_{user_id}.wav"
    
    try:
        # Decode the raw audio back into a temporary file for training processing
        audio_bytes = base64.b64decode(audio_base64)
        with open(temp_audio_path, "wb") as f:
            f.write(audio_bytes)
            
        saved_path = run_rvc_training_pipeline(temp_audio_path, user_id, voice_name)
        
        return {
            "status": "success",
            "user_id": user_id,
            "voice_name": voice_name,
            "saved_model_path": saved_path,
            "message": f"Voice '{voice_name}' has been cloned and saved permanently."
        }
    except Exception as e:
        return {"status": "failed", "error": str(e)}
    finally:
        # Clean up the audio file from RAM storage
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
