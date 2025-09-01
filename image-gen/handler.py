import os, uuid
import runpod
import torch
from diffusers import DiffusionPipeline
from huggingface_hub import snapshot_download

MODEL_ID = "HiDream-ai/HiDream-I1-Dev"

# Cache Hugging Face model inside persistent RunPod volume
MODEL_PATH = snapshot_download(repo_id=MODEL_ID, cache_dir="/runpod-volume/models")

pipe = DiffusionPipeline.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16
).to("cuda")

def generate_images(job_id, scenes):
    results = []
    os.makedirs("/outputs", exist_ok=True)

    for scene in scenes:
        prompt = scene["description"]
        image = pipe(prompt, num_inference_steps=28).images[0]

        file_name = f"{job_id}_scene{scene['id']}.png"
        path = f"/outputs/{file_name}"
        image.save(path)
        results.append(path)

    return results

def handler(job):
    input_data = job["input"]
    job_id = input_data.get("job_id", str(uuid.uuid4()))
    scenes = input_data.get("scenes", [])
    images = generate_images(job_id, scenes)
    return {"job_id": job_id, "images": images}

runpod.serverless.start({"handler": handler})
