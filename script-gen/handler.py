import os
import runpod
from openai import OpenAI
import uuid
import json

# Init OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_script(prompt: str, job_id: str | None = None):
    job_id = job_id or str(uuid.uuid4())

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},  # strict JSON mode
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a screenplay generator. "
                    "Always respond in valid JSON format. "
                    "The JSON must have a top-level key 'scenes' (array). "
                    "Each scene must include: id (int), description (string), narration (string), duration (int seconds)."
                )
            },
            {
                "role": "user",
                "content": f"Generate a short video script for: {prompt}"
            }
        ],
        temperature=0.7
    )

    # Parse JSON string into Python dict
    try:
        parsed = json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        parsed = {"scenes": []}  # fallback if parsing fails

    return {
        "job_id": job_id,
        "scenes": parsed.get("scenes", [])
    }

def handler(job):
    """RunPod entrypoint"""
    input_data = job["input"]
    prompt = input_data.get("prompt", "A cinematic demo video")
    job_id = input_data.get("job_id", None)

    return generate_script(prompt, job_id)

# Register handler with RunPod serverless
runpod.serverless.start({"handler": handler})
