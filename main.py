import os
import json
import time
import random
import requests
from datetime import datetime
from config import COMFYUI_API_URL, COMFYUI_WS_URL, LLAMA_API_URL, OUTPUT_DIR, METADATA_FILE

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Types of environments for various scenarios
ENVIRONMENTS = [
    "dense forest", "pine forest", "oak woodland", "eucalyptus forest",
    "urban residential area", "apartment complex", "suburban neighborhood", "city center",
    "industrial factory", "chemical plant", "oil refinery", "manufacturing facility",
    "warehouse", "office building", "shopping mall", "school campus",
    "rural landscape", "farmland", "mountain terrain", "grassland",
    "national park", "wilderness area", "campground", "hiking trail"
]

# Time of day and weather conditions
TIME_WEATHER = [
    "early morning with clear sky", "bright sunny day", "cloudy afternoon", 
    "sunset with orange sky", "dusk with fading light", "night with moonlight",
    "foggy morning", "misty conditions", "after light rain", "windy day",
    "humid summer day", "dry autumn afternoon", "cold winter morning", "spring evening"
]

# Fire and Smoke Stages
FS_STAGES = [
    "very early stage with barely visible thin smoke wisp rising between trees",
    "small amount of white smoke rising slowly, no visible flames yet",
    "thin smoke column starting to form, barely noticeable",
    "light gray smoke beginning to accumulate near ground level",
    "small smoke plume developing, with tiny ember just becoming visible",
    "early smoke formation with first small flames beginning to appear",
    "visible smoke with small flames starting to spread",
    "moderate smoke development with growing flames",
    "thickening smoke with established fire beginning to spread"
]

# POVs
POVs = [
    "aerial drone view from 50 meters height", "aerial drone view from 100 meters height",
    "security camera perspective mounted on pole", "security camera view from building",
    "ground level perspective", "from distance of 100 meters", "from hillside overlooking area",
    "from forest watchtower", "through trees", "from road perspective"
]

def generate_prompt_with_llama():
    """
    Generate image prompt using Llama
    """

    environment = random.choice(ENVIRONMENTS)
    time_weather = random.choice(TIME_WEATHER)
    fire_stage = random.choice(FS_STAGES)
    pov = random.choice(POVs)

    system_message = """
    You are an expert at creating detailed prompts
    for AI image generation. Create realistic, detailed
    prompt generating an image of smoke/fire detection
    scenarios.

    Focus on early detection stages where smoke or initial
    is just becoming visible. Make your prompts photorealistic
    and detailed.
    """

    user_message = f"""
    Create a single detailed image generation prompt for a scenario
    with:
        - Environment: {environment}
        - Conditions: {time_weather}
        - Fire/smoke stage: {fire_stage}
        - Camera perspective: {pov}

    Focus on photorealism and detail. REturn ONLY the prompt text with
    no explanations or additional text.
    """

    payload = {
        "model": "llama3.1",
        "prompt": system_message + "\n\n" + user_message,
        "temperature": 0.7
    }

    try:
        response = requests.post(LLAMA_API_URL, json=payload)
        if response.status_code == 200:
            # content = response.json()["choices"][0]["message"]["content"]
            # content = content.strip()

            # Process streaming response
            content = ""
            for line in response.text.strip().split('\n'):
                if line:
                    json_response = json.loads(line)
                    if 'response' in json_response:
                        content += json_response['response']
                    if json_response.get('done', False):
                        break

            return content, {"environment": environment, "time_weather": time_weather, "fire_stage": fire_stage, "pov": pov}
        else:
            print(f"Error from Llama API: {response.status_code}")
            return None, None
    except Exception as e:
        print(f"Exception when calling Llama API: {e}")
        return None, None

def create_comfyui_workflow(prompt):
    """
    Create ComfUI workflow for Stable Diffusion XL with given prompt
    """
    negative_prompt = "low quality, bad image, blurry, distorted, deformed, disfigured, text, watermark, signature, poor composition, unrealistic, cartoonish"

    workflow = {
        "3": {
            "inputs": {
                "seed": random.randint(1, 9999999),
                "steps": 30,
                "cfg": 7.5,
                "sampler_name": "euler_ancestral",
                "scheduler": "normal",
                "denoise": 1,
                "model": ["4", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["5", 0]
            },
            "class_type": "KSampler",
        },
        "4": {
            "inputs": {
                "ckpt_name": "sd_xl_base_1.0.safetensors"
            },
            "class_type": "CheckpointLoaderSimple"
        },
        "5": {
            "inputs": {
                "width": 512,
                "height": 512,
                "batch_size": 1
            },
            "class_type": "EmptyLatentImage"
        },
        "6": {
            "inputs": {
                "text": prompt,
                "clip": ["4", 1]
            },
            "class_type": "CLIPTextEncode"
        },
        "7": {
            "inputs": {
                "text": negative_prompt,
                "clip": ["4", 1]
            },
            "class_type": "CLIPTextEncode"
        },
        "8": {
            "inputs": {
                "samples": ["3", 0],
                "vae": ["4", 2]
            },
            "class_type": "VAEDecode"
        },
        "9": {
            "inputs": {
                "filename_prefix": "FS_",
                "images": ["8", 0]
            },
            "class_type": "SaveImage"
        }
    }

    return workflow

def run_comfyui_workflow(workflow):
    """
    Execute the workflow on ComfyUI API and return the image
    """

    try:
        response = requests.post(COMFYUI_API_URL, json={"prompt": workflow})
        if response.status_code == 200:
            prompt_id = response.json()["prompt_id"]

            time.sleep(30)

            return prompt_id
        else:
            print(f"Error from ComfyUI API: {response.status_code}")
            return None
    except Exception as e:
        print(f"Exception when calling ComfyUI API: {e}")
        return None

def save_metadata(prompt, metadata, image_path, prompt_id):
    """
    Save the metadata about the generated image
    """
    if not os.path.exists(METADATA_FILE):
        all_metadata = []
    else:
        with open(METADATA_FILE, "r") as f:
            all_metadata = json.load(f)

    entry = {
        "timestamp": datetime.now().isoformat(),
        "prompt": prompt,
        "metadata": metadata,
        "image_path": image_path,
        "prompt_id": prompt_id
    }

    all_metadata.append(entry)

    with open(METADATA_FILE, "w") as f:
        json.dump(all_metadata, f, indent=2)

def generate_batch(num_images=10):
    """
    Generate a batch of iamages
    """
    for i in range(num_images):
        print(f"Generating image {i+1}/{num_images}")

        prompt, metadata = generate_prompt_with_llama()
        if not prompt:
            print("Failed to generate a prompt. Skipping.")
            continue

        print(f"Generated prompt: {prompt}")

        workflow = create_comfyui_workflow(prompt)
        prompt_id = run_comfyui_workflow(workflow)

        if prompt_id:
            image_path = f"output_from_comfyui_{prompt_id}.png"

            save_metadata(prompt, metadata, image_path, prompt_id)
            print(f"Generated image with prompt_id: {prompt_id}")
        else:
            print("Failed to generate image")

    time.sleep(5)


if __name__ == "__main__":
    print("Starting automated smoke/fire image generation")
    generate_batch(10)  # Generate 10 images by default
    print("Batch generation complete")
