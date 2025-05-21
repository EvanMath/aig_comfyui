import os
import json
import time
import random
import requests
import logging
from websocket import WebSocketApp
import threading
import argparse
from datetime import datetime
from config import (
    COMFYUI_API_URL, 
    COMFYUI_VIEW_URL, 
    COMFYUI_BASE_URL, 
    COMFYUI_WS_URL, 
    LLAMA_API_URL, 
    OUTPUT_DIR, 
    METADATA_FILE, 
    ENVIRONMENTS, 
    TIME_WEATHER, 
    FS_STAGES, 
    POVs
)

# Set up logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"generation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
logger.info(f"Ensuring output directory exists: {OUTPUT_DIR}")


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

    Focus on early detection stages where smoke or fire
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

    Focus on photorealism and detail. Return ONLY the prompt text with
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

def create_comfyui_workflow(prompt, batch_size=1, model_name="sd_xl_base_1.0.safetensors"):
    """
    Create ComfUI workflow for Stable Diffusion XL with given prompt
    
    Args:
        prompt (str): The text prompt for image generation
        batch_size (int): Number of images to generate per prompt (default: 1)
        model_name (str): Name of the model checkpoint to use (default: sd_xl_base_1.0.safetensors)
    """
    negative_prompt = "low quality, bad image, blurry, distorted, deformed, disfigured, text, watermark, signature, poor composition, unrealistic, cartoonish"

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

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
                "ckpt_name": model_name
            },
            "class_type": "CheckpointLoaderSimple"
        },
        "5": {
            "inputs": {
                "width": 512,
                "height": 512,
                "batch_size": batch_size
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
                "filename_prefix": os.path.join(OUTPUT_DIR, "FS_"),
                "images": ["8", 0]
            },
            "class_type": "SaveImage"
        }
    }

    return workflow

def run_comfyui_workflow(workflow):
    """Run a workflow in ComfyUI and return the generated image."""
    try:
        # Send the workflow to ComfyUI
        logger.info("Sending workflow to ComfyUI...")
        response = requests.post(COMFYUI_API_URL, json={"prompt": workflow})
        response.raise_for_status()
        
        # Get the prompt_id from the response
        prompt_id = response.json()["prompt_id"]
        logger.info(f"Workflow accepted with prompt_id: {prompt_id}")
        
        # Create and start WebSocket in a separate thread
        execution_status = {"completed": False}
        last_progress = 0
        
        def on_message(ws, message):
            nonlocal execution_status, last_progress
            try:
                data = json.loads(message)
                if data.get("type") == "execution_complete":
                    execution_status["completed"] = True
                    logger.info("Execution completed")
                elif data.get("type") == "executing":
                    logger.info(f"Executing node: {data.get('data', {}).get('node')}")
                elif data.get("type") == "progress":
                    progress = min(1.0, data.get('data', {}).get('value', 0))
                    if progress > last_progress:
                        logger.info(f"Overall progress: {progress * 100:.1f}%")
                        last_progress = progress
            except Exception as e:
                logger.error(f"Error processing WebSocket message: {e}")

        def on_error(ws, error):
            logger.error(f"WebSocket error: {error}")

        def on_close(ws, close_status_code, close_msg):
            logger.info("WebSocket connection closed")

        def on_open(ws):
            logger.info("WebSocket connection established")
            
        ws = WebSocketApp(
            COMFYUI_WS_URL,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open
        )
        
        wst = threading.Thread(target=ws.run_forever)
        wst.daemon = True
        wst.start()
        
        # Wait for execution to complete with timeout
        start_time = time.time()
        while not execution_status["completed"]:
            time.sleep(1)
            elapsed = time.time() - start_time
            
            # Check if the prompt is still being processed
            try:
                history_response = requests.get(f"{COMFYUI_BASE_URL}/history/{prompt_id}")
                if history_response.status_code == 200:
                    history_data = history_response.json()
                    if history_data.get(prompt_id, {}).get("status", {}).get("completed", False):
                        execution_status["completed"] = True
                        logger.info("Execution completed according to history API")
            except Exception as e:
                logger.warning(f"Error checking history: {e}")
                
            if elapsed > 300:  # 5 minute timeout
                logger.error("Execution timed out after 5 minutes")
                break
        
        # Close WebSocket connection
        ws.close()
        
        # Wait a moment to ensure files are saved
        time.sleep(2)
        
        try:
            # Use the history endpoint to get output information
            history_response = requests.get(f"{COMFYUI_BASE_URL}/history/{prompt_id}")
            if history_response.status_code == 200:
                history_data = history_response.json()
                
                # Extract the output image filename from history
                outputs = history_data.get(prompt_id, {}).get("outputs", {})
                image_paths = []
                for node_id, node_output in outputs.items():
                    if "images" in node_output:
                        # Process all images in the batch
                        for image_info in node_output["images"]:
                            filename = image_info["filename"]
                            subfolder = image_info.get("subfolder", "")
                            
                            # Build the correct URL for the view endpoint
                            view_url = f"{COMFYUI_VIEW_URL}?filename={filename}"
                            if subfolder:
                                view_url += f"&subfolder={subfolder}"
                                
                            logger.info(f"Attempting to retrieve image with URL: {view_url}")
                            
                            # Download the image
                            image_response = requests.get(view_url)
                            image_response.raise_for_status()
                            
                            # Save the image with a unique identifier
                            image_path = os.path.join(OUTPUT_DIR, f"FS_{prompt_id}_{len(image_paths)}.png")
                            with open(image_path, "wb") as f:
                                f.write(image_response.content)
                                
                            logger.info(f"Successfully saved image to {image_path}")
                            image_paths.append(image_path)
                
                if image_paths:
                    return prompt_id, image_paths
        except Exception as e:
            logger.error(f"Error retrieving image: {e}")
        
        logger.error("Failed to retrieve the generated image")
        return None, None
        
    except Exception as e:
        logger.error(f"Error in workflow execution: {e}")
        return None, None

def save_metadata(prompt, metadata, image_path, prompt_id):
    """
    Save the metadata about the generated image
    """
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(METADATA_FILE), exist_ok=True)
        
        if not os.path.exists(METADATA_FILE):
            all_metadata = []
        else:
            try:
                with open(METADATA_FILE, "r") as f:
                    all_metadata = json.load(f)
            except json.JSONDecodeError:
                logger.error("Error reading metadata file, starting with empty list")
                all_metadata = []

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

        logger.info(f"Metadata saved for image: {image_path}")
    except Exception as e:
        logger.error(f"Error saving metadata: {str(e)}")

def generate_custom_prompt_with_llama(topic):
    """
    Generate image prompt using Llama for a custom topic
    
    Args:
        topic (str): The topic to generate prompts for
    """
    system_message = """
    You are an expert at creating detailed prompts
    for AI image generation. Create realistic, detailed
    prompts for the given topic.

    Focus on creating photorealistic and detailed prompts
    that would generate high-quality images.
    """

    user_message = f"""
    Create a single detailed image generation prompt for the topic:
    {topic}

    Focus on photorealism and detail. Return ONLY the prompt text with
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
            # Process streaming response
            content = ""
            for line in response.text.strip().split('\n'):
                if line:
                    json_response = json.loads(line)
                    if 'response' in json_response:
                        content += json_response['response']
                    if json_response.get('done', False):
                        break

            return content, {"topic": topic}
        else:
            print(f"Error from Llama API: {response.status_code}")
            return None, None
    except Exception as e:
        print(f"Exception when calling Llama API: {e}")
        return None, None

def generate_batch(mode="auto", topic=None, num_prompts=2, batch_size=2, model_name="sd_xl_base_1.0.safetensors"):
    """
    Generate a batch of images based on the specified mode
    
    Args:
        mode (str): Either "auto" for wildfire prompts or "custom" for custom topic
        topic (str): The topic to generate prompts for (only used in custom mode)
        num_prompts (int): Number of different prompts to generate
        batch_size (int): Number of images to generate per prompt
        model_name (str): Name of the model checkpoint to use
    """
    for i in range(num_prompts):
        if mode == "auto":
            prompt, metadata = generate_prompt_with_llama()
        else:
            prompt, metadata = generate_custom_prompt_with_llama(topic)
            
        if prompt:
            logger.info(f"Generated prompt {i+1}/{num_prompts}: {prompt}")
            
            # Create and run workflow
            workflow = create_comfyui_workflow(prompt, batch_size, model_name)
            result = run_comfyui_workflow(workflow)
            
            if result:
                prompt_id, image_paths = result
                if image_paths:
                    for image_path in image_paths:
                        save_metadata(prompt, metadata, image_path, prompt_id)
            else:
                logger.error(f"Failed to generate images for prompt {i+1}/{num_prompts}")
        else:
            logger.error(f"Failed to generate prompt {i+1}/{num_prompts}")

def main():
    parser = argparse.ArgumentParser(description='Generate images using ComfyUI and Llama')
    parser.add_argument('--mode', choices=['auto', 'custom'], default='auto',
                      help='Generation mode: auto for wildfire prompts, custom for custom topic')
    parser.add_argument('--topic', type=str,
                      help='Topic for custom mode (required if mode is custom)')
    parser.add_argument('--num-prompts', type=int, default=2,
                      help='Number of different prompts to generate')
    parser.add_argument('--batch-size', type=int, default=2,
                      help='Number of images to generate per prompt')
    parser.add_argument('--model', type=str, default="sd_xl_base_1.0.safetensors",
                      help='Model checkpoint to use')

    args = parser.parse_args()

    if args.mode == 'custom' and not args.topic:
        parser.error("--topic is required when using custom mode")

    generate_batch(
        mode=args.mode,
        topic=args.topic,
        num_prompts=args.num_prompts,
        batch_size=args.batch_size,
        model_name=args.model
    )

if __name__ == "__main__":
    main()
