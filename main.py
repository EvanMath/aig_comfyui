import os
import json
import time
import random
import requests
import logging
import websocket
import threading
from datetime import datetime
from config import COMFYUI_API_URL, COMFYUI_WS_URL, LLAMA_API_URL, OUTPUT_DIR, METADATA_FILE

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
                "filename_prefix": os.path.join(OUTPUT_DIR, "FS_"),
                "images": ["8", 0]
            },
            "class_type": "SaveImage"
        }
    }

    return workflow

def on_message(ws, message):
    """Handle WebSocket messages"""
    try:
        data = json.loads(message)
        if data.get("type") == "executing":
            logger.info(f"Executing node: {data.get('data', {}).get('node')}")
        elif data.get("type") == "progress":
            logger.info(f"Progress: {data.get('data', {}).get('value') * 100:.0f}%")
        elif data.get("type") == "executed":
            logger.info("Node execution completed")
    except Exception as e:
        logger.error(f"Error processing WebSocket message: {e}")

def on_error(ws, error):
    """Handle WebSocket errors"""
    logger.error(f"WebSocket error: {error}")

def on_close(ws, close_status_code, close_msg):
    """Handle WebSocket connection close"""
    logger.info("WebSocket connection closed")

def on_open(ws):
    """Handle WebSocket connection open"""
    logger.info("WebSocket connection established")

def run_comfyui_workflow(workflow):
    """
    Execute the workflow on ComfyUI API and return the image
    """
    try:
        # First check if ComfyUI server is running
        try:
            health_check = requests.get("http://127.0.0.1:8188/")
            if health_check.status_code != 200:
                logger.error(f"ComfyUI server is not responding properly. Status code: {health_check.status_code}")
                return None, None
            logger.info("ComfyUI server is running and responding")
        except requests.exceptions.ConnectionError:
            logger.error("Could not connect to ComfyUI server. Make sure it's running at the configured URL.")
            return None, None

        # Ensure output directory exists and is writable
        try:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            test_file = os.path.join(OUTPUT_DIR, "test_write.tmp")
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
            logger.info(f"Output directory {OUTPUT_DIR} is writable")
        except Exception as e:
            logger.error(f"Output directory is not writable: {str(e)}")
            return None, None

        # Send the workflow
        logger.info("Sending workflow to ComfyUI...")
        response = requests.post(COMFYUI_API_URL, json={"prompt": workflow})
        
        if response.status_code == 200:
            prompt_id = response.json()["prompt_id"]
            logger.info(f"Workflow accepted with prompt_id: {prompt_id}")
            
            # Set up WebSocket connection for status updates
            ws = websocket.WebSocketApp(
                COMFYUI_WS_URL,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
                on_open=on_open
            )
            
            # Start WebSocket connection in a separate thread
            ws_thread = threading.Thread(target=ws.run_forever)
            ws_thread.daemon = True
            ws_thread.start()
            
            # Wait for the image to be generated with proper status checking
            max_wait_time = 120  # Maximum wait time in seconds
            start_time = time.time()
            image_found = False
            
            while time.time() - start_time < max_wait_time:
                try:
                    # Check execution status
                    status_response = requests.get(f"{COMFYUI_API_URL}/prompt")
                    if status_response.status_code == 200:
                        status_data = status_response.json()
                        logger.debug(f"Status data: {json.dumps(status_data, indent=2)}")
                        
                        # Check if our prompt is still executing
                        if prompt_id in status_data:
                            logger.info("Workflow is still running...")
                        else:
                            # Check for the output image
                            output_response = requests.get(f"{COMFYUI_API_URL}/output")
                            if output_response.status_code == 200:
                                output_data = output_response.json()
                                logger.debug(f"Output data: {json.dumps(output_data, indent=2)}")
                                
                                # Look for our image in the output
                                for node_id, node_output in output_data.items():
                                    if "images" in node_output:
                                        for image_data in node_output["images"]:
                                            if "filename" in image_data:
                                                # Get the image data
                                                image_response = requests.get(f"{COMFYUI_API_URL}/view?filename={image_data['filename']}")
                                                if image_response.status_code == 200:
                                                    # Save the image to our output directory
                                                    image_path = os.path.join(OUTPUT_DIR, f"FS_{prompt_id}.png")
                                                    with open(image_path, "wb") as f:
                                                        f.write(image_response.content)
                                                    logger.info(f"Image saved successfully at: {image_path}")
                                                    image_found = True
                                                    break
                    else:
                        logger.error(f"Failed to get execution status. Status code: {status_response.status_code}")
                        logger.error(f"Response content: {status_response.text}")
                except Exception as e:
                    logger.error(f"Error during status check: {str(e)}")
                
                if image_found:
                    break
                    
                time.sleep(2)  # Wait 2 seconds before checking again
                logger.info(f"Still waiting for image generation... ({int(time.time() - start_time)} seconds elapsed)")
            
            # Close WebSocket connection
            ws.close()
            
            if image_found:
                image_path = os.path.join(OUTPUT_DIR, f"FS_{prompt_id}.png")
                if os.path.exists(image_path):
                    logger.info(f"Image verified at path: {image_path}")
                    return prompt_id, image_path
                else:
                    logger.error(f"Image was not saved properly at: {image_path}")
            else:
                logger.error("Image generation timed out or failed")
                # List contents of output directory for debugging
                logger.info(f"Contents of {OUTPUT_DIR}:")
                for file in os.listdir(OUTPUT_DIR):
                    logger.info(f"  - {file}")
                return None, None
        else:
            logger.error(f"Error from ComfyUI API: {response.status_code}")
            logger.error(f"Response content: {response.text}")
            return None, None
    except Exception as e:
        logger.error(f"Exception when calling ComfyUI API: {str(e)}")
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

def generate_batch(num_images=10):
    """
    Generate a batch of images
    """
    logger.info(f"Starting batch generation of {num_images} images")

    for i in range(num_images):
        logger.info(f"Generating image {i+1}/{num_images}")
        
        prompt, metadata = generate_prompt_with_llama()
        if not prompt:
            logger.error("Failed to generate a prompt. Skipping.")
            continue

        logger.info(f"Generated prompt: {prompt}")

        workflow = create_comfyui_workflow(prompt)
        result = run_comfyui_workflow(workflow)

        if result:
            prompt_id, image_path = result
            if image_path:
                save_metadata(prompt, metadata, image_path, prompt_id)
                logger.info(f"Successfully generated and saved image {i+1}/{num_images}")
            else:
                logger.error(f"Failed to save image {i+1}/{num_images}")
        else:
            logger.error(f"Failed to generate image {i+1}/{num_images}")
        
        time.sleep(5)  # Wait between generations

    logger.info("Batch generation complete")

if __name__ == "__main__":
    logger.info("Starting automated smoke/fire image generation")
    generate_batch(5)  # Generate 10 images by default
    logger.info("Batch generation complete")
