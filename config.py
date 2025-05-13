import os

# Configuration
COMFYUI_BASE_URL = "http://127.0.0.1:8188"
COMFYUI_API_URL = f"{COMFYUI_BASE_URL}/prompt"  # Main endpoint for workflows
COMFYUI_VIEW_URL = f"{COMFYUI_BASE_URL}/view"  # For retrieving images
COMFYUI_WS_URL = "ws://127.0.0.1:8188/ws"
LLAMA_API_URL = "http://localhost:11434/api/generate"
OUTPUT_DIR = "fs_dataset"
METADATA_FILE = os.path.join(OUTPUT_DIR, "metadata.json")
