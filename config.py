import os

# Configuration
COMFYUI_API_URL = "http://127.0.0.1:8188/prompt"
COMFYUI_WS_URL = "ws://127.0.0.1:8188/ws"
LLAMA_API_URL = "http://localhost:8080/v1/chat/completions"
OUTPUT_DIR = "fs_dataset"
METADATA_FILE = os.path.join(OUTPUT_DIR, "metadata.json")
