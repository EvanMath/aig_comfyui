import os

# Configuration
COMFYUI_BASE_URL = "http://127.0.0.1:8188"
COMFYUI_API_URL = f"{COMFYUI_BASE_URL}/prompt"  # Main endpoint for workflows
COMFYUI_VIEW_URL = f"{COMFYUI_BASE_URL}/view"  # For retrieving images
COMFYUI_WS_URL = "ws://127.0.0.1:8188/ws"
LLAMA_API_URL = "http://localhost:11434/api/generate"
OUTPUT_DIR = "fs_dataset"
METADATA_FILE = os.path.join(OUTPUT_DIR, "metadata.json")

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