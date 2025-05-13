# Automated Image Generation with ComfyUI

## Tools

- [ComfyUI](https://www.comfy.org/download) is used to handle **Stable Diffusion** prompts
- **Llama3.1** downloaded from [Ollama](https://ollama.com/)
    - You can also download **Llama3.2**. In this case you need to **modify** the **line 104** in function `generate_prompt_with_llama()`
- [SDXL1.0 Base](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/sd_xl_base_1.0.safetensors) from HuggingFace
- [SDXL1.0 Refiner](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/blob/main/sd_xl_refiner_1.0.safetensors) from Hugging Face