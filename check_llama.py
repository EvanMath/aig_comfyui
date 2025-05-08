import requests
import json

llama_urls = [
    "http://localhost:8080/v1/chat/completions",
    "http://localhost:1234/v1/chat/completions", 
    "http://localhost:11434/api/generate"
]

for url in llama_urls:
    print(f"Trying {url}...")

    try:
        if "/v1/" in url:
            payload = {
                "model": "llama3.1",
                "messages": [{"role": "user", "content": "Say hello!"}],
                "temperature": 0.7
            }
            response = requests.post(url, json=payload, timeout=20)
            print(f"Response: {response.status_code}")
            print(response.json())
        else:
            # Handle Ollama API streaming response
            payload = {
                "model": "llama3.1",
                "prompt": "Say hello!",
                "temperature": 0.7
            }
            response = requests.post(url, json=payload, timeout=20)
            print(f"Response: {response.status_code}")
            
            # Process streaming response
            full_response = ""
            for line in response.text.strip().split('\n'):
                if line:
                    json_response = json.loads(line)
                    if 'response' in json_response:
                        full_response += json_response['response']
                    if json_response.get('done', False):
                        break
            
            print(f"Complete response: {full_response}")
        
        print("Success with this URL!")
        break
    except Exception as e:
        print(f"Failed: {e}")
        continue