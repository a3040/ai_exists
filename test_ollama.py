import requests
url = "http://localhost:11434/api/generate"
payload = {
    "model": "qwen2.5:1.5b",
    "prompt": "안녕? 너는 누구니?",
    "stream": False,
    "options": {
        "temperature": 0.0
    }
}
try:
    response = requests.post(url, json=payload, timeout=10)
    print(response.json().get('response', ''))
except Exception as e:
    print(f"Error: {e}")
