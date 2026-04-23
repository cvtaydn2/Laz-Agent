import httpx
import json

def test_selam():
    url = "http://127.0.0.1:8000/v1/chat/completions"
    payload = {
        "model": "laz-agent",
        "messages": [
            {"role": "user", "content": "selam"}
        ],
        "extra_body": {
            "workspace": "C:/Users/Cevat/Documents/Github/Laz-Agent/editor-agent",
            "mode": "ask"
        }
    }
    
    print("Sending request...")
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(url, json=payload)
            print(f"Status: {response.status_code}")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_selam()
