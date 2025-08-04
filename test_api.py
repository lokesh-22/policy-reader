import requests
import json

# Test the API endpoint
def test_api():
    url = "http://localhost:8000/api/v1/hackrx/run"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer 9d0a6a0d59a944b7b92b0a33b4cee5b30f2c00b4b098f133cfd1e36a90ada7d1"
    }
    
    payload = {
        "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
        "questions": [
            "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
            "What is the waiting period for pre-existing diseases (PED) to be covered?"
        ]
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_api()




# curl -X POST "http://localhost:8000/api/v1/hackrx/run" \
#      -H "Content-Type: application/json" \
#      -H "Authorization: Bearer 9d0a6a0d59a944b7b92b0a33b4cee5b30f2c00b4b098f133cfd1e36a90ada7d1" \
#      -d '{
#        "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
#        "questions": ["What is the grace period for premium payment?"]
#      }'