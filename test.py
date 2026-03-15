from google import genai

client = genai.Client(api_key="AIzaSyDfdC2sPhc_aM08GFHl7-6h3ANDwozr9x4")

try:
    models = list(client.models.list())
    print(f"Found {len(models)} models")
    for m in models:
        print(m.name)
except Exception as e:
    print(f"Error: {e}")