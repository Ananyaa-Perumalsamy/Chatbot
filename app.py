from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import google.generativeai as genai
import uvicorn
from PIL import Image
import io
import base64
import requests

# Initialize FastAPI
app = FastAPI()

# Configure Gemini API
genai.configure(api_key="AIzaSyD2AEOW9M_O3ibqkGPdG9Pd5VFwt0tdxPE")

class TextRequest(BaseModel):
    text: str

@app.post("/chat/text")
async def chat_text(request: TextRequest):
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(request.text)
    return {"response": response.text}

@app.post("/chat/image")
async def chat_image(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read()))
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="PNG")
    image_bytes = image_bytes.getvalue()
    
    model = genai.GenerativeModel("gemini-pro-vision")
    response = model.generate_content(["Describe this image", image_bytes])
    return {"response": response.text}

@app.post("/generate/image")
async def generate_image(request: TextRequest):
    dalle_url = "https://api.openai.com/v1/images/generations"  # DALL·E API endpoint
    headers = {"Authorization": "Bearer YOUR_OPENAI_API_KEY", "Content-Type": "application/json"}
    payload = {"model": "dall-e-3", "prompt": request.text, "n": 1, "size": "1024x1024"}
    
    response = requests.post(dalle_url, json=payload, headers=headers)
    if response.status_code == 200:
        image_url = response.json()["data"][0]["url"]
        return {"image_url": image_url}
    else:
        return {"error": "Image generation failed", "details": response.text}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
