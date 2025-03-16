from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import google.generativeai as genai
import uvicorn
from PIL import Image
import io
import base64
import torch
from torchvision import transforms
from diffusers import StableDiffusionPipeline

# Initialize FastAPI
app = FastAPI()

# Configure Gemini API
genai.configure(api_key="YOUR_GEMINI_API_KEY")

# Load Stable Diffusion Model
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe.to("cuda" if torch.cuda.is_available() else "cpu")

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
    image = pipe(request.text).images[0]
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="PNG")
    img_base64 = base64.b64encode(img_bytes.getvalue()).decode("utf-8")
    return {"image_base64": img_base64}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

