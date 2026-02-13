import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    JINA_API_KEY = os.getenv("JINA_API_KEY")
    
    # Models
    VISION_MODEL = "llama-3.3-70b-versatile"
    GROQ_VISION_MODEL = "llama-3.2-90b-vision-preview"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"