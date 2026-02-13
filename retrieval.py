import faiss
import numpy as np
import base64
from sentence_transformers import SentenceTransformer
from config import Config
from groq import Groq

# Initialize Groq client
client = Groq(api_key=Config.GROQ_API_KEY)

def get_query_embedding(query):
    """Converts the user's question into a vector using local Sentence Transformers."""
    model = SentenceTransformer(Config.EMBEDDING_MODEL)
    embedding = model.encode([query])[0]
    return np.array(embedding).astype('float32')

def retrieve_context(query, top_k=3):
    """Searches the FAISS index for the most relevant data."""
    # 1. Load the saved index
    index = faiss.read_index("vector_db/index.faiss")
    
    # 2. Vectorize the user's question
    query_vector = get_query_embedding(query).reshape(1, -1)
    
    # 3. Search FAISS
    distances, indices = index.search(query_vector, top_k)
    
    # In a real app, you'd use 'indices' to map back to your text/image chunks.
    # For now, we return the indices as a placeholder for the retrieved text.
    return indices[0]

def generate_answer(query, retrieved_chunks, temperature=0.7, max_tokens=1024):
    """Sends the query + retrieved context to Groq (Llama 3 Vision)."""
    
    # Constructing the prompt with 'Retrieved Context'
    context_text = "\n".join([f"Source {i}: {text}" for i, text in enumerate(retrieved_chunks)])
    
    system_prompt = """You are a highly intelligent and versatile AI assistant, similar to ChatGPT or Gemini.
    
    Instructions:
    1. **Primary Goal**: Answer the user's question helpfully, accurately, and creatively.
    2. **Context Usage**: You have access to a document snippet (Context) below. Use it IF it helps answer the question.
    3. **General Knowledge**: If the Context is not relevant or does not contain the answer, completely ignore it. Use your own general knowledge to answer just like a standard AI assistant.
    4. **Tone**: Be friendly, professional, and conversational.
    """

    user_message = f"""
    CONTEXT FROM DOCUMENTS:
    {context_text}
    
    USER QUESTION:
    {query}
    """
    
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        model=Config.VISION_MODEL,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    
    return chat_completion.choices[0].message.content

def analyze_image(image_bytes):
    """Analyzes an uploaded image using Groq's Vision model."""
    # Encode image to base64
    base64_image = base64.b64encode(image_bytes).decode('utf-8')
    
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze this image in detail. Describe everything you see, including objects, text, colors, and context."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ],
            }
        ],
        model=Config.GROQ_VISION_MODEL,
    )
    
    return chat_completion.choices[0].message.content

def summarize_document(text_chunks):
    """Generates a summary of the provided text chunks."""
    if not text_chunks:
        return "No text content found to summarize."

    # Combine chunks into one text block
    full_text = "\n".join(text_chunks)
    
    # Truncate if extremely long to avoid API errors (approx 30k chars)
    if len(full_text) > 30000:
        full_text = full_text[:30000] + "...\n(Content truncated for length)"
        
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Please provide a comprehensive summary of the following document."},
            {"role": "user", "content": f"Document Content:\n{full_text}"}
        ],
        model=Config.VISION_MODEL,
    )
    
    return chat_completion.choices[0].message.content