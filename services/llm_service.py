# services/llm_service.py
import os
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError("GROQ_API_KEY not found in environment")

    return Groq(api_key=api_key)

SYSTEM_PROMPT = """ You are a helpful AI assistant. 
                    Answer clearly, accurately, and simply. 
                    If the question is about banking transactions, say: 
                    "I cannot help with banking actions." 
                    """

def ask_llm(question: str) -> str:
    client = get_groq_client()

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question}
        ],
        temperature=0.3
    )

    return response.choices[0].message.content


