import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Load environment variables from .env
load_dotenv()

# Get Hugging Face token
token = os.getenv("HF_TOKEN")

# Connect to Hugging Face
client = InferenceClient(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    token=token
)

# Take topic from user
topic = input("Enter the topic for flashcards: ")

# Send request to the model
response = client.chat_completion(
    messages=[
        {
            "role": "user",
            "content": f"""
Create 5 simple study flashcards about: {topic}

For each flashcard, give:
1. Question
2. Answer

Keep the answers short and easy to understand.
"""
        }
    ],
    max_tokens=500
)

# Display the generated flashcards
print("\n--- FLASHCARDS ---\n")
print(response.choices[0].message.content)