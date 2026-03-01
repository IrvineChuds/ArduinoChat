print("[SYSTEM] Initializing API...")

from fastapi import FastAPI

from google import genai
from google.genai import types
from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer
from faiss import IndexFlatIP
import numpy as np


import os

SYSTEM_INSTRUCTION = """
You are a conversational AI named "Makani". Always respond as "Makani" when asked for your name.\n
From now on, respond in a natural, casual, speaking tone.\n
Keep responses short-length unless I ask for detail.\n
Ask follow-up questions like a real conversation.\n
"""

MODEL = "gemini-3-flash-preview"
EMBEDDING_MODEL = "BAAI/bge-small-en"
INDEX_DIMENSIONS = 384

load_dotenv()

# setup client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
# single chat for all users
chat = client.chats.create(
    model=MODEL,
    config=types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION
    )
)

print("[SYSTEM] Loading Embedder...")
embedder = SentenceTransformer("BAAI/bge-small-en")

indexes: dict[str, IndexFlatIP] = {}
documents: dict[str, list[str]] = {}

print("[SYSTEM] Initializing Fast API...")
app = FastAPI()

def extract_user_facts(user_query: str) -> list[str] | None:
    response = client.models.generate_content(
        model=MODEL,
        contents=("From the following text, extract stable, long-term facts about the user.\n"
                  "Ignore temporary states, emotions, or situational context.\n"
                  "Return one concise fact per line.\n"
                  'Return "NO" if none exist.\n'
                  f'Text: "{user_query}"')
    )

    if response.text.strip().upper().startswith("NO"):
        return None

    return response.text.split("\n")

def generate_content(msg: str, user: str) -> str:
    # extract user facts from message
    content = f"USER NAME: {user}\n"

    if user not in indexes:
        indexes[user] = IndexFlatIP(INDEX_DIMENSIONS)

    if user not in documents:
        documents[user] = []

    embedding = embedder.encode(["Represent this user memory for searching relevant passages: " + msg], normalize_embeddings=True)
    lims, _, indices = indexes[user].range_search(np.array(embedding), 0.8)

    if len(indices) != 0:
        # add relevant memories to content
        content += "USER PROFILE:\n"
        for i in indices:
            content += f"- {documents[user][i]}\n"
            print(f"[SYSTEM] USER PROFILE SENT: {documents[user][i]}")

    facts = extract_user_facts(msg)

    if facts is not None:
        # embed message for memory
        embeddings = embedder.encode(["Represent this user memory for searching relevant passages: " + fact for fact in facts], normalize_embeddings=True)

        # remove duplicates
        lims, scores, indices = indexes[user].range_search(np.array(embeddings), 0.9)

        non_duplicate_embeddings = []
        non_duplicate_facts = []

        for i in range(len(lims) - 1):
            if (lims[i + 1] - lims[i] == 0): # keeps facts with NO matches
                non_duplicate_embeddings.append(embeddings[i])
                non_duplicate_facts.append(facts[i])

        if len(non_duplicate_embeddings) != 0:
            indexes[user].add(np.array(non_duplicate_embeddings))
            documents[user].extend(non_duplicate_facts)
        

    content += f"USER: {msg}" # actual message

    return content

def send_message(msg: str, user: str) -> str:
    # every now and then, resend system instruction
    print(f"[SYSTEM] USER ({user}) SENT: {msg}")

    content = generate_content(msg, user)

    response = chat.send_message(content)

    return response.text

@app.post("/model")
def receive_data(data: dict):
    user: str = data["user"]
    message: str = data["message"]

    response = send_message(message, user)

    print(f"[SYSTEM] MODEL RESPONSE: {response}")
    print()
    print(f"[SYSTEM] Document: {documents}")
    print()

    return response


print("[SYSTEM] Ready!")

# def main() -> None:
#     print("[SYSTEM] Ready!")

#     while True:
#         user = input("User: ")
#         message = input("Message: ")
#         print()

#         if message == "q":
#             return

#         print(f"[SYSTEM] MODEL RESPONSE: {send_message(message, user)}")
#         print()
#         print(f"[SYSTEM] Document: {documents}")
#         print()

# if __name__ == "__main__":
#     main()