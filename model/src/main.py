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
You are a conversational AI named "Thaumazein". Always respond as "Thaumazein" when asked for your name.
From now on, respond in a natural, casual, speaking tone that matches the user.
Keep responses super-short-length unless specifically asked for detail.
Remember relevant user preferences and context during the conversation to personalize responses.
Ask follow-up questions every 3 to 4 turns to keep the conversation flowing naturally.
If unsure of an answer, respond honestly and casually instead of making up details.
"""

MODEL = "gemini-3-flash-preview"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
INDEX_DIMENSIONS = 384

counter = 0

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
embedder = SentenceTransformer(EMBEDDING_MODEL)

indexes: dict[str, IndexFlatIP] = {}
documents: dict[str, list[str]] = {}

print("[SYSTEM] Initializing Fast API...")
app = FastAPI()

def extract_user_facts(user_query: str) -> list[str] | None:
    response = client.models.generate_content(
        model=MODEL,
        contents=("From the following text, extract stable, long-term facts about the user.\n"
                  "Treat obvious typos as if they were spelled correctly.\n"
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

    print("[SYSTEM] Encoding...")
    embedding = embedder.encode(["Represent this sentence for searching relevant passages: " + msg], normalize_embeddings=True)
    print("[SYSTEM] Searching...")
    lims, _, indices = indexes[user].range_search(np.array(embedding), 0.8)
    print("[SYSTEM] Finished Search!")

    if len(indices) != 0:
        # add relevant memories to content
        content += "USER PROFILE:\n"
        for i in indices:
            content += f"- {documents[user][i]}\n"
            print(f"[SYSTEM] USER PROFILE SENT: {documents[user][i]}")

    print("[SYSTEM] Extracting User Facts...")
    facts = extract_user_facts(msg)
    print("[SYSTEM] Finished Exraction!")

    if facts is not None:
        # embed message for memory
        print("[SYSTEM] Encoding...")
        embeddings = embedder.encode(["Represent this sentence for searching relevant passages: " + fact for fact in facts], normalize_embeddings=True)

        # remove duplicates
        print("[SYSTEM] Searching...")
        lims, _, indices = indexes[user].range_search(np.array(embeddings), 0.9)
        print("[SYSTEM] Finished Search!")


        non_duplicate_embeddings = []
        non_duplicate_facts = []

        print("[SYSTEM] Processing Facts")
        for i in range(len(lims) - 1):
            if (lims[i + 1] - lims[i] == 0): # keeps facts with NO matches
                non_duplicate_embeddings.append(embeddings[i])
                non_duplicate_facts.append(facts[i])

        if len(non_duplicate_embeddings) != 0:
            indexes[user].add(np.array(non_duplicate_embeddings))
            documents[user].extend(non_duplicate_facts)

        print("[SYSTEM] Finished Processing!")
        

    content += f"USER: {msg}" # actual message

    return content

def send_message(msg: str, user: str) -> str:
    # every now and then, resend system instruction
    global counter

    counter += 1

    if (counter >= 10):
        content = "Reminder: respond in a short, natural, speaking tone that matches the user.\n"
        counter = 0
    else:
        content = ""

    print(f"[SYSTEM] USER ({user}) SENT: {msg}")

    content += generate_content(msg, user)

    print("[SYSTEM] Querying...")
    response = chat.send_message(content)
    print("[SYSTEM] Finished Query!")

    return response.text

@app.post("/model")
def receive_data(data: dict):
    try:
        user: str = data["user"]
        message: str = data["message"]

        response = send_message(message, user)

        print(f"[SYSTEM] MODEL RESPONSE: {response}")
        print()
        print(f"[SYSTEM] Document: {documents}")
        print()

        return response
    except Exception as e:
        print(f"[SYSTEM] Error: {e}")
        return "Something went wrong, please try again later."


print("[SYSTEM] Ready!")