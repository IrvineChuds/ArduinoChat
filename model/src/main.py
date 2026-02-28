print("[SYSTEM] Initializing API...")

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
documents: list[str] = []

def extract_user_facts(user_query: str) -> list[str]:
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=f'Decide if this sentence contains a long-term fact about the user that should be remembered: "{user_query}"\nAnswer return one fact per line in concise statements if YES, NO if there are no facts'
    )

    if response.text == "NO":
        return None

    return response.text.split("\n")

def generate_content(msg: str, user: str) -> str:
    # extract user facts from message
    content = f"USER NAME: {user}\n"

    if user not in indexes:
        indexes[user] = IndexFlatIP(384)

    embedding = embedder.encode(["Represent this user memory for searching relevant passages: " + msg], normalize_embeddings=True)
    lims, scores, indices = indexes[user].range_search(np.array(embedding), 0.8)

    if len(indices) != 0:
        # add relevant memories to content
        content += "USER PROFILE:\n"
        for i in indices:
            content += f"- {documents[i]}\n"

    facts = extract_user_facts(msg)

    if facts is not None:
        # embed message for memory
        embeddings = embedder.encode(["Represent this user memory for searching relevant passages: " + fact for fact in facts], normalize_embeddings=True)

        # remove duplicates
        lims, scores, indices = indexes[user].range_search(np.array(embeddings), 0.9)

        non_duplicate_embeddings = []
        non_duplicate_facts = []

        for i in range(len(lims) - 1):
            if (lims[i + 1] - lims[i] == 0):
                non_duplicate_embeddings.append(embeddings[i])
                non_duplicate_facts.append(facts[i])

        if len(non_duplicate_embeddings) != 0:
            indexes[user].add(np.array(non_duplicate_embeddings))
            documents.extend(non_duplicate_facts)
        

    content += f"USER: {msg}" # actual message

    return content

def send_message(msg: str, user: str) -> str:
    # every now and then, resend system instruction

    # TODO: extract user facts from message

    content = generate_content(msg, user)

    return content

    # response = chat.send_message(content)

    # return response.text

def main() -> None:
    print("[SYSTEM] Ready!")

    while True:
        message = input()
        print()

        if message == "q":
            return

        print(send_message(message, "Jaden"))
        print()

if __name__ == "__main__":
    main()