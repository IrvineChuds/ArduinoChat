from google import genai
from google.genai import types
from dotenv import load_dotenv

import os

SYSTEM_INSTRUCTION = """
You are a conversational AI named "Makani". Always respond as "Makani" when asked for your name.
From now on, respond in a natural, casual, speaking tone.
Keep responses short-length unless I ask for detail.
Ask follow-up questions like a real conversation.
"""

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
chat = client.chats.create(
    model="gemini-3-flash-preview",
    config=types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION
    )
)

def send_message(msg: str, user: str):
    content = (f"USER NAME: {user}" # dynamic profile
               f"USER: {msg}") # actual message

    response = chat.send_message(content)

    return response.text


def main():
    while True:
        msg = input()

        if msg == "q":
            return

        print(send_message(msg, "Ben"))

if __name__ == "__main__":
    main()