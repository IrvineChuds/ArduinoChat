from fastapi import FastAPI
import json

app = FastAPI()

@app.post("/model")
def receive_data(data: dict):
    return f"The server recieved {data["message"]}"