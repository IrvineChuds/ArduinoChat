import requests
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
import whisper

from yapper import PiperSpeaker, PiperVoiceUS

URL = "http://11.28.48.208:8000/model"  # don't change this

# ----------------------------
# Audio config
# ----------------------------
SAMPLE_RATE = 16000
CHANNELS = 1
OUT_WAV = Path("utterance.wav")

# ----------------------------
# TTS config (macOS say)
# ----------------------------
VOICE = "Melina"
RATE = 170  # lower = slower, higher = faster (words per minute)

# ----------------------------
# Silence detection tuning
# ----------------------------
RMS_THRESHOLD = 400
MIN_SPEECH_SEC = 0.4
MAX_UTTERANCE_SEC = 12.0
SILENCE_HANG_SEC = 0.8
BLOCK_SEC = 0.1
BLOCK_SAMPLES = int(SAMPLE_RATE * BLOCK_SEC)

# ----------------------------
# Server config
# ----------------------------


def rms_int16(x: np.ndarray) -> float:
    """Compute RMS volume of an int16 audio block."""
    xf = x.astype(np.float32)
    return float(np.sqrt(np.mean(xf * xf)))


def record_one_utterance() -> np.ndarray:
    """
    Wait for speech, then record until silence is detected (or max duration).
    Returns int16 audio array shaped (n_samples, 1).
    """
    frames: list[np.ndarray] = []
    in_speech = False
    speech_start = None
    last_voice_time = None

    def now() -> float:
        return time.time()

    print("\nListening for speech... (Ctrl+C to quit)")

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="int16") as stream:
        while True:
            block, _ = stream.read(BLOCK_SAMPLES)
            level = rms_int16(block)
            t = now()

            if level >= RMS_THRESHOLD:
                if not in_speech:
                    in_speech = True
                    speech_start = t
                last_voice_time = t
                frames.append(block.copy())
            else:
                if in_speech:
                    frames.append(block.copy())

            if in_speech and speech_start is not None:
                utterance_len = t - speech_start
                since_voice = t - (last_voice_time or t)

                if utterance_len >= MAX_UTTERANCE_SEC:
                    break

                if since_voice >= SILENCE_HANG_SEC and utterance_len >= MIN_SPEECH_SEC:
                    break

    if not frames:
        return np.zeros((0, 1), dtype=np.int16)

    return np.concatenate(frames, axis=0)



def main():
    engine = PiperSpeaker(voice=PiperVoiceUS.HFC_FEMALE)

    while True:
        user = input("Name: ")

        # 1) Record once
        audio = record_one_utterance()
        if audio.size == 0:
            print("Heard: [nothing]")
            return

        # 2) Transcribe once
        whisper_model = whisper.load_model("base")
        write(str(OUT_WAV), SAMPLE_RATE, audio)

        print("Transcribing...")
        result = whisper_model.transcribe(str(OUT_WAV))
        user_text = result.get("text", "").strip()

        if not user_text:
            print("Heard: [no speech detected]")
            return

        # 3) Send once
        data = {"user": user, "message": user_text}
        print(user_text)
        print("Sending to server...")

        try:
            response = requests.post(URL, json=data, timeout=60)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"Request failed: {e}")
            return

        # 4) Speak server reply
        print("Speaking server reply...")
        print(response.status_code)  # should be 200
        print(response.text)  # the "return" statement from the server

        text = response.text.encode().decode("unicode_escape")
        text = text.replace("\n", ". ")
        text = text.replace("\"", '"')

        engine.say(text)


if __name__ == "__main__":
    main()