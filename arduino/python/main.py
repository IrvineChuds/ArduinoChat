import time
import threading
import numpy as np
import sounddevice as sd
import requests
import random

import cv2, json, time, threading
from dataclasses import dataclass
from collections import deque

from faster_whisper import WhisperModel

from yapper import PiperSpeaker, PiperVoiceUS

import bridge

@bridge.call()
def set_state(c: str) -> None:
    pass

@dataclass
class FaceState:
    name: str = "Unknown"
    dist: float = 9999.0
    last_update: float = 0.0

_face = FaceState()
_face_lock = threading.Lock()

def set_face(name: str, dist: float):
    with _face_lock:
        _face.name = name
        _face.dist = float(dist)
        _face.last_update = time.time()

def get_face() -> FaceState:
    with _face_lock:
        return FaceState(_face.name, _face.dist, _face.last_update)

def face_recognition_worker(stop_event: threading.Event):
    with open("labels.json", "r") as f:
        id_to_name = {int(k): v for k, v in json.load(f).items()}

    recognizer = cv2.face.LBPHFaceRecognizer.create()
    recognizer.read("trainer.yml")

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    UNKNOWN_THRESHOLD = 70.0
    STABLE_SEC = 0.3
    MIN_STABLE_FRAMES = 6
    HISTORY_SEC = 1.5

    last_announced = None
    candidate_name = None
    candidate_start = None
    history = deque()

    def stable_enough(now: float, name: str) -> bool:
        while history and (now - history[0][0]) > HISTORY_SEC:
            history.popleft()
        total = len(history)
        if total < MIN_STABLE_FRAMES:
            return False
        hits = sum(1 for _, n in history if n == name)
        return hits / total >= 0.75

    try:
        while not stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                continue

            now = time.time()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80)
            )
            if len(faces) == 0:
                continue

            (x, y, w, h) = max(faces, key=lambda b: b[2] * b[3])
            face_roi = cv2.resize(gray[y:y+h, x:x+w], (200, 200))

            label_id, dist = recognizer.predict(face_roi)
            if dist <= UNKNOWN_THRESHOLD:
                name = id_to_name.get(label_id, "Unknown")
            else:
                name = "Unknown"

            history.append((now, name))

            if candidate_name != name:
                candidate_name = name
                candidate_start = now

            if (now - candidate_start) >= STABLE_SEC and stable_enough(now, candidate_name):
                 if candidate_name != last_announced and candidate_name != "Unknown":
                    print(f"[face] {candidate_name} (dist={dist:.1f})")
                    set_face(candidate_name, dist)
                    last_announced = candidate_name
    finally:
        cap.release()

# URL = "http://11.28.48.208:8000/model"  # don't change this
URL = "https://unwordably-twinborn-wynell.ngrok-free.dev/model"  # don't change this

# ----------------------------
# Audio config
# ----------------------------
SAMPLE_RATE = 16000
CHANNELS = 1

#Tuning
RMS_THRESHOLD = 400            # still used, but we also allow dynamic thresholding
MIN_SPEECH_SEC = 0.4
MAX_UTTERANCE_SEC = 12.0
SILENCE_HANG_SEC = 0.8
BLOCK_SEC = 0.1
BLOCK_SAMPLES = int(SAMPLE_RATE * BLOCK_SEC)

# More tuning
START_TRIGGER_BLOCKS = 3       # need this many consecutive loud blocks to "start"
PREROLL_SEC = 0.5              # audio kept before trigger
MIN_VOICE_SEC = 0.25           # require at least this much loud audio before allowing stop
NOISE_CALIB_SEC = 0.6          # listen briefly to estimate background RMS
THRESH_MULT = 2.5              # dynamic threshold = max(RMS_THRESHOLD, noise_rms * THRESH_MULT)


def rms_int16(x: np.ndarray) -> float:
    xf = x.astype(np.float32)
    return float(np.sqrt(np.mean(xf * xf)))


def record_one_utterance(device=1) -> np.ndarray:
    """
    Wait for *real* speech, then record until silence is detected (or max duration).
    Adds: noise calibration, preroll, and 'consecutive loud blocks' start gating.
    """
    def now() -> float:
        return time.time()

    preroll_blocks = max(1, int(PREROLL_SEC / BLOCK_SEC))
    prebuf: deque[np.ndarray] = deque(maxlen=preroll_blocks)

    frames: list[np.ndarray] = []
    in_speech = False
    speech_start = None
    last_voice_time = None
    voice_blocks = 0
    consecutive_loud = 0

    print("\nListening for speech... (Ctrl+C to quit)")

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="int16") as stream:
        # --------- 1) Calibrate noise floor ----------
        calib_blocks = max(1, int(NOISE_CALIB_SEC / BLOCK_SEC))
        noise_levels = []
        for _ in range(calib_blocks):
            block, _ = stream.read(BLOCK_SAMPLES)
            prebuf.append(block.copy())
            noise_levels.append(rms_int16(block))

        noise_rms = float(np.median(noise_levels)) if noise_levels else 0.0
        dyn_threshold = max(RMS_THRESHOLD, noise_rms * THRESH_MULT)

        # --------- 2) Main loop ----------
        while True:
            block, _ = stream.read(BLOCK_SAMPLES)
            level = rms_int16(block)
            t = now()

            is_loud = level >= dyn_threshold

            if not in_speech:
                # Always keep preroll buffer
                prebuf.append(block.copy())

                # Require N consecutive loud blocks to start
                if is_loud:
                    consecutive_loud += 1
                else:
                    consecutive_loud = 0

                if consecutive_loud >= START_TRIGGER_BLOCKS:
                    in_speech = True
                    speech_start = t
                    last_voice_time = t
                    voice_blocks = consecutive_loud

                    # include preroll + current block(s)
                    frames.extend(list(prebuf))
                    frames.append(block.copy())
                    prebuf.clear()

            else:
                # We are in speech: record everything
                frames.append(block.copy())

                if is_loud:
                    last_voice_time = t
                    voice_blocks += 1

                utterance_len = t - (speech_start or t)
                since_voice = t - (last_voice_time or t)
                voice_sec = voice_blocks * BLOCK_SEC

                # Stop conditions
                if utterance_len >= MAX_UTTERANCE_SEC:
                    break

                # Only allow "silence hang" to stop if we've had enough actual voice
                if voice_sec >= MIN_VOICE_SEC and since_voice >= SILENCE_HANG_SEC and utterance_len >= MIN_SPEECH_SEC:
                    break

    if not frames:
        return np.zeros((0, CHANNELS), dtype=np.int16)

    return np.concatenate(frames, axis=0)

def main():
    engine = PiperSpeaker(voice=PiperVoiceUS.HFC_FEMALE)
    whisper_model = WhisperModel("tiny.en", device="auto")

    stop_event = threading.Event()
    face_thread = threading.Thread(
        target=face_recognition_worker, args=(stop_event,), daemon=True
    )
    face_thread.start()

    print("Waiting 15 seconds for face recognition to start")
    time.sleep(15)

    try:
        while True:
            # 1) Record once
            set_state('l')
            audio = record_one_utterance()
            if audio.size == 0:
                print("Heard: [nothing]")
                break

            set_state('k')

            face = get_face()
            face_age = time.time() - face.last_update if face.last_update else None

            # 2) Transcribe once
            audio_float = audio.flatten().astype(np.float32) / 32768.0
            print("Transcribing...")
            segments, _ = whisper_model.transcribe(audio_float)
            user_text = " ".join(seg.text for seg in segments).strip()

            if not user_text:
                print("Heard: [no speech detected]")
                break

            # 3) Sending Data
            data = {
                "user": face.name,
                "message": user_text,
            }

            print(user_text)
            print(f"Using face at send time: {face.name} (age={face_age})")
            print("Sending to server...")

            try:
                response = requests.post(URL, json=data)
                response.raise_for_status()
            except requests.RequestException as e:
                print(f"Request failed: {e}")
                break

            # 4) Speak server reply
            print("Speaking server reply...")
            text = response.text.encode().decode("unicode_escape")
            text = text.replace("\n", ". ").replace("\"", '"')

            state_choices = ['a', 'c', 'g', 'i']
            set_state(random.choice(state_choices))
            engine.say(text)

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        stop_event.set()
        time.sleep(0.2)  # let face thread exit

if __name__ == "__main__":
    main()
