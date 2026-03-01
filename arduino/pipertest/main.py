import wave
from piper import PiperVoice
# Optional: Using specific synthesis settings
from piper import SynthesisConfig

config = SynthesisConfig(
    length_scale=1,  # Speed (higher is slower)
    noise_scale=0.667, # Phoneme variability
    volume=0.8         # Volume level
)

# 1. Load the voice model
# Make sure the .onnx and .onnx.json files are in the same folder
model_path = "en_US-bryce-medium.onnx"
voice = PiperVoice.load(model_path)

text = "Hello! I'm Peter and I like minors"
output_file = "output.wav"

# 2. Synthesize to a WAV file
with wave.open(output_file, "wb") as wav_file:
    voice.synthesize_wav(text, wav_file)

print(f"Audio saved to {output_file}")


# You can also stream audio chunks for real-time playback
# for audio_chunk in voice.synthesize(text):
#     # audio_chunk is raw int16 bytes
#     pass
