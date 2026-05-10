"""
Configuration File
==================
All settings for the EmotionBot are stored here.
You can change these values to adjust the behavior.
"""

import os

# AUDIO SETTINGS

# Audio format: 16-bit PCM (pyaudio.paInt16 = 8)
FORMAT = 8

# Channels: 1 = Mono (one microphone), 2 = Stereo (left and right)
CHANNELS = 1

# Sample rate: How many audio samples per second (48,000 is high quality)
RATE = 48000

# Chunk duration: Audio is processed in 30-millisecond pieces
# (Required by the voice activity detection library)
CHUNK_DURATION_MS = 30

# Voice Activity Detection (VAD) aggressiveness
# Scale: 0 (least aggressive) to 3 (most aggressive)
# Higher = filters background noise better, but may miss quiet speech
VAD_AGGRESSIVENESS = 3

# How long to wait in silence before stopping the recording? (in milliseconds)
# 1000 ms = 1 second
SILENCE_DURATION_MS = 2000



# AI MODEL SETTINGS

# Whisper model sizes: "tiny", "base", "small", "medium", "large"
# Larger = more accurate, but slower and requires more memory
# Recommendation for Raspberry Pi 5: "small" (good balance)
WHISPER_MODEL_SIZE = "tiny"

# Path to the emotion detection model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMOTION_MODEL_NAME = os.path.join(BASE_DIR, "models", "emotion_classifier")
