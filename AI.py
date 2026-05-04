"""
AI Processor
============
Handles two AI tasks:
1. Speech-to-Text: Converts spoken words into written text (using Whisper)
2. Emotion Detection: Analyzes the text to recognize the speaker's emotion
"""

import torch
import numpy as np
from faster_whisper import WhisperModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import WHISPER_MODEL_SIZE, EMOTION_MODEL_NAME


class AIProcessor:
    """
    Processes audio to extract text and detect emotions.
    """
    
    def __init__(self):
        """
        Loads the AI models at startup.
        This takes a moment, but only happens once when the program starts.
        """
        print("Loading Whisper speech recognition model...")
        # Use CPU mode with int8 to save memory on the Raspberry Pi
        self.whisper = WhisperModel(
            WHISPER_MODEL_SIZE, 
            device="cpu", 
            compute_type="int8"
        )
        
        print("Loading emotion detection model...")
        self.tokenizer = AutoTokenizer.from_pretrained(EMOTION_MODEL_NAME)
        self.emotion_model = AutoModelForSequenceClassification.from_pretrained(
            EMOTION_MODEL_NAME
        )
        
        # List of emotions the model can recognize
        self.emotions = ["sad", "joy", "love", "anger", "fear", "surprise"]
        
        print("AI models loaded successfully!")

    def detect_emotion(self, text):
        """
        Analyzes a text to determine which emotion it expresses.
        
        Args:
            text: The text to analyze (string)
            
        Returns:
            emotion: One of ["sad", "joy", "love", "anger", "fear", "surprise"]
        """
        # Convert the text into numbers the model understands
        inputs = self.tokenizer(text, return_tensors="pt")
        
        # Run the emotion detection model (without computing gradients)
        with torch.inference_mode():
            logits = self.emotion_model(**inputs).logits
        
        # Find out which emotion has the highest score
        prediction_index = torch.argmax(logits, dim=-1).item()
        detected_emotion = self.emotions[prediction_index]
        
        return detected_emotion

    def transcribe(self, audio_frames, sample_rate=16000):
        """
        Converts recorded audio into text. 
        
        Args:
            audio_frames: List of audio data chunks (bytes)
            sample_rate: How many samples per second in the audio
            
        Returns:
            dict: {"text": "transcribed text"}
        """
        # Combine all audio chunks into one piece
        audio_data = b"".join(audio_frames)
        
        # Convert bytes into numbers between -1.0 and 1.0
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Whisper expects 16kHz audio, so resample if necessary
        if sample_rate != 16000:
            duration = len(audio_array) / sample_rate
            target_length = int(duration * 16000)
            audio_array = np.interp(
                np.linspace(0, len(audio_array), target_length),
                np.arange(len(audio_array)),
                audio_array
            )
        
        # Use Whisper to convert speech to text
        segments, _ = self.whisper.transcribe(
            audio_array, 
            language="en", 
            beam_size=1
        )
        
        # Combine all segments into one text
        full_text = " ".join([segment.text for segment in segments])
        
        return {"text": full_text}

