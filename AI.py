"""
AI Processor
============
Handles two AI tasks:
1. Speech-to-Text: Converts spoken words into written text (using Whisper)
2. LLM Response: Sends transcribed text to Gemma 4 via llama-cpp-python
"""

# import torch                                                          # OLD: not needed anymore
# from transformers import AutoTokenizer, AutoModelForSequenceClassification  # OLD: emotion model
import numpy as np
from faster_whisper import WhisperModel
from llama_cpp import Llama
from config import WHISPER_MODEL_SIZE, GEMMA_MODEL_PATH


class AIProcessor:
    """
    Processes audio to extract text and get a Gemma 4 LLM response.
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

        # OLD: emotion detection model
        # print("Loading emotion detection model...")
        # self.tokenizer = AutoTokenizer.from_pretrained(EMOTION_MODEL_NAME)
        # self.emotion_model = AutoModelForSequenceClassification.from_pretrained(
        #     EMOTION_MODEL_NAME
        # )
        # self.emotions = ["sad", "joy", "love", "anger", "fear", "surprise"]

        print("Loading Gemma 4 model via llama-cpp...")
        self.llm = Llama(
            model_path=GEMMA_MODEL_PATH,
            n_ctx=2048,
            n_threads=4,
            verbose=False,
        )

        print("AI models loaded successfully!")

    # OLD: emotion detection via classifier
    # def detect_emotion(self, text):
    #     inputs = self.tokenizer(text, return_tensors="pt")
    #     with torch.inference_mode():
    #         logits = self.emotion_model(**inputs).logits
    #     prediction_index = torch.argmax(logits, dim=-1).item()
    #     return self.emotions[prediction_index]

    def get_response(self, text, on_token=None):
        """
        Sends transcribed text to Gemma 4 and streams the reply token by token.

        Args:
            text: The transcribed speech (string)
            on_token: optional callback(str) called with each new token chunk

        Returns:
            response: The full text reply (string)
        """
        stream = self.llm.create_chat_completion(
            messages=[{"role": "user", "content": text}],
            max_tokens=256,
            stream=True,
        )
        full = ""
        for chunk in stream:
            delta = chunk["choices"][0]["delta"].get("content", "")
            if delta:
                full += delta
                if on_token:
                    on_token(full)
        return full

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

