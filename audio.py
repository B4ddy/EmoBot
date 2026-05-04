"""
Audio Engine
============
Listens to the microphone and detects when someone is speaking.
Uses Voice Activity Detection (VAD) to know when speech starts and ends.
"""

import pyaudio
import webrtcvad
import time
from config import (
    FORMAT, CHANNELS, RATE,
    VAD_AGGRESSIVENESS, SILENCE_DURATION_MS
)


class AudioEngine:
    """
    Manages microphone input and speech detection.
    """
    
    def __init__(self, on_speech_start=None, on_processing=None, on_transcription=None):
        """
        Initializes the audio engine with callback functions.
        
        Args:
            on_speech_start: Function called when speech is detected
            on_processing: Function called when processing begins
            on_transcription: Function called with results (text, emotion)
        """
        # Voice Activity Detector - detects whether audio contains speech
        self.vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        
        # Store callback functions
        self.on_speech_start = on_speech_start
        self.on_processing = on_processing
        self.on_transcription = on_transcription
        
        # Control flag for the listening loop
        self.is_running = True
        
        # Actual sample rate (may differ from config if hardware doesn't support it)
        self.actual_rate = RATE

    def _find_working_audio_config(self, audio_interface):
        """
        Tries different sample rates to find one that works with your microphone.
        
        Args:
            audio_interface: PyAudio instance
            
        Returns:
            tuple: (sample_rate, chunk_size) that works
        """
        # WebRTC VAD only supports these specific sample rates
        supported_rates = [16000, 48000, 32000, 8000]
        
        for rate in supported_rates:
            try:
                # Try to open an audio stream with this rate
                test_stream = audio_interface.open(
                    format=FORMAT,
                    channels=CHANNELS,
                    rate=rate,
                    input=True,
                    frames_per_buffer=int(rate * 30 / 1000)  # 30ms chunks
                )
                test_stream.close()
                
                print(f" Using sample rate: {rate} Hz")
                return rate, int(rate * 30 / 1000)
                
            except Exception as e:
                print(f" Sample rate {rate} Hz is not supported: {e}")
                continue
        
        # If we get here, no sample rate worked
        raise RuntimeError(
            "No supported audio configuration found. "
            "Please check your microphone connection."
        )

    def start_listening(self, ai_processor):
        """
        Main listening loop - continuously monitors the microphone for speech.
        
        Args:
            ai_processor: AIProcessor instance for analyzing recorded speech
        """
        # Initialize PyAudio
        audio_interface = pyaudio.PyAudio()
        
        # Show available microphones for debugging
        print("\nAvailable microphones:")
        for i in range(audio_interface.get_device_count()):
            device_info = audio_interface.get_device_info_by_index(i)
            if device_info['maxInputChannels'] > 0:
                print(f"  Device {i}: {device_info['name']}")
        
        # Find a sample rate that works with the microphone
        try:
            self.actual_rate, chunk_size = self._find_working_audio_config(audio_interface)
        except RuntimeError as e:
            print(f"Error: {e}")
            audio_interface.terminate()
            return
        
        # Open the microphone stream
        try:
            stream = audio_interface.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=self.actual_rate,
                input=True,
                frames_per_buffer=chunk_size
            )
        except Exception as e:
            print(f"Could not open microphone: {e}")
            audio_interface.terminate()
            return

        # Storage for recorded audio chunks
        recorded_frames = []
        
        # Status tracking
        is_recording = False
        silence_start_time = None

        try:
            # Main listening loop
            while self.is_running:
                try:
                    # Read one audio chunk from the microphone
                    audio_chunk = stream.read(chunk_size, exception_on_overflow=False)
                except OSError:
                    # Ignore buffer overflow errors
                    continue

                # Check whether this chunk contains speech
                contains_speech = self.vad.is_speech(audio_chunk, self.actual_rate)
                
                if contains_speech:
                    # Speech detected!
                    if not is_recording:
                        # This is the beginning of new speech
                        if self.on_speech_start:
                            self.on_speech_start()
                        is_recording = True
                    
                    # Save this audio chunk
                    recorded_frames.append(audio_chunk)
                    silence_start_time = None
                    
                elif is_recording:
                    # No speech in this chunk, but we are still recording
                    recorded_frames.append(audio_chunk)
                    
                    # Track how long the silence has lasted
                    if silence_start_time is None:
                        silence_start_time = time.time()
                    else:
                        silence_duration = (time.time() - silence_start_time) * 1000
                        
                        # If the silence was long enough, stop recording
                        if silence_duration > SILENCE_DURATION_MS:
                            # Notify that we are processing
                            if self.on_processing:
                                self.on_processing()
                            
                            
                            
                            # Send result back via callback
                            if self.on_transcription:
                                result = ai_processor.transcribe(
                                    list(recorded_frames), 
                                    self.actual_rate
                                )
                                self.on_transcription(result)

                            
                            # Reset for next recording
                            recorded_frames.clear()
                            is_recording = False
                            silence_start_time = None
                            
        finally:
            # Clean up audio resources
            stream.stop_stream()
            stream.close()
            audio_interface.terminate()
