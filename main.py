"""
EmotionBot - Main Program
=========================
This is where everything starts! The program connects three main components:
1. AI (speech recognition and emotion detection)
2. User interface (displays emojis)
3. Audio (listens via the microphone)
"""

import threading
import os

# Kivy configuration MUST be set before any other Kivy imports
os.environ['KIVY_NO_CONSOLELOG'] = '1'
from kivy.config import Config
Config.set('kivy',     'log_level',   'warning')
Config.set('graphics', 'width',       '800')
Config.set('graphics', 'height',      '480')
Config.set('graphics', 'fullscreen',  '1')
Config.set('graphics', 'show_cursor', '0')


from UI import KivyUserInterface
from AI import AIProcessor
from audio import AudioEngine

def main():
    """
    The main function - this is where the EmotionBot is started.
    """
    print("EmotionBot starting...")
    print("Loading AI models (this may take 30 seconds on the Raspberry Pi)...")
    
    # Step 1: Initialize AI processor (loads speech and emotion models)
    ai_processor = AIProcessor()
    
    # Step 2: Initialize user interface (emoji display)
    UserInterface = KivyUserInterface()

    # Step 3: Define what should happen on audio events
    def on_speech_detected():
        """Called when the microphone hears someone speaking"""
        UserInterface.show_text("listening...")

    def on_processing_speech():
        """Called when the AI is analyzing the recorded speech"""
        UserInterface.show_text(" processing...")

    def on_speech_analyzed(result):
        text = result['text']
        print(f"You said: {text}")
        ai_processor.get_response(text, on_token=UserInterface.show_text)


    # Step 4: Create audio engine with our callback functions
    audio_engine = AudioEngine(
        on_speech_start=on_speech_detected,
        on_processing=on_processing_speech,
        on_transcription=on_speech_analyzed
    )

    # Step 5: Start listening in a separate thread
    # (This keeps the user interface responsive while we listen for speech)
    audio_thread = threading.Thread(
        target=audio_engine.start_listening,
        args=(ai_processor,),
        daemon=True  # Thread is terminated when the main program ends
    )
    audio_thread.start()

    # Step 6: Start the graphical user interface
    print("EmotionBot is ready! Start speaking...")
    UserInterface.run()


if __name__ == '__main__':
    main()
