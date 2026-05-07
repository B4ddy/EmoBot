"""
User Interface
==============
Displays emoji animations on the screen to show the bot's state and emotions.
Uses the Kivy framework for the graphical interface.
"""

from kivy.app import App
from kivy.uix.image import Image
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.clock import mainthread, Clock
from Emojis.emoji_paths import EMOJI_PATHS


class KivyUserInterface(App):
    """
    Main application that displays emojis on the screen.
    """
    
    def build(self):
        """
        Sets up the user interface when the app starts.
        
        Returns:
            The main layout for displaying the emoji and close button
        """
        root = FloatLayout()
        
        # Create the image widget that will display emojis
        self.emoji_image = Image(
            source=EMOJI_PATHS["default"][0],
            fit_mode="contain"  # Scale image to fit without distortion
        )
        # root.add_widget(self.emoji_image)

        # NEW: Label for displaying Gemma text responses
        self.response_label = Label(
            text="",
            halign="center",
            valign="middle",
            size_hint=(1, 1),
        )
        self.response_label.bind(size=lambda inst, val: setattr(inst, "text_size", val))
        root.add_widget(self.response_label)

        close_button = Button(
            text="X",
            size_hint=(None, None),
            size=(48, 48),
            pos_hint={"right": 0.99, "top": 0.99}
        )
        close_button.bind(on_release=self.close_application)
        root.add_widget(close_button)
        
        # Animation state
        self.current_emotion = None
        self.emotion_index = 0
        self.animation_event = None
        
        return root
    
    @mainthread
    def show_state(self, state):
        """Display a single (non-animated) emoji for a bot state.
        
        Args:
            state: Key from EMOJI_PATHS, e.g. "default", "listening", "thinking"
        """
        self.stop_animation()
        self.emoji_image.source = EMOJI_PATHS[state][0]
        self.emoji_image.reload()

    @mainthread
    def show_emotion(self, emotion):
        """
        Display and animate an emotion emoji.
        
        Args:
            emotion: Name of the emotion ("joy", "sad", "anger", etc.)
        """
        self.stop_animation()
        self.current_emotion = emotion
        self.emotion_index = 0
        self.animate_emotion()

    def animate_emotion(self, dt=None):
        """
        Cycle through emotion images to create an animation effect.
        
        Args:
            dt: Delta time (provided by Kivy Clock, can be ignored)
        """
        if self.current_emotion and self.current_emotion in EMOJI_PATHS:
            # Get list of emoji images for this emotion
            emoji_images = EMOJI_PATHS[self.current_emotion]
            
            # Show current image
            self.emoji_image.source = emoji_images[self.emotion_index]
            self.emoji_image.reload()
            
            # Advance to next image (wrap back to start when at the end)
            self.emotion_index = (self.emotion_index + 1) % len(emoji_images)
            
            # Schedule next frame in 0.5 seconds
            self.animation_event = Clock.schedule_once(self.animate_emotion, 0.5)

    def stop_animation(self):
        """Stop any currently running emoji animation."""
        if self.animation_event:
            self.animation_event.cancel()
            self.animation_event = None
        self.current_emotion = None

    @mainthread
    def show_text(self, text):
        """
        Display a text response from Gemma on screen.

        Args:
            text: The response string to display
        """
        self.stop_animation()
        self.emoji_image.source = EMOJI_PATHS["default"][0]
        self.emoji_image.reload()
        self.response_label.text = text

    def close_application(self, _instance):
        """Closes the application via the X button."""
        self.stop()
