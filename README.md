# EmotionBot

A Raspberry Pi-based system for real-time emotion recognition from spoken language, developed as part of a project assignment. The EmotionBot continuously listens for speech input, transcribes it using a pre-trained language model, and classifies the detected emotion, which is then visualized via animated emojis on a display.

---

## How It Works

For each speech input, the bot goes through four steps:

1. **Recording** – A USB microphone continuously records audio. Voice Activity Detection (VAD) is used to detect when someone is speaking.
2. **Transcription** – The recorded audio is converted to text using [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (an optimized implementation of OpenAI's Whisper model).
3. **Emotion Classification** – The transcribed text is passed to a fine-tuned DistilBERT model that predicts one of six emotions: `joy`, `sad`, `anger`, `fear`, `love`, `surprise`.
4. **Visualization** – The detected emotion is displayed as an animated emoji on the connected display via a Kivy-based GUI.

---

## Requirements

### Hardware

- **Raspberry Pi 5** (recommended, tested with 8 GB RAM)
- **USB Microphone**
- **Display** (tested with Waveshare 4.3" DSI LCD, 800×480)

### Software

- Python 3.10+
- pip

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/B4ddy/EmoBot
cd EmoBot
```

### 2. Download the emotion detection model
Unfortunately GitHub has a file size limit, so the models had to be hosted on Google Drive.
Please create a folder called "models":

```bash
mkdir -p models/emotion_classifier
cd models/emotion_classifier
```

Here is the project folder as it should look when complete. The model can be downloaded from the models folder and copied into your own project:
https://drive.google.com/drive/folders/1Qi__mDNT-VP3x5WpCeT3K-H0MzO4_V-j?usp=sharing

Then go back to the project directory:

```bash
cd ../..
```


### 3. Install system dependencies

```bash
sudo apt update
sudo apt install -y python3-pip python3-venv portaudio19-dev python3-pyaudio
```

### 4. Create a virtual environment with system package access

```bash
# Create the virtual environment with system package access
python3 -m venv venv --system-site-packages

# Activate the virtual environment
source venv/bin/activate
```
(You will know this worked because your terminal prompt will now start with (venv)).




### 5. Install PyTorch (CPU-optimized for Raspberry Pi)

```bash
pip3 install torch --index-url https://download.pytorch.org/whl/cpu
```

### 6. Install Python packages

```bash
pip3 install -r requirements.txt
```

---

## Running

```bash
# 1. Activate the environment (if not already active)
source venv/bin/activate

# 2. Run the bot
python3 main.py
```

On the first start, the AI models are loaded into memory — this typically takes around 15 seconds on the Raspberry Pi. After that, the bot is ready and waiting for speech input.

---

## Project Structure

```
EmotionBot/
├── main.py          # Entry point, connects all components
├── AI.py            # Transcription (Whisper) and emotion classification (BERT)
├── audio.py         # Microphone recording and Voice Activity Detection
├── UI.py            # Kivy GUI, emoji display and state animations
├── config.py        # Central configuration file
├── requirements.txt # Python dependencies
├── Emojis/          # Emoji images for each emotion
├── models/          # Local model weights (not included in the repository)
└── trainingcode/    # Code used to fine-tune the model
```

---

## Configuration

All relevant parameters are located in [`config.py`](config.py):

### Whisper model size (`WHISPER_MODEL_SIZE`)

| Value | Speed | Accuracy | RAM Usage |
|-------|-------|----------|-----------|
| `"tiny"` | very fast | low | ~75 MB |
| `"base"` | fast | moderate | ~150 MB |
| `"small"` | moderate | good | ~500 MB |
| `"medium"` | slow | very good | ~1.5 GB |

`"base"` has proven to be a good compromise for the Raspberry Pi 5.

### VAD aggressiveness (`VAD_AGGRESSIVENESS`)

Controls how sensitive the voice activity detection is to background noise:

- `0` – low aggressiveness, also picks up quiet speech
- `3` – very aggressive, reliably filters out background noise

### Silence threshold (`SILENCE_DURATION_MS`)

Specifies how long (in milliseconds) to wait after the last speech signal before ending the recording. Default value: `1000` ms.

---

## Libraries Used

| Library | Purpose | Link |
|---|---|---|
| [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | Speech recognition (ASR) | https://github.com/SYSTRAN/faster-whisper |
| [transformers](https://github.com/huggingface/transformers) | Emotion classification (DistilBERT) | https://huggingface.co/docs/transformers |
| [PyTorch](https://pytorch.org/) | Deep learning backend | https://pytorch.org/ |
| [webrtcvad](https://github.com/wiseman/py-webrtcvad) | Voice Activity Detection | https://github.com/wiseman/py-webrtcvad |
| [PyAudio](https://people.csail.mit.edu/hubert/pyaudio/) | Microphone access | https://people.csail.mit.edu/hubert/pyaudio/ |
| [Kivy](https://kivy.org/) | GUI framework | https://kivy.org/ |
| [NumPy](https://numpy.org/) | Audio data processing | https://numpy.org/ |

---

## Known Limitations

- The Whisper model expects English-language input (`language="en"`). The emotion model is also only trained on an English dataset. For other languages, the parameter in [`AI.py`](AI.py) must be adjusted.
- Emotion classification works at the text level — prosodic features (pitch, speaking rate) are not taken into account.
- On the Raspberry Pi, noticeable processing delays may occur for longer utterances, since all computations run on the CPU.

---

## Troubleshooting

**"No supported audio configuration found"**  
→ Check the microphone, try a different USB port if needed. Use `arecord -l` to list available recording devices.

**Models load very slowly**  
→ This is normal on the Raspberry Pi. A smaller Whisper model (`"tiny"` or `"base"`) can significantly reduce loading time.

**Emotion detection gives implausible results**  
→ The model was trained on English-language texts and works best with clear, expressive phrasing. Short or ambiguous utterances can lead to misclassifications.

**Too sensitive or not sensitive enough**  
→ On Raspberry Pi OS, you can click the microphone icon in the top-right corner to adjust the sensitivity. If no microphone icon is shown, the device is not being recognized.
