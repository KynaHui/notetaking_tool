# Real-Time Transcription Scripts

This repository contains two Python scripts for real-time audio transcription.

1.  **`transcript_deepgram.py`**: Uses the Deepgram cloud API for fast and accurate transcription.
2.  **`transcript_local.py`**: Runs a local Whisper model, optimized for energy efficiency.

---

### 1. Deepgram Cloud Transcription (`transcript_deepgram.py`)

This script streams microphone audio to Deepgram's API for near real-time transcription.

**Setup:**

1.  **Install dependencies:**
    ```bash
    pip install deepgram-sdk sounddevice numpy requests
    ```

2.  **Add API Key:** Create a file named `api_config.json` and add your key:
    ```json
    {
      "deepgram_api_key": "YOUR_DEEPGRAM_API_KEY_HERE"
    }
    ```

**Usage:**

Run the script to start transcribing. Press `Ctrl+C` to stop.
```bash
python transcript_deepgram.py
```
*A transcript file will be saved automatically.*

---

### 2. Local Whisper Transcription (`transcript_local.py`)

This script uses `faster-whisper` to run transcription locally on your machine, with a focus on minimizing CPU and power usage.

**Setup:**

1.  **Install dependencies:**
    ```bash
    pip install faster-whisper sounddevice numpy psutil
    ```

**Usage:**

Run the script to start. The `base` model is used by default. Press `Ctrl+C` to stop.
```bash
python transcript_local.py
```
*A transcript file will be saved automatically.*

---

### Customization Examples

You can modify the behavior of both scripts with command-line arguments.

**Deepgram:**
```bash
# Use a 3-second chunk size and more sensitive voice detection
python transcript_deepgram.py --chunk-duration 3 --vad-threshold 0.006
```

**Local Whisper:**
```bash
# Use the 'tiny' model for lowest power usage
python transcript_local.py --model tiny

# Use a larger chunk size and enable low-power sleep mode
python transcript_local.py --chunk-duration 8 --low-power
```
