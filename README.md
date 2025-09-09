# Real-Time Deepgram Transcription

This script provides minimal, yet enhanced, near real-time microphone transcription using the Deepgram API. It captures audio from a microphone, sends it in chunks to Deepgram for transcription, and displays the results on the console.

## Features

*   **Near Real-Time Transcription**: Captures audio in chunks and transcribes it with low latency.
*   **VAD (Voice Activity Detection)**: Skips silent chunks to save API calls and processing time.
*   **Overlap Trimming**: Intelligently trims overlapping text between consecutive chunks to produce a clean, continuous transcript.
*   **Duplicate Suppression**: Removes stutters and repeated phrases.
*   **Resilient**: Includes robust error handling, graceful shutdown, and a fallback to raw HTTP requests if the Deepgram SDK fails.
*   **Configurable**: Allows customization of the model, language, chunk size, VAD threshold, and more via command-line arguments.
*   **File Logging**: Saves the full transcript to a timestamped text file.
*   **Context Management**: Optionally bounds the internal context to manage memory usage over long sessions.

## Dependencies

You can install the required Python libraries using pip:

```bash
pip install deepgram-sdk sounddevice numpy requests
```

## Configuration

Before running the script, you need to provide your Deepgram API key.

1.  Create a file named `api_config.json` in the same directory as the script.
2.  Add your API key to this file in the following format:

```json
{
  "deepgram_api_key": "YOUR_DEEPGRAM_API_KEY_HERE"
}
```

## Usage

To start transcribing, simply run the script:

```bash
python transcript_deepgram.py
```

The script will start capturing audio from your default microphone and print the transcribed text to the console. Press `Ctrl+C` to stop the script and save the final transcript.

### Command-Line Arguments

You can customize the behavior of the script using various command-line arguments:

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--config` | Path to the JSON config file. | `api_config.json` |
| `--model` | Deepgram model to use (e.g., `nova-3`, `nova-3-meeting`). | `nova-3` |
| `--language` | Language code for transcription. | `en` |
| `--chunk-duration` | Duration of each audio chunk in seconds. | `10.0` |
| `--overlap` | Fractional overlap between chunks (0.0 to 0.9). | `0.3` |
| `--vad-threshold`| RMS threshold for speech detection. Lower is more sensitive. | `0.005` |
| `--output` | Path to save the transcript file. | `transcript_{timestamp}.txt`|
| `--device` | Specify the input audio device by index or name. | `None` (default device) |
| `--no-sdk` | Force the script to use raw HTTP requests instead of the Deepgram SDK. | `False` |

**Example Usage:**

*   Use a more sensitive VAD and a 3-second chunk size:
    ```bash
    python transcript_deepgram.py --chunk-duration 3 --vad-threshold 0.006
    ```

*   Limit the internal transcript history to 4000 characters to save memory:
    ```bash
    python transcript_deepgram.py --context-max-chars 4000
    ```
