#!/usr/bin/env python
"""
Minimal (enhanced) Deepgram microphone transcription (chunked, near real-time).

Enhancements vs original draft:
  - Safer Deepgram response parsing (SDK + HTTP fallback)
  - File initialization & thread‑safe appends
  - Overlap trimming & duplicate suppression (trim_overlap, collapse_internal_repeats)
  - Config + CLI arguments (model selection, VAD threshold, chunk length, overlap)
  - Graceful shutdown (Ctrl+C) with clean thread join
  - Optional context truncation to bound memory
  - Robust handling of empty / malformed responses
  - Optional diarization / utterances flags (placeholders if you want to extend)
  - Logging with timestamps
  - Python 3.10+ type hints (| union operator)
  - Improved SDK parsing (no automatic HTTP fallback on parse shape changes)

Dependencies:
  pip install deepgram-sdk sounddevice numpy requests

Config file (api_config.json):
{
  "deepgram_api_key": "YOUR_KEY_HERE"
}

Usage examples:
  python transcript_deepgram_revised.py
  python transcript_deepgram_revised.py --chunk-duration 3 --vad-threshold 0.006 --model nova-3
  python transcript_deepgram_revised.py --overlap 0.25 --context-max-chars 4000

Press Ctrl+C to stop.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import signal
import sys
import threading
import time
import wave
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from typing import Optional, List, Any

import numpy as np
import sounddevice as sd
import requests

# Optional Deepgram SDK
sdk_available = True
try:
    from deepgram import DeepgramClient, PrerecordedOptions
except Exception as e:
    sdk_available = False
    sdk_import_error = e


# ----------------------------
# Utility / Helper Functions
# ----------------------------

def log(msg: str) -> None:
    """Timestamped stderr logging."""
    ts = datetime.now().strftime("%H:%M:%S")
    sys.stderr.write(f"[{ts}] {msg}\n")
    sys.stderr.flush()


def load_config(path: str) -> dict:
    if not os.path.exists(path):
        log(f"ERROR: Config file '{path}' not found. Create it with your Deepgram API key:")
        log('{"deepgram_api_key": "YOUR_KEY_HERE"}')
        sys.exit(1)
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        if "deepgram_api_key" not in cfg or not cfg["deepgram_api_key"]:
            raise ValueError("Missing 'deepgram_api_key' in config.")
        return cfg
    except json.JSONDecodeError as e:
        log(f"ERROR: Invalid JSON in {path}: {e}")
        sys.exit(1)
    except Exception as e:
        log(f"ERROR: {e}")
        sys.exit(1)


def float_audio_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """Convert float32 mono [-1,1] audio to WAV bytes."""
    audio = np.clip(audio, -1.0, 1.0)
    int16 = (audio * 32767).astype(np.int16)
    buf = BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int16.tobytes())
    return buf.getvalue()


def trim_overlap(prev_text: str,
                 new_text: str,
                 min_overlap: int = 12,
                 max_window: int = 180) -> str:
    """
    Remove duplicated prefix of new_text that already exists as a suffix of prev_text.
    Exact match search (case-insensitive) up to max_window chars of prev_text.
    """
    if not prev_text or not new_text:
        return new_text

    prev_slice = prev_text[-max_window:]
    new_slice = new_text[:max_window]

    prev_low = prev_slice.lower()
    new_low = new_slice.lower()

    longest = 0
    max_len = min(len(prev_low), len(new_low))
    for k in range(max_len, min_overlap - 1, -1):
        if prev_low.endswith(new_low[:k]):
            longest = k
            break

    if longest >= min_overlap:
        return new_text[longest:].lstrip()
    return new_text


def collapse_internal_repeats(text: str) -> str:
    """
    1. Collapse immediate duplicate words (case-insensitive).
    2. Remove duplicated starting clause (4–10 tokens).
    """
    if not text:
        return text
    words = text.split()
    collapsed: List[str] = []
    last_low: Optional[str] = None
    for w in words:
        lw = w.lower()
        if lw == last_low:
            continue
        collapsed.append(w)
        last_low = lw
    text2 = " ".join(collapsed)

    tokens = text2.split()
    for span in range(10, 3, -1):
        if len(tokens) >= span * 2:
            first = [t.lower() for t in tokens[:span]]
            second = [t.lower() for t in tokens[span:span * 2]]
            if first == second:
                text2 = " ".join(tokens[span:])
                break
    return text2


def normalize_spacing(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


# ----------------------------
# Deepgram Transcriber
# ----------------------------

class DeepgramChunkTranscriber:
    def __init__(
        self,
        api_key: str,
        model: str = "nova-3",
        language: str = "en",
        smart_format: bool = True,
        punctuate: bool = True,
        timeout: float = 30.0,
        use_sdk: bool = True
    ):
        self.api_key = api_key
        # Model default changed to nova-3 (upgrade)
        self.model = model
        self.language = language
        self.smart_format = smart_format
        self.punctuate = punctuate
        self.timeout = timeout
        self.use_sdk = use_sdk and sdk_available
        self._dg: Optional[DeepgramClient] = None
        self._sdk_shape_warned = False  # optional informational flag

        if self.use_sdk:
            try:
                self._dg = DeepgramClient(self.api_key)
            except Exception as e:
                log(f"WARN: Failed to init Deepgram SDK ({e}). Falling back to raw HTTP.")
                self.use_sdk = False

        if not self.use_sdk and not sdk_available:
            log(f"INFO: SDK not available ({sdk_import_error}); using HTTP fallback.")

        self._http_session = requests.Session()
        self._base_url = "https://api.deepgram.com/v1/listen"

    # ---- New helper function for robust parsing ----
    def _extract_transcript_from_sdk_response(self, response: Any) -> str:
        """
        Attempt to extract transcript from Deepgram SDK response without assuming
        a fixed internal representation. Strategy:

        1. If already a dict, use directly.
        2. Try common serialization methods: to_dict(), model_dump(), json().
        3. If still not a dict, attempt attribute traversal:
           response.results.channels[0].alternatives[0].transcript
        4. Return "" on failure (do NOT raise).
        """
        data = None

        # Step 1: direct dict
        if isinstance(response, dict):
            data = response
        else:
            # Step 2: serialization methods
            for attr_name in ("to_dict", "model_dump"):
                if hasattr(response, attr_name):
                    try:
                        data = getattr(response, attr_name)()
                        break
                    except Exception:
                        pass
            if data is None and hasattr(response, "json"):
                try:
                    raw = response.json()
                    if isinstance(raw, str):
                        try:
                            data = json.loads(raw)
                        except Exception:
                            pass
                    elif isinstance(raw, dict):
                        data = raw
                except Exception:
                    pass

        # Step 3: attribute traversal if still no dict
        if data is None:
            try:
                res = getattr(response, "results", None)
                if res is None:
                    return ""
                channels = getattr(res, "channels", None)
                if not channels:
                    return ""
                first_channel = channels[0]
                alts = getattr(first_channel, "alternatives", None)
                if not alts:
                    return ""
                first_alt = alts[0]
                transcript = getattr(first_alt, "transcript", "") or ""
                return transcript
            except Exception:
                return ""

        # Step 4: dict style extraction
        try:
            results = data.get("results") or {}
            channels = results.get("channels") or []
            if channels:
                alts = channels[0].get("alternatives") or []
                if alts:
                    return alts[0].get("transcript", "") or ""
            return ""
        except Exception:
            return ""

    def transcribe_wav_bytes(self, wav_bytes: bytes) -> str:
        """Send a WAV chunk to Deepgram and return transcript (may be empty)."""
        if self.use_sdk:
            try:
                source = {"buffer": wav_bytes, "mimetype": "audio/wav"}
                options = PrerecordedOptions(
                    model=self.model,
                    language=self.language,
                    smart_format=self.smart_format,
                    punctuate=self.punctuate,
                )
                response = self._dg.listen.rest.v("1").transcribe_file(source, options)
                transcript = self._extract_transcript_from_sdk_response(response)
                return transcript  # empty string is fine (silence or no speech)
            except Exception as e:
                log(f"WARN: SDK request error ({e}). Using HTTP fallback for this chunk.")
                return self._transcribe_http(wav_bytes)
        # SDK disabled or unavailable
        return self._transcribe_http(wav_bytes)

    def _transcribe_http(self, wav_bytes: bytes) -> str:
        params = {
            "model": self.model,
            "language": self.language,
            "smart_format": "true" if self.smart_format else "false",
            "punctuate": "true" if self.punctuate else "false",
        }
        headers = {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": "audio/wav",
        }
        try:
            resp = self._http_session.post(
                self._base_url,
                params=params,
                data=wav_bytes,
                headers=headers,
                timeout=self.timeout,
            )
            if resp.status_code != 200:
                snippet = resp.text[:300].replace("\n", " ")
                log(f"ERROR: HTTP {resp.status_code}: {snippet}")
                return ""
            data = resp.json()
            try:
                results = data.get("results", {})
                channels = results.get("channels") or []
                if channels:
                    alts = channels[0].get("alternatives") or []
                    if alts:
                        return alts[0].get("transcript", "") or ""
                return ""
            except Exception as pe:
                log(f"WARN: Parse error in HTTP JSON ({pe}).")
                return ""
        except requests.RequestException as e:
            log(f"ERROR: HTTP request failed: {e}")
            return ""


# ----------------------------
# Audio Capture Manager
# ----------------------------

@dataclass
class CaptureStats:
    total_chunks: int = 0
    skipped_silence: int = 0
    empty_results: int = 0
    non_empty: int = 0


class AudioCaptureManager:
    def __init__(
        self,
        transcriber: DeepgramChunkTranscriber,
        chunk_duration: float = 10.0,
        sample_rate: int = 16000,
        vad_threshold: float = 0.005,
        overlap: float = 0.3,
        max_queue: int = 20,
        output_path: str | None = None,
        min_transcript_len: int = 1,
        overlap_min_chars: int = 12,
        overlap_max_window: int = 180,
        context_max_chars: int | None = None,
        device: str | int | None = None
    ):
        self.transcriber = transcriber
        self.sample_rate = sample_rate
        self.chunk_samples = int(chunk_duration * sample_rate)
        self.overlap_samples = int(overlap * self.chunk_samples) if 0 < overlap < 0.95 else 0
        self.vad_threshold = vad_threshold
        self.min_transcript_len = min_transcript_len
        self.overlap_min_chars = overlap_min_chars
        self.overlap_max_window = overlap_max_window
        self.context_max_chars = context_max_chars
        self.device = device

        self.audio_queue: queue.Queue[np.ndarray] = queue.Queue(max_queue)
        self.stop_event = threading.Event()
        self.process_thread: Optional[threading.Thread] = None

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.output_file = output_path or f"transcript_{timestamp}.txt"
        self._file_lock = threading.Lock()

        # Buffers
        self._capture_buffer = np.array([], dtype=np.float32)
        self._last_transcript = ""
        self._accumulated_text = ""

        self.stats = CaptureStats()

        self._init_output_file()

    def _init_output_file(self):
        with open(self.output_file, "w", encoding="utf-8") as f:
            f.write(f"# Transcript started {datetime.now().isoformat()}\n")

    def _write_transcript(self, line: str):
        with self._file_lock:
            with open(self.output_file, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            log(f"Audio status: {status}")
        # indata is (frames, channels)
        mono = indata[:, 0] if indata.ndim > 1 else indata
        mono = mono.astype(np.float32)
        self._capture_buffer = np.concatenate([self._capture_buffer, mono])
        # Create chunks
        while self._capture_buffer.shape[0] >= self.chunk_samples:
            chunk = self._capture_buffer[:self.chunk_samples]
            # Keep overlap tail in buffer, remove consumed region except overlap
            if self.overlap_samples > 0:
                tail = self._capture_buffer[self.chunk_samples - self.overlap_samples:self.chunk_samples]
                self._capture_buffer = np.concatenate([tail, self._capture_buffer[self.chunk_samples:]])
            else:
                self._capture_buffer = self._capture_buffer[self.chunk_samples:]
            try:
                self.audio_queue.put_nowait(chunk.copy())
            except queue.Full:
                log("WARN: Chunk queue full; dropping chunk.")

    def _rms(self, audio: np.ndarray) -> float:
        if audio.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(audio))))

    def start(self):
        if self.process_thread and self.process_thread.is_alive():
            log("WARN: Processing thread already running.")
            return
        self.stop_event.clear()
        self.process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.process_thread.start()
        log("INFO: Started processing thread.")

        # Start audio input stream
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            callback=self._audio_callback,
            blocksize=0,  # Let sounddevice choose
            device=self.device
        )
        self.stream.start()
        log("INFO: Audio stream started.")

    def stop(self):
        self.stop_event.set()
        log("INFO: Stopping capture...")
        try:
            if hasattr(self, "stream"):
                self.stream.stop()
                self.stream.close()
        except Exception as e:
            log(f"WARN: Error stopping stream: {e}")
        if self.process_thread:
            self.process_thread.join(timeout=5)
            if self.process_thread.is_alive():
                log("WARN: Processing thread did not exit cleanly.")
        log("INFO: Stopped.")

    def _truncate_context_if_needed(self):
        if self.context_max_chars and len(self._accumulated_text) > self.context_max_chars:
            self._accumulated_text = self._accumulated_text[-self.context_max_chars:]

    def _process_loop(self):
        while not self.stop_event.is_set():
            try:
                chunk = self.audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            self.stats.total_chunks += 1
            rms = self._rms(chunk)
            if rms < self.vad_threshold:
                self.stats.skipped_silence += 1
                continue

            wav_bytes = float_audio_to_wav_bytes(chunk, self.sample_rate)
            text = self.transcriber.transcribe_wav_bytes(wav_bytes)
            if not text:
                self.stats.empty_results += 1
                continue

            raw = text.strip()
            if raw and len(raw) >= self.min_transcript_len:
                trimmed = trim_overlap(self._accumulated_text,
                                       raw,
                                       min_overlap=self.overlap_min_chars,
                                       max_window=self.overlap_max_window)
                cleaned = normalize_spacing(collapse_internal_repeats(trimmed))
                if cleaned and cleaned != self._last_transcript:
                    self._last_transcript = cleaned
                    if self._accumulated_text:
                        self._accumulated_text += " " + cleaned
                    else:
                        self._accumulated_text = cleaned
                    self._truncate_context_if_needed()
                    self.stats.non_empty += 1
                    self._write_transcript(cleaned)
                    print(cleaned, flush=True)  # Console real-time output

    def summarize(self):
        return {
            "total_chunks": self.stats.total_chunks,
            "skipped_silence": self.stats.skipped_silence,
            "empty_results": self.stats.empty_results,
            "non_empty_chunks": self.stats.non_empty
        }


# ----------------------------
# CLI / Main
# ----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Deepgram microphone chunked transcriber (nova-3 default).")
    p.add_argument("--config", default="api_config.json", help="Path to JSON config with deepgram_api_key.")
    p.add_argument("--model", default="nova-3", help="Deepgram model (e.g. nova-3, nova-3-meeting).")
    p.add_argument("--language", default="en", help="Language code.")
    p.add_argument("--chunk-duration", type=float, default=10.0, help="Seconds per chunk.")
    p.add_argument("--overlap", type=float, default=0.3, help="Fractional overlap between chunks (0.0-0.9).")
    p.add_argument("--vad-threshold", type=float, default=0.005, help="RMS threshold for speech detection.")
    p.add_argument("--sample-rate", type=int, default=16000, help="Microphone sample rate.")
    p.add_argument("--output", default=None, help="Transcript output file (default: timestamped).")
    p.add_argument("--min-transcript-len", type=int, default=1, help="Minimum transcript length to accept.")
    p.add_argument("--overlap-min-chars", type=int, default=12, help="Min chars to consider as overlap match.")
    p.add_argument("--overlap-max-window", type=int, default=180, help="Max window size for overlap detection.")
    p.add_argument("--context-max-chars", type=int, default=6000, help="Limit accumulated context (chars). 0=unlimited.")
    p.add_argument("--no-smart-format", action="store_true", help="Disable smart_format.")
    p.add_argument("--no-punctuate", action="store_true", help="Disable punctuation.")
    p.add_argument("--no-sdk", action="store_true", help="Force HTTP fallback instead of SDK.")
    p.add_argument("--device", default=None, help="Sounddevice input device index or name.")
    p.add_argument("--timeout", type=float, default=30.0, help="HTTP/SDK timeout per request.")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    api_key = cfg["deepgram_api_key"]

    context_max = args.context_max_chars if args.context_max_chars > 0 else None

    transcriber = DeepgramChunkTranscriber(
        api_key=api_key,
        model=args.model,
        language=args.language,
        smart_format=not args.no_smart_format,
        punctuate=not args.no_punctuate,
        timeout=args.timeout,
        use_sdk=not args.no_sdk
    )

    cap = AudioCaptureManager(
        transcriber=transcriber,
        chunk_duration=args.chunk_duration,
        sample_rate=args.sample_rate,
        vad_threshold=args.vad_threshold,
        overlap=args.overlap,
        output_path=args.output,
        min_transcript_len=args.min_transcript_len,
        overlap_min_chars=args.overlap_min_chars,
        overlap_max_window=args.overlap_max_window,
        context_max_chars=context_max,
        device=args.device
    )

    # Graceful shutdown via signal
    stopping = {"flag": False}

    def handle_sigint(signum, frame):
        if stopping["flag"]:
            return
        stopping["flag"] = True
        print("\nStopping...", flush=True)
        cap.stop()
        summary = cap.summarize()
        log(f"Session summary: {summary}")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_sigint)
    signal.signal(signal.SIGTERM, handle_sigint)

    log("INFO: Starting capture. Press Ctrl+C to stop.")
    cap.start()

    # Idle loop
    try:
        while True:
            time.sleep(0.8)
    except KeyboardInterrupt:
        handle_sigint(None, None)


if __name__ == "__main__":
    # Sanity: advise Python version
    if sys.version_info < (3, 10):
        log("WARN: Python 3.10+ recommended for '|' union type hints.")
    main()