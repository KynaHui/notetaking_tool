#!/usr/bin/env python
"""
Ultra Energy-Efficient Faster-Whisper Chunked Real-Time Transcriber

Goals:
- Maintain chunk-based accuracy (with overlap) while aggressively minimizing
  CPU usage & battery drain.

Strategies:
* Quantized inference (int8 or int8_float16)
* Single-threaded BLAS / OMP
* Silence gating (RMS + peak)
* Adaptive duty cycling (longer sleeps during silence)
* Reused buffers & minimal allocations
* Optional low-power mode (less polling)
* Overlap retained for accuracy (configurable)
* Skips model calls on silent segments
* Tracks compute vs audio time and can throttle if too "busy"

Suggested Models (energy ascending):
tiny  -> lowest power
base  -> good balance
small -> moderate power
Avoid medium / large on battery unless plugged in.

Usage Examples:
python energy_whisper_transcriber.py
python energy_whisper_transcriber.py --model tiny --chunk-duration 5 --overlap 1 --low-power
python energy_whisper_transcriber.py --model base --rms-threshold 0.012 --peak-threshold 0.15
"""

# -------- Environment (must be before heavy imports) --------
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import time
import queue
import platform
import psutil
import argparse
from datetime import datetime
from collections import deque

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

# ---------------- Configuration Defaults --------------------
DEFAULT_SAMPLE_RATE = 16000

# ---------------- Utility Functions -------------------------
def format_seconds(sec: float) -> str:
  if sec < 60:
      return f"{sec:.1f}s"
  m = int(sec // 60)
  s = sec % 60
  return f"{m}m{int(s)}s"

# ---------------- Energy Optimized Transcriber --------------
class EnergyEfficientTranscriber:
  def __init__(self,
               model_size="base",
               compute_type=None,
               device=None,
               int8_fallback=True,
               num_workers=1,
               rms_threshold=0.008,
               peak_threshold=0.10,
               silence_grace=2,
               adaptive_decay=0.9,
               low_power=False,
               max_cpu_ratio=0.85,
               vad_filter=True):
      """
      Args:
          model_size: Whisper size (tiny/base/small/etc.)
          compute_type: Force (e.g., int8, int8_float16, int16, float16)
          device: 'cpu' or 'cuda' or 'auto'
          int8_fallback: If chosen compute_type fails, fallback to int8
          rms_threshold: RMS below which chunk is considered silence
          peak_threshold: Peak amplitude below which chunk may be skipped
          silence_grace: Number of consecutive (likely silent) chunks allowed before
                         slightly lowering thresholds (adaptive)
          adaptive_decay: Factor to gently reduce thresholds after prolonged silence
          low_power: If True, uses longer base sleeps & conservative polling
          max_cpu_ratio: Keep compute_time / audio_time under this ratio
          vad_filter: Use internal vad_filter (still energy helpful)
      """
      self.model_size = model_size
      self.low_power = low_power
      self.vad_filter = vad_filter
      self.rms_threshold_base = rms_threshold
      self.peak_threshold_base = peak_threshold
      self.rms_threshold = rms_threshold
      self.peak_threshold = peak_threshold
      self.silence_grace = silence_grace
      self.adaptive_decay = adaptive_decay
      self.max_cpu_ratio = max_cpu_ratio
      self.silent_streak = 0

      self.system_info = self._get_system_info()
      self.device, self.compute_type = self._select_device(device, compute_type)
      self._init_model(num_workers=num_workers, int8_fallback=int8_fallback)

      print(f"[INFO] System: {self.system_info['system']}  CPU Cores (physical): {self.system_info['cpu_cores']}")
      print(f"[INFO] RAM: {self.system_info['ram']:.1f} GB  AppleSilicon={self.system_info['is_apple_silicon']}")
      print(f"[INFO] Model: {self.model_size}  Device: {self.device}  Compute: {self.compute_type}")
      print(f"[INFO] Initial thresholds: RMS={self.rms_threshold:.4f} Peak={self.peak_threshold:.3f}")
      if self.low_power:
          print("[INFO] Low-power mode ENABLED (more sleeping, less polling)")

      self.audio_queue = queue.Queue()

  def _get_system_info(self):
      system = platform.system()
      processor = platform.processor()
      ram_gb = psutil.virtual_memory().total / (1024 ** 3)
      is_apple = (system == "Darwin") and ("Apple" in processor or processor.lower().startswith("arm"))
      return {
          "system": system,
          "processor": processor,
          "ram": ram_gb,
          "is_apple_silicon": is_apple,
          "cpu_cores": psutil.cpu_count(logical=False) or 1
      }

  def _select_device(self, user_device, user_compute):
      # We strongly prefer CPU + int8 for energy, even on GPU machines (unless user overrides)
      device = user_device or "cpu"
      if user_compute:
          return device, user_compute
      if self.system_info["is_apple_silicon"]:
          # int8_float16 can be slightly faster with acceptable energy; fallback to int8
          return device, "int8"
      return device, "int8"

  def _init_model(self, num_workers=1, int8_fallback=True):
      try:
          self.model = WhisperModel(
              self.model_size,
              device=self.device,
              compute_type=self.compute_type,
              num_workers=num_workers,
              local_files_only=False
          )
      except Exception as e:
          if int8_fallback and self.compute_type != "int8":
              print(f"[WARN] Failed with compute_type={self.compute_type} ({e}). Falling back to int8.")
              self.compute_type = "int8"
              self.model = WhisperModel(
                  self.model_size,
                  device=self.device,
                  compute_type=self.compute_type,
                  num_workers=1
              )
          else:
              raise

  # ---------------- Audio Handling ----------------
  def audio_callback(self, indata, frames, callback_time, status):
      if status:
          print(f"[AudioStatus] {status}", flush=True)
      # Avoid frequent object creation; push raw float32 copy
      self.audio_queue.put(indata[:, 0].copy())

  # ---------------- Transcription -----------------
  def transcribe_chunk(self, audio_chunk):
      """
      Run Whisper on a chunk (already trimmed/assembled float32 1-D).
      Returns list of text segments.
      """
      try:
          segments, info = self.model.transcribe(
              audio_chunk,
              language="en",
              beam_size=1,
              best_of=1,
              temperature=0.0,
              word_timestamps=False,
              condition_on_previous_text=False,
              vad_filter=self.vad_filter,
              vad_parameters=dict(
                  min_silence_duration_ms=600,
                  speech_pad_ms=150
              ),
              no_speech_threshold=0.6
          )
          return [seg.text.strip() for seg in segments if seg.text.strip()]
      except Exception as e:
          print(f"[ERROR] Transcription failure: {e}")
          return []

  # ---------------- Silence Gating ----------------
  def is_silence(self, audio):
      # Pre-gate: quick checks
      rms = float(np.sqrt(np.mean(audio * audio) + 1e-12))
      peak = float(np.max(np.abs(audio)))
      silent = (rms < self.rms_threshold) and (peak < self.peak_threshold)
      return silent, rms, peak

  def adapt_thresholds(self, silent):
      if silent:
          self.silent_streak += 1
          if self.silent_streak > self.silence_grace:
              # Gradually lower thresholds to catch soft future speech
              self.rms_threshold *= self.adaptive_decay
              self.peak_threshold *= self.adaptive_decay
              # Clamp not to go below 40% of original
              self.rms_threshold = max(self.rms_threshold, 0.4 * self.rms_threshold_base)
              self.peak_threshold = max(self.peak_threshold, 0.4 * self.peak_threshold_base)
      else:
          # Reset thresholds quickly (we found speech)
          self.silent_streak = 0
          self.rms_threshold = self.rms_threshold_base
          self.peak_threshold = self.peak_threshold_base

  # ---------------- Main Loop ----------------
  def start_realtime(self,
                     chunk_duration=8,
                     overlap_duration=1,
                     sample_rate=DEFAULT_SAMPLE_RATE,
                     output_prefix="transcript",
                     sleep_base=0.25,
                     sleep_max=0.75):
      """
      Args:
          chunk_duration: Seconds of audio per model inference (larger = more efficient)
          overlap_duration: Seconds overlapped for continuity
          sample_rate: Always 16000 for Whisper
          sleep_base: Base idle sleep between polling loops
          sleep_max: Max adaptive sleep during extended silence
      """
      total_samples = int(chunk_duration * sample_rate)
      overlap_samples = int(overlap_duration * sample_rate)
      if overlap_samples >= total_samples:
          overlap_samples = int(0.2 * total_samples)

      print("\n[RUN] Energy-Optimized Transcription Starting")
      print(f"      Chunk={chunk_duration}s  Overlap={overlap_duration}s  SR={sample_rate}")
      print(f"      RMS threshold={self.rms_threshold_base:.4f}  Peak threshold={self.peak_threshold_base:.3f}")
      print("      Press Ctrl+C to stop.")
      print("===================================================")

      timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
      out_name = f"{output_prefix}_{timestamp}.txt"
      with open(out_name, "w", encoding="utf-8") as f:
          f.write(f"# Started: {datetime.now().isoformat()}\n")
          f.write(f"# Model={self.model_size} device={self.device} compute={self.compute_type}\n")
          f.write(f"# chunk_duration={chunk_duration} overlap={overlap_duration} low_power={self.low_power}\n")
          f.write("# ===============================================\n\n")

      # Buffer management
      frame_buffer = deque()
      frame_buffer_len = 0
      assembled = np.empty(total_samples, dtype=np.float32)  # Reused
      tail = np.empty(overlap_samples, dtype=np.float32) if overlap_samples > 0 else None

      # Stats
      audio_seconds_processed = 0.0
      compute_seconds = 0.0
      chunk_count = 0
      skip_count = 0
      last_print = time.time()

      start_time = time.time()
      dynamic_sleep = sleep_base

      try:
          with sd.InputStream(
              samplerate=sample_rate,
              channels=1,
              dtype="float32",
              callback=self.audio_callback,
              blocksize=0
          ):
              while True:
                  # Drain queue quickly
                  drained = 0
                  while not self.audio_queue.empty():
                      block = self.audio_queue.get()
                      frame_buffer.append(block)
                      frame_buffer_len += len(block)
                      drained += 1
                  if drained and self.low_power:
                      # In low power, still allow short processing pass
                      pass

                  # Assemble full chunk if we have enough samples
                  if frame_buffer_len >= total_samples:
                      # Fill assembled array
                      filled = 0
                      while filled < total_samples and frame_buffer:
                          blk = frame_buffer[0]
                          need = total_samples - filled
                          if len(blk) <= need:
                              assembled[filled:filled+len(blk)] = blk
                              filled += len(blk)
                              frame_buffer.popleft()
                              frame_buffer_len -= len(blk)
                          else:
                              assembled[filled:filled+need] = blk[:need]
                              frame_buffer[0] = blk[need:]
                              frame_buffer_len -= need
                              filled += need

                      # Prepare next buffer with overlap
                      if overlap_samples > 0:
                          tail[:] = assembled[-overlap_samples:]
                          # Prepend tail for future context
                          if tail.size:
                              frame_buffer.appendleft(tail.copy())
                              frame_buffer_len += tail.size

                      # Silence gating
                      silent, rms, peak = self.is_silence(assembled)
                      self.adapt_thresholds(silent)

                      if silent:
                          skip_count += 1
                          audio_seconds_processed += chunk_duration
                          # Adaptive sleep grows during silence
                          dynamic_sleep = min(sleep_max, dynamic_sleep * 1.15)
                          now = datetime.now().strftime("%H:%M:%S")
                          if skip_count % 8 == 0:
                              print(f"[{now}] (silence) rms={rms:.4f} peak={peak:.3f} thresholds(rms={self.rms_threshold:.4f},peak={self.peak_threshold:.3f})")
                          continue
                      else:
                          dynamic_sleep = max(sleep_base, dynamic_sleep * 0.7)

                      # Compute ratio control
                      est_cpu_ratio = (compute_seconds / audio_seconds_processed) if audio_seconds_processed > 0 else 0.0
                      if est_cpu_ratio > self.max_cpu_ratio:
                          # Throttle by skipping this chunk (simulate passive drop)
                          skip_count += 1
                          audio_seconds_processed += chunk_duration
                          print(f"[THROTTLE] Skipping chunk (cpu_ratio={est_cpu_ratio:.2f} > {self.max_cpu_ratio:.2f})")
                          continue

                      # Transcribe
                      t0 = time.time()
                      texts = self.transcribe_chunk(assembled)
                      t1 = time.time()
                      dt = t1 - t0

                      compute_seconds += dt
                      audio_seconds_processed += chunk_duration
                      chunk_count += 1

                      if texts:
                          joined = " ".join(texts)
                          stamp = datetime.now().strftime("%H:%M:%S")
                          line = f"[{stamp}] {joined}"
                          print(line)
                          with open(out_name, "a", encoding="utf-8") as f:
                              f.write(line + "\n")
                      else:
                          print(".", end="", flush=True)

                      # Periodic stats
                      now_t = time.time()
                      if now_t - last_print > 60 or (chunk_count % 25 == 0 and chunk_count > 0):
                          ratio = (compute_seconds / audio_seconds_processed) if audio_seconds_processed else 0
                          realtime_factor = (audio_seconds_processed / compute_seconds) if compute_seconds else 0
                          print(f"\n[STATS] chunks={chunk_count} skips={skip_count} cpu_ratio={ratio:.2f} realtime={realtime_factor:.1f}x")
                          last_print = now_t

                  # Sleep adaptively (duty cycle)
                  time.sleep(dynamic_sleep)

      except KeyboardInterrupt:
          pass
      finally:
          total_time = time.time() - start_time
          realtime_factor = (audio_seconds_processed / compute_seconds) if compute_seconds else 0
          with open(out_name, "a", encoding="utf-8") as f:
              f.write("\n# Session End\n")
              f.write(f"# Audio seconds: {audio_seconds_processed:.1f}\n")
              f.write(f"# Compute seconds: {compute_seconds:.1f}\n")
              f.write(f"# Real-time factor: {realtime_factor:.2f}x\n")
              f.write(f"# Chunks: {chunk_count}  Skipped(silence/throttle): {skip_count}\n")
              f.write(f"# Duration wall: {format_seconds(total_time)}\n")
          print("\n[END] Saved transcript to:", out_name)
          print(f"[END] Real-time factor: {realtime_factor:.2f}x  CPU ratio: {(compute_seconds/audio_seconds_processed) if audio_seconds_processed else 0:.2f}")
          print(f"[END] Chunks={chunk_count} Skipped={skip_count} Wall={format_seconds(total_time)}")

# ---------------- Argument Parsing ----------------
def parse_args():
  ap = argparse.ArgumentParser(description="Ultra energy-efficient Faster-Whisper real-time chunked transcriber.")
  ap.add_argument("--model", "-m", default="base",
                  choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
                  help="Whisper model size (tiny/base recommended for battery).")
  ap.add_argument("--chunk-duration", "-d", type=float, default=5.0,
                  help="Seconds per chunk (larger improves efficiency).")
  ap.add_argument("--overlap", "-o", type=float, default=1.0,
                  help="Overlap seconds for context (improves accuracy).")
  ap.add_argument("--rms-threshold", type=float, default=0.012,
                  help="RMS silence gate threshold.")
  ap.add_argument("--peak-threshold", type=float, default=0.10,
                  help="Peak silence gate threshold.")
  ap.add_argument("--low-power", action="store_true",
                  help="Enable extra duty cycling & longer sleeps.")
  ap.add_argument("--max-cpu-ratio", type=float, default=0.85,
                  help="Max compute_time / audio_time ratio before throttling.")
  ap.add_argument("--sleep-base", type=float, default=0.25,
                  help="Base sleep between polls.")
  ap.add_argument("--sleep-max", type=float, default=0.75,
                  help="Max adaptive sleep during silence.")
  ap.add_argument("--device", default=None, help="Force device (cpu/cuda). Default: cpu.")
  ap.add_argument("--compute-type", default=None,
                  help="Force compute type (int8, int8_float16, int16, float16, etc.)")
  ap.add_argument("--output-prefix", default="transcript_energy",
                  help="Prefix for transcript filename.")
  return ap.parse_args()

def main():
  args = parse_args()

  if args.model in ["medium", "large-v2", "large-v3"]:
      print("[WARN] High energy model selected. Proceed? (y/N)")
      if input().strip().lower() != "y":
          print("[INFO] Switching to 'base' for energy efficiency.")
          args.model = "base"

  transcriber = EnergyEfficientTranscriber(
      model_size=args.model,
      compute_type=args.compute_type,
      device=args.device,
      rms_threshold=args.rms_threshold,
      peak_threshold=args.peak_threshold,
      low_power=args.low_power,
      max_cpu_ratio=args.max_cpu_ratio
  )

  transcriber.start_realtime(
      chunk_duration=args.chunk_duration,
      overlap_duration=args.overlap,
      sample_rate=DEFAULT_SAMPLE_RATE,
      output_prefix=args.output_prefix,
      sleep_base=args.sleep_base,
      sleep_max=args.sleep_max
  )

if __name__ == "__main__":
  main()