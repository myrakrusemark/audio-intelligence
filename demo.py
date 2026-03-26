#!/usr/bin/env python3
"""
On-device STT + Environmental Sound Classification Demo
Everything runs locally — no cloud APIs, no data leaves your machine.
"""

import sys
import time
import queue
from difflib import SequenceMatcher
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from pathlib import Path

# YAMNet imports
from ai_edge_litert import interpreter as tflite_interp
import csv
import urllib.request

SAMPLE_RATE = 16000
CHUNK_SECONDS = 3
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_SECONDS
OVERLAP_SECONDS = 1        # feed Whisper 1s of prior audio for context
OVERLAP_SAMPLES = SAMPLE_RATE * OVERLAP_SECONDS

# ── Thresholds ──────────────────────────────────────────────────────────
ENERGY_FLOOR = 0.005       # RMS below this = silence, skip entirely
SOUND_CONFIDENCE = 0.15    # only show YAMNet labels above this
SPEECH_ENERGY = 0.01       # RMS needed before we bother running Whisper

# Speech-related YAMNet labels — used to gate Whisper and suppress from display
SPEECH_SOUNDS = {
    "Speech", "Narration, monologue", "Conversation", "Speech synthesizer",
    "Whispering", "Shout", "Yell", "Children shouting", "Screaming",
    "Singing", "Chant", "Male singing", "Female singing", "Child singing",
    "Choir", "Male speech, man speaking", "Female speech, woman speaking",
}

# Boring YAMNet classes to suppress (redundant with STT or just noise)
BORING_SOUNDS = SPEECH_SOUNDS | {
    "Silence", "White noise", "Noise", "Static", "Hum", "Buzz",
}

# ── Colors ──────────────────────────────────────────────────────────────
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[92m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
RED = "\033[91m"
RESET = "\033[0m"


def rms(audio):
    """Root mean square energy of an audio chunk."""
    return float(np.sqrt(np.mean(audio ** 2)))


def stitch(prev_text, new_text, min_overlap_words=2):
    """Fuzzy-merge new_text onto prev_text by finding the overlapping tail/head."""
    if not prev_text:
        return new_text
    if not new_text:
        return prev_text

    prev_words = prev_text.split()
    new_words = new_text.split()

    # Try progressively shorter suffixes of prev against prefixes of new
    best_match_len = 0
    best_ratio = 0.0

    max_check = min(len(prev_words), len(new_words), 15)
    for overlap_len in range(min_overlap_words, max_check + 1):
        tail = " ".join(prev_words[-overlap_len:]).lower()
        head = " ".join(new_words[:overlap_len]).lower()
        ratio = SequenceMatcher(None, tail, head).ratio()
        # Accept fuzzy match (handles minor Whisper variations)
        if ratio > 0.6 and ratio >= best_ratio:
            best_ratio = ratio
            best_match_len = overlap_len

    if best_match_len > 0:
        # Merge: keep prev + everything after the overlapping prefix in new
        return prev_text + " " + " ".join(new_words[best_match_len:])
    else:
        # No overlap found — just append with a separator
        return prev_text + " // " + new_text


def download_yamnet():
    """Download YAMNet TFLite model + class map if not cached."""
    cache = Path(__file__).parent / ".cache"
    cache.mkdir(exist_ok=True)

    model_path = cache / "yamnet.tflite"
    classes_path = cache / "yamnet_classes.csv"

    model_url = "https://github.com/DENGRENHAO/Yamnet_tflite/raw/main/yamnet.tflite"
    classes_url = "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"

    if not model_path.exists():
        print(f"{DIM}Downloading YAMNet model...{RESET}", flush=True)
        urllib.request.urlretrieve(model_url, model_path)

    if not classes_path.exists():
        print(f"{DIM}Downloading YAMNet class map...{RESET}", flush=True)
        urllib.request.urlretrieve(classes_url, classes_path)

    # Parse class names
    class_names = []
    with open(classes_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            class_names.append(row["display_name"])

    return str(model_path), class_names


def classify_sounds(interpreter, audio_chunk, class_names, top_k=5):
    """Run YAMNet on an audio chunk, return top-k (label, score) pairs."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # YAMNet expects float32 waveform in [-1, 1]
    waveform = audio_chunk.astype(np.float32)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)

    # Ensure correct length — YAMNet TFLite expects 15600 samples (0.975s)
    expected_len = input_details[0]["shape"][-1]
    if len(waveform) > expected_len:
        # Classify multiple windows and average
        n_windows = len(waveform) // expected_len
        all_scores = []
        for i in range(n_windows):
            segment = waveform[i * expected_len : (i + 1) * expected_len]
            segment = segment.reshape(1, -1).astype(np.float32)
            interpreter.resize_tensor_input(input_details[0]["index"], segment.shape)
            interpreter.allocate_tensors()
            interpreter.set_tensor(input_details[0]["index"], segment)
            interpreter.invoke()
            scores = interpreter.get_tensor(output_details[0]["index"])
            all_scores.append(scores)
        scores = np.mean(all_scores, axis=0)
    else:
        waveform = np.pad(waveform, (0, max(0, expected_len - len(waveform))))
        waveform = waveform[:expected_len].reshape(1, -1).astype(np.float32)
        interpreter.resize_tensor_input(input_details[0]["index"], waveform.shape)
        interpreter.allocate_tensors()
        interpreter.set_tensor(input_details[0]["index"], waveform)
        interpreter.invoke()
        scores = interpreter.get_tensor(output_details[0]["index"])

    scores = scores.flatten()
    top_indices = np.argsort(scores)[-top_k:][::-1]
    return [(class_names[i], float(scores[i])) for i in top_indices]


def energy_bar(level, width=20):
    """Render a small energy meter."""
    filled = int(min(level / 0.05, 1.0) * width)
    bar = "█" * filled + "░" * (width - filled)
    if level < ENERGY_FLOOR:
        return f"{DIM}{bar}{RESET}"
    elif level < SPEECH_ENERGY:
        return f"{YELLOW}{bar}{RESET}"
    else:
        return f"{GREEN}{bar}{RESET}"


def wrap_text(text, width):
    """Word-wrap text into a list of lines."""
    words = text.split()
    lines = []
    line = ""
    for w in words:
        if len(line) + len(w) + 1 > width:
            lines.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    if line:
        lines.append(line)
    return lines


def print_wrapped(prefix, text, suffix, width):
    """Print word-wrapped text with prefix/suffix on each line."""
    for line in wrap_text(text, width):
        print(f"{prefix}{line}{suffix}")


def main():
    print(f"\n{BOLD}{'═' * 60}{RESET}")
    print(f"{BOLD}  On-Device Audio Intelligence Demo{RESET}")
    print(f"{DIM}  STT (Whisper) + Environmental Sounds (YAMNet){RESET}")
    print(f"{DIM}  Everything runs locally — nothing leaves your machine.{RESET}")
    print(f"{BOLD}{'═' * 60}{RESET}\n")

    # ── Load models ─────────────────────────────────────────────────
    print(f"{YELLOW}Loading models (first run downloads ~75MB)...{RESET}")

    print(f"  {DIM}Loading Whisper (tiny)...{RESET}", end=" ", flush=True)
    whisper = WhisperModel("tiny", device="cpu", compute_type="int8")
    print(f"{GREEN}ok{RESET}")

    print(f"  {DIM}Loading YAMNet...{RESET}", end=" ", flush=True)
    yamnet_path, class_names = download_yamnet()
    interpreter = tflite_interp.Interpreter(model_path=yamnet_path)
    interpreter.allocate_tensors()
    print(f"{GREEN}ok{RESET}")

    print(f"\n{GREEN}Ready! Listening on your mic.{RESET}")
    print(f"{DIM}Chunks: {CHUNK_SECONDS}s | silence gate: RMS < {ENERGY_FLOOR} | sound threshold: {SOUND_CONFIDENCE:.0%}{RESET}")
    print(f"{DIM}Press Ctrl+C to stop.{RESET}\n")
    print(f"{'─' * 60}")

    # ── Audio capture loop ──────────────────────────────────────────
    audio_q = queue.Queue()
    silent_chunks = 0
    prev_chunk = np.zeros(OVERLAP_SAMPLES, dtype=np.float32)  # overlap for Whisper
    passage = ""               # current in-progress passage
    passages = []              # completed passages
    lines_to_clear = 0        # lines to wipe for live passage redraw

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(f"{DIM}(audio: {status}){RESET}", file=sys.stderr)
        audio_q.put(indata.copy())

    buffer = np.empty((0,), dtype=np.float32)

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=SAMPLE_RATE,  # 1s blocks
            callback=audio_callback,
        ):
            while True:
                # Accumulate audio until we have a full chunk
                data = audio_q.get()
                data = data.flatten()
                buffer = np.concatenate([buffer, data])

                if len(buffer) < CHUNK_SAMPLES:
                    continue

                chunk = buffer[:CHUNK_SAMPLES]
                buffer = buffer[CHUNK_SAMPLES:]

                # ── Energy gate ─────────────────────────────────────
                level = rms(chunk)
                ts = time.strftime("%H:%M:%S")

                if level < ENERGY_FLOOR:
                    silent_chunks += 1
                    prev_chunk = np.zeros(OVERLAP_SAMPLES, dtype=np.float32)
                    # First silence after speech = flush the passage
                    if silent_chunks == 1 and passage:
                        # Overwrite the live draft with the final version
                        if lines_to_clear > 0:
                            sys.stdout.write(f"\033[{lines_to_clear}A\033[J")
                        print(f"  {BOLD}{'·' * 50}{RESET}")
                        print_wrapped(f"  {GREEN}", passage, f"{RESET}", 70)
                        print(f"  {BOLD}{'·' * 50}{RESET}\n")
                        passages.append(passage)
                        passage = ""
                        lines_to_clear = 0
                    # Show a heartbeat every 5 silent chunks so you know it's alive
                    if silent_chunks % 5 == 1:
                        print(f"  {DIM}[{ts}] {energy_bar(level)} silent — listening...{RESET}", flush=True)
                    continue

                silent_chunks = 0

                # ── Sound classification (always run if above energy floor) ──
                sounds = classify_sounds(interpreter, chunk, class_names)
                notable_sounds = [
                    (label, score)
                    for label, score in sounds
                    if score > SOUND_CONFIDENCE and label not in BORING_SOUNDS
                ]

                # ── Speech-to-text (only if YAMNet thinks there's a voice) ──
                # Check all YAMNet results (not just notable) for speech-like labels
                yamnet_sees_speech = any(
                    label in SPEECH_SOUNDS and score > 0.08
                    for label, score in sounds
                )
                text = ""
                if level >= SPEECH_ENERGY and yamnet_sees_speech:
                    whisper_input = np.concatenate([prev_chunk, chunk])
                    segments, info = whisper.transcribe(
                        whisper_input, beam_size=1, language="en", vad_filter=True
                    )
                    text = " ".join(s.text.strip() for s in segments).strip()

                # Save tail of this chunk as overlap for next round
                prev_chunk = chunk[-OVERLAP_SAMPLES:]

                # ── Inject sounds into passage as stage directions ────
                if notable_sounds:
                    stage_dir = "[" + ", ".join(
                        f"{label.lower()}" for label, score in notable_sounds
                    ) + "]"
                    if passage:
                        passage += f" {stage_dir}"
                    else:
                        passage = stage_dir

                # ── Build passage live ────────────────────────────────
                if text:
                    passage = stitch(passage, text)
                    display = wrap_text(passage, 70)
                    # Wipe previous draft and reprint
                    if lines_to_clear > 0:
                        sys.stdout.write(f"\033[{lines_to_clear}A\033[J")
                    for line in display:
                        print(f"  {DIM}▌{RESET} {GREEN}{line}{RESET}")
                    lines_to_clear = len(display)
                    sys.stdout.flush()
                elif not notable_sounds:
                    if lines_to_clear > 0:
                        sys.stdout.write(f"\033[{lines_to_clear}A\033[J")
                        lines_to_clear = 0
                    print(f"  {DIM}[{ts}] {energy_bar(level)} activity, nothing recognized{RESET}", flush=True)

    except KeyboardInterrupt:
        # Flush any in-progress passage
        if passage:
            passages.append(passage)
        print(f"\n\n{YELLOW}Stopped.{RESET}")
        if passages:
            print(f"\n{BOLD}{'═' * 60}{RESET}")
            print(f"{BOLD}  All passages:{RESET}")
            for i, p in enumerate(passages, 1):
                print(f"\n  {DIM}[{i}]{RESET}")
                print_wrapped(f"  {GREEN}", p, f"{RESET}", 70)
            print(f"\n{BOLD}{'═' * 60}{RESET}")


if __name__ == "__main__":
    main()
