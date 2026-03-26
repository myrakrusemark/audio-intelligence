#!/usr/bin/env python3
"""
On-device STT + Environmental Sound Classification + Speaker Diarization Demo
Everything runs locally — no cloud APIs, no data leaves your machine.
"""

import sys
import time
import json
import queue
import argparse
from dataclasses import dataclass, field
from difflib import SequenceMatcher
import numpy as np
from scipy.signal import butter, sosfilt
import sounddevice as sd
from faster_whisper import WhisperModel
from pathlib import Path

# YAMNet imports
from ai_edge_litert import interpreter as tflite_interp
import csv
import urllib.request

# Speaker embedding imports
import onnxruntime as ort
from huggingface_hub import hf_hub_download

SAMPLE_RATE = 16000
CHUNK_SECONDS = 2
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_SECONDS
OVERLAP_SECONDS = 1.5      # feed Whisper 1.5s of prior audio for context
OVERLAP_SAMPLES = int(SAMPLE_RATE * OVERLAP_SECONDS)

# ── Whisper prompt — biases decoder toward these words/names ────────────
WHISPER_PROMPT = "Fathom, the AI agent called Fathom"

# ── Post-processing fixes for known Whisper mishearings ────────────────
WORD_FIXES = {
    "Adam": "Fathom",
    "adam": "Fathom",
    "Phantom": "Fathom",
    "phantom": "Fathom",
}

# ── Thresholds ──────────────────────────────────────────────────────────
ENERGY_FLOOR_MIN = 0.005   # absolute minimum noise floor
ENERGY_FLOOR_MARGIN = 1.5  # noise floor = ambient RMS * this multiplier
NOISE_FLOOR_WINDOW = 10    # seconds of history for rolling noise floor
SOUND_CONFIDENCE = 0.15    # default threshold for most sounds

# ── Voice bandpass filter (cuts AC hum, fan noise, high-freq artifacts) ──
VOICE_BANDPASS = butter(4, [300, 3400], btype='band', fs=SAMPLE_RATE, output='sos')

# Sounds that are reliably accurate even at low confidence — give them a lower bar
SENSITIVE_SOUNDS = {
    "Cat": 0.06, "Meow": 0.06, "Purr": 0.06, "Hiss": 0.08,
    "Dog": 0.06, "Bark": 0.06, "Growling": 0.08,
    "Doorbell": 0.06, "Door": 0.08, "Knock": 0.08,
    "Alarm": 0.06, "Smoke detector, smoke alarm": 0.05,
    "Fire alarm": 0.05, "Siren": 0.06,
    "Glass": 0.08, "Shatter": 0.06,
    "Gunshot, gunfire": 0.06, "Explosion": 0.06,
    "Baby cry, infant cry": 0.06,
    "Telephone": 0.08, "Ringtone": 0.08,
}
SPEECH_ENERGY = 0.01       # RMS needed before we bother running Whisper

# ── Speaker diarization ──────────────────────────────────────────────
SPEAKER_SIM_THRESHOLD = 0.35  # cosine similarity above this = same speaker
SPEAKER_PRESENCE_THRESHOLD = 0.25  # lower bar for detecting voice in mixed audio
SPEAKER_MIN_AUDIO = int(SAMPLE_RATE * 1.5)  # need 1.5s of speech to diarize
SPEAKER_ID_WINDOW = 2.0      # seconds of audio in sliding window for real-time ID
SPEAKER_ID_STRIDE = 1.0      # run identification every N seconds
AI_VOICE_NAMES = {"Fathom"}  # enrolled voices that are AI/TTS (for barge-in detection)

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


class NoiseFloor:
    """Rolling adaptive noise floor from the quietest blocks."""

    def __init__(self, window=NOISE_FLOOR_WINDOW):
        self._levels = []
        self._window = window
        self.level = ENERGY_FLOOR_MIN
        self._last_printed = None

    def update(self, block_rms):
        """Feed a 1s block RMS. Updates the noise floor. Prints on change."""
        self._levels.append(block_rms)
        if len(self._levels) > self._window:
            self._levels.pop(0)
        # Noise floor = 10th percentile of recent blocks (ignores speech spikes)
        p10 = float(np.percentile(self._levels, 10))
        old = self.level
        self.level = max(p10 * ENERGY_FLOOR_MARGIN, ENERGY_FLOOR_MIN)
        # Print when floor changes meaningfully (>20% shift)
        rounded = round(self.level, 4)
        if self._last_printed is None or abs(self.level - old) / max(old, 1e-6) > 0.2:
            print(f"  {DIM}noise floor: {self.level:.4f}{RESET}")
            self._last_printed = rounded


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
        return prev_text + " ... " + new_text


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


# ── Speaker embedding (WeSpeaker ECAPA-TDNN via ONNX) ────────────────

def download_speaker_model():
    """Download WeSpeaker ECAPA-TDNN ONNX model via huggingface_hub."""
    path = hf_hub_download(
        repo_id="Wespeaker/wespeaker-ecapa-tdnn512-LM",
        filename="voxceleb_ECAPA512_LM.onnx",
    )
    return path


def compute_fbank(audio, sample_rate=16000, n_mels=80,
                  frame_length_ms=25, frame_shift_ms=10):
    """Compute log mel-filterbank features from raw audio (pure numpy)."""
    frame_length = int(sample_rate * frame_length_ms / 1000)
    frame_shift = int(sample_rate * frame_shift_ms / 1000)
    n_fft = 1
    while n_fft < frame_length:
        n_fft *= 2

    # Hamming window
    window = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(frame_length) / (frame_length - 1))

    # Pad audio so we get at least one frame
    if len(audio) < frame_length:
        audio = np.pad(audio, (0, frame_length - len(audio)))

    # Frame the signal
    n_frames = 1 + (len(audio) - frame_length) // frame_shift
    frames = np.stack([
        audio[i * frame_shift : i * frame_shift + frame_length]
        for i in range(n_frames)
    ])

    # Windowed FFT
    windowed = frames * window
    spectrum = np.fft.rfft(windowed, n=n_fft)
    power = np.abs(spectrum) ** 2 / n_fft

    # Mel filterbank
    fmin, fmax = 20.0, sample_rate / 2.0
    mel_min = 2595.0 * np.log10(1.0 + fmin / 700.0)
    mel_max = 2595.0 * np.log10(1.0 + fmax / 700.0)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    fbank_matrix = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(n_mels):
        left, center, right = bin_points[m], bin_points[m + 1], bin_points[m + 2]
        for k in range(left, center):
            if center > left:
                fbank_matrix[m, k] = (k - left) / (center - left)
        for k in range(center, right):
            if right > center:
                fbank_matrix[m, k] = (right - k) / (right - center)

    mel_spec = power @ fbank_matrix.T
    mel_spec = np.where(mel_spec < 1e-10, 1e-10, mel_spec)
    log_mel = np.log(mel_spec)

    # Per-utterance mean normalization (CMVN)
    log_mel = log_mel - np.mean(log_mel, axis=0, keepdims=True)

    return log_mel.astype(np.float32)


def extract_speaker_embedding(session, audio):
    """Extract 192-dim speaker embedding from audio using WeSpeaker ONNX model."""
    feats = compute_fbank(audio)
    feats = feats[np.newaxis, :, :]  # (1, T, 80)
    outputs = session.run(None, {"feats": feats})
    embedding = outputs[0].flatten()
    # L2 normalize
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    return embedding


def cosine_similarity(a, b):
    """Cosine similarity between two L2-normalized vectors."""
    return float(np.dot(a, b))


def assign_speakers(embeddings, threshold=SPEAKER_SIM_THRESHOLD):
    """Cluster embeddings into speakers. Returns (labels, profiles, confidences).

    Re-runs assignment from scratch each time so early passages benefit
    from later speaker profile refinement.
    """
    if not embeddings:
        return [], {}, {}

    profiles = {}   # speaker_id -> running average embedding
    counts = {}     # speaker_id -> number of passages assigned
    labels = []
    next_id = 0

    # Multiple passes: assign, rebuild profiles, re-assign
    # Start with single-pass, then refine
    prev_labels = None
    for _ in range(3):
        profiles.clear()
        counts.clear()
        labels.clear()
        next_id = 0

        for emb in embeddings:
            best_id = -1
            best_sim = -1.0
            for sid, profile in profiles.items():
                sim = cosine_similarity(emb, profile)
                if sim > best_sim:
                    best_sim = sim
                    best_id = sid

            if best_id >= 0 and best_sim >= threshold:
                labels.append(best_id)
                # Update running average
                n = counts[best_id]
                profiles[best_id] = (profiles[best_id] * n + emb) / (n + 1)
                # Re-normalize
                norm = np.linalg.norm(profiles[best_id])
                if norm > 0:
                    profiles[best_id] = profiles[best_id] / norm
                counts[best_id] = n + 1
            else:
                labels.append(next_id)
                profiles[next_id] = emb.copy()
                counts[next_id] = 1
                next_id += 1

        if labels == prev_labels:
            break
        prev_labels = labels.copy()

    # Compute per-speaker confidence (average similarity to profile)
    confidences = {}
    for sid in profiles:
        sims = [
            cosine_similarity(embeddings[i], profiles[sid])
            for i, lbl in enumerate(labels) if lbl == sid
        ]
        confidences[sid] = np.mean(sims) if sims else 0.0

    return labels, profiles, confidences


SPEAKER_NAMES = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
VOICES_DIR = Path(__file__).parent / ".voices"


def load_voice_library():
    """Load saved voice profiles from .voices/ directory."""
    voices = {}  # name -> {"embedding": np.array, "samples": int}
    if not VOICES_DIR.exists():
        return voices
    for f in VOICES_DIR.glob("*.json"):
        data = json.loads(f.read_text())
        voices[data["name"]] = {
            "embedding": np.array(data["embedding"], dtype=np.float32),
            "samples": data.get("samples", 1),
        }
    return voices


def save_voice(name, embedding, samples=1):
    """Save a voice profile to .voices/ directory."""
    VOICES_DIR.mkdir(exist_ok=True)
    path = VOICES_DIR / f"{name.lower().replace(' ', '_')}.json"
    path.write_text(json.dumps({
        "name": name,
        "embedding": embedding.tolist(),
        "samples": samples,
    }, indent=2))
    return path


def remove_voice(name):
    """Remove a voice profile. Returns True if found."""
    if not VOICES_DIR.exists():
        return False
    path = VOICES_DIR / f"{name.lower().replace(' ', '_')}.json"
    if path.exists():
        path.unlink()
        return True
    return False


def list_voices():
    """List all enrolled voice names."""
    if not VOICES_DIR.exists():
        return []
    names = []
    for f in VOICES_DIR.glob("*.json"):
        data = json.loads(f.read_text())
        names.append((data["name"], data.get("samples", 1)))
    return sorted(names)


def assign_speakers(embeddings, known_voices=None, threshold=SPEAKER_SIM_THRESHOLD):
    """Cluster embeddings into speakers. Returns (labels, profiles, confidences).

    known_voices: dict of name -> {"embedding": np.array, "samples": int}
    Known voices are seeded as initial profiles and matched first.
    Labels for known voices use the person's name (str), unknown use int IDs.
    """
    if not embeddings:
        return [], {}, {}

    # Seed profiles from known voices
    profiles = {}   # speaker_id (str name or int) -> embedding
    counts = {}
    next_anon_id = 0

    if known_voices:
        for name, info in known_voices.items():
            profiles[name] = info["embedding"].copy()
            counts[name] = info["samples"]

    labels = []

    prev_labels = None
    for _ in range(3):
        # Reset to known voices only (don't lose them between passes)
        profiles_reset = {}
        counts_reset = {}
        if known_voices:
            for name, info in known_voices.items():
                profiles_reset[name] = info["embedding"].copy()
                counts_reset[name] = info["samples"]
        profiles = profiles_reset
        counts = counts_reset
        labels.clear()
        next_anon_id = 0

        for emb in embeddings:
            best_id = None
            best_sim = -1.0
            for sid, profile in profiles.items():
                sim = cosine_similarity(emb, profile)
                if sim > best_sim:
                    best_sim = sim
                    best_id = sid

            if best_id is not None and best_sim >= threshold:
                labels.append(best_id)
                n = counts[best_id]
                profiles[best_id] = (profiles[best_id] * n + emb) / (n + 1)
                norm = np.linalg.norm(profiles[best_id])
                if norm > 0:
                    profiles[best_id] = profiles[best_id] / norm
                counts[best_id] = n + 1
            else:
                anon_id = next_anon_id
                next_anon_id += 1
                labels.append(anon_id)
                profiles[anon_id] = emb.copy()
                counts[anon_id] = 1

        if labels == prev_labels:
            break
        prev_labels = labels.copy()

    confidences = {}
    for sid in profiles:
        sims = [
            cosine_similarity(embeddings[i], profiles[sid])
            for i, lbl in enumerate(labels) if lbl == sid
        ]
        confidences[sid] = np.mean(sims) if sims else 0.0

    return labels, profiles, confidences


def speaker_display_name(label):
    """Convert a speaker label (str name or int ID) to display string."""
    if isinstance(label, str):
        return label
    return f"Person {SPEAKER_NAMES[label]}" if label < len(SPEAKER_NAMES) else f"Person {label}"


def speaker_status_line(profiles, confidences):
    """Render the speaker status bar."""
    if not profiles:
        return f"  {DIM}Speakers: (none detected){RESET}"
    parts = []
    for sid in sorted(profiles.keys(), key=str):
        name = speaker_display_name(sid)
        conf = confidences.get(sid, 0.0)
        color = GREEN if conf > 0.6 else YELLOW if conf > 0.4 else RED
        parts.append(f"{name} {color}{conf:.0%}{RESET}")
    return f"  {BOLD}Speakers:{RESET} " + f" {DIM}|{RESET} ".join(parts)


# ── Real-time speaker identification ───────────────────────────────

@dataclass
class SpeakerEvent:
    """Emitted by SpeakerIdentifier when speaker state changes."""
    timestamp: float
    event_type: str               # "silence", "unknown", "single", "multiple"
    speakers: list = field(default_factory=list)          # matched voice names
    similarities: dict = field(default_factory=dict)      # {name: cosine_sim} for all enrolled
    is_barge_in: bool = False     # AI voice + human voice both detected


class SpeakerIdentifier:
    """Sliding-window real-time speaker identification against enrolled voices.

    Call feed() with each 1-second audio block. Returns a SpeakerEvent when
    speaker state changes, None otherwise.
    """

    def __init__(self, speaker_session, known_voices, noise_floor=None, on_event=None):
        self._session = speaker_session
        self._known_voices = known_voices or {}
        self._noise_floor = noise_floor
        self._on_event = on_event
        self._window_samples = int(SAMPLE_RATE * SPEAKER_ID_WINDOW)
        self._stride_samples = int(SAMPLE_RATE * SPEAKER_ID_STRIDE)
        self._ring = np.empty((0,), dtype=np.float32)
        self._samples_since_id = 0
        self._last_event_type = None
        self._last_speakers = frozenset()

    def update_voices(self, known_voices):
        """Hot-update the voice library (e.g. after enrolling a new voice)."""
        self._known_voices = known_voices or {}

    def feed(self, audio_block):
        """Feed a 1-second audio block. Returns SpeakerEvent on state change."""
        self._ring = np.concatenate([self._ring, audio_block])
        if len(self._ring) > self._window_samples:
            self._ring = self._ring[-self._window_samples:]

        self._samples_since_id += len(audio_block)
        if self._samples_since_id < self._stride_samples:
            return None
        self._samples_since_id = 0

        event = self._identify()

        if self._on_event:
            self._on_event(event)
        return event

    def _identify(self):
        """Run speaker identification on the current window."""
        level = rms(self._ring)
        now = time.time()

        floor = self._noise_floor.level if self._noise_floor else ENERGY_FLOOR_MIN
        if level < floor:
            return SpeakerEvent(timestamp=now, event_type="silence")

        if len(self._ring) < SPEAKER_MIN_AUDIO:
            return SpeakerEvent(timestamp=now, event_type="silence")

        if not self._known_voices:
            return SpeakerEvent(timestamp=now, event_type="unknown")

        emb = extract_speaker_embedding(self._session, self._ring)

        # Compare against all enrolled voices
        sims = {}
        for name, info in self._known_voices.items():
            sims[name] = cosine_similarity(emb, info["embedding"])

        # Classify matches
        confident = [n for n, s in sims.items() if s >= SPEAKER_SIM_THRESHOLD]
        present = [n for n, s in sims.items() if s >= SPEAKER_PRESENCE_THRESHOLD]

        if len(confident) >= 2 or (len(confident) >= 1 and len(present) >= 2):
            speakers = sorted(present, key=lambda n: sims[n], reverse=True)
            has_ai = any(n in AI_VOICE_NAMES for n in speakers)
            has_human = any(n not in AI_VOICE_NAMES for n in speakers)
            return SpeakerEvent(
                timestamp=now, event_type="multiple", speakers=speakers,
                similarities=sims, is_barge_in=has_ai and has_human,
            )
        elif len(confident) == 1:
            return SpeakerEvent(
                timestamp=now, event_type="single", speakers=confident,
                similarities=sims,
            )
        elif len(present) >= 2:
            speakers = sorted(present, key=lambda n: sims[n], reverse=True)
            has_ai = any(n in AI_VOICE_NAMES for n in speakers)
            has_human = any(n not in AI_VOICE_NAMES for n in speakers)
            return SpeakerEvent(
                timestamp=now, event_type="multiple", speakers=speakers,
                similarities=sims, is_barge_in=has_ai and has_human,
            )
        elif len(present) == 1:
            return SpeakerEvent(
                timestamp=now, event_type="single", speakers=present,
                similarities=sims,
            )
        else:
            return SpeakerEvent(
                timestamp=now, event_type="unknown", similarities=sims,
            )


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


ENROLL_PASSAGE = """The rainbow is a division of white light into many beautiful
colors. These take the shape of a long round arch, with its path
high above, and its two ends apparently beyond the horizon. There
is, according to legend, a pot of gold at one end. People look,
but no one ever finds it. When a man looks for something beyond
his reach, his friends say he is looking for the pot of gold at
the end of the rainbow."""

ENROLL_ROUNDS = [
    ("Normal voice", "Read this at your normal speaking pace and tone:"),
    ("Energetic", "Now read it again — upbeat, like you're excited about it:"),
    ("Calm and slow", "One more time — slow, calm, like you're winding down:"),
]

ENROLL_DURATION = 15  # seconds per round


def enroll_voice(name, fresh=False):
    """Record audio across multiple intonations and save a voice profile."""
    print(f"\n{BOLD}{'═' * 60}{RESET}")
    print(f"{BOLD}  Voice Enrollment: {name}{RESET}")
    print(f"{BOLD}{'═' * 60}{RESET}\n")

    existing = load_voice_library()
    if name in existing and not fresh:
        n = existing[name]["samples"]
        print(f"{YELLOW}A voice profile for \"{name}\" already exists ({n} sample{'s' if n != 1 else ''}).{RESET}")
        print(f"{YELLOW}This will blend new audio into the existing profile.{RESET}")
        print(f"{RED}Only continue if this is the SAME person's voice!{RESET}\n")
        resp = input(f"{DIM}Continue? [y/N] {RESET}").strip().lower()
        if resp != "y":
            print(f"{DIM}Cancelled.{RESET}")
            return
    elif name in existing and fresh:
        print(f"{YELLOW}Starting fresh profile for \"{name}\" (replacing existing).{RESET}\n")
        existing.pop(name)

    print(f"{DIM}Loading speaker model...{RESET}", end=" ", flush=True)
    speaker_model_path = download_speaker_model()
    speaker_session = ort.InferenceSession(speaker_model_path)
    print(f"{GREEN}ok{RESET}\n")

    print(f"{DIM}You'll read a short passage 3 times with different energy.{RESET}")
    print(f"{DIM}This captures the natural range of your voice.{RESET}\n")

    embeddings = []

    for i, (style, instruction) in enumerate(ENROLL_ROUNDS):
        print(f"{BOLD}Round {i+1}/{len(ENROLL_ROUNDS)}: {style}{RESET}")
        print(f"{YELLOW}{instruction}{RESET}\n")
        for line in ENROLL_PASSAGE.strip().splitlines():
            print(f"  {GREEN}{line.strip()}{RESET}")

        print(f"\n{DIM}Press Enter when ready...{RESET}", end="")
        input()
        print(f"{YELLOW}Recording ({ENROLL_DURATION}s)...{RESET}")

        audio = sd.rec(int(SAMPLE_RATE * ENROLL_DURATION), samplerate=SAMPLE_RATE,
                       channels=1, dtype="float32")
        for sec in range(ENROLL_DURATION, 0, -1):
            print(f"\r  {BOLD}{sec}s remaining...{RESET}  ", end="", flush=True)
            time.sleep(1)
        sd.wait()
        print(f"\r  {GREEN}Done.{RESET}                    ")

        audio = audio.flatten()
        level = rms(audio)
        if level < ENERGY_FLOOR:
            print(f"{RED}Too quiet — skipping this round.{RESET}\n")
            continue

        # Extract multiple embeddings from 5s segments for robustness
        seg_len = SAMPLE_RATE * 5
        for start in range(0, len(audio) - seg_len + 1, seg_len):
            segment = audio[start : start + seg_len]
            if rms(segment) >= ENERGY_FLOOR:
                embeddings.append(extract_speaker_embedding(speaker_session, segment))

        print(f"  {DIM}Extracted {len(embeddings)} segments so far{RESET}\n")

    if not embeddings:
        print(f"{RED}No usable audio captured. Try again closer to the mic.{RESET}")
        return

    # Average all segment embeddings
    avg_embedding = np.mean(embeddings, axis=0)
    norm = np.linalg.norm(avg_embedding)
    if norm > 0:
        avg_embedding = avg_embedding / norm

    # Blend with existing profile if present
    if name in existing:
        old = existing[name]
        n = old["samples"]
        blended = (old["embedding"] * n + avg_embedding) / (n + 1)
        norm = np.linalg.norm(blended)
        if norm > 0:
            blended = blended / norm
        path = save_voice(name, blended, n + 1)
        print(f"{GREEN}Updated voice profile for {BOLD}{name}{RESET}{GREEN} ({n+1} samples){RESET}")
    else:
        path = save_voice(name, avg_embedding, 1)
        print(f"{GREEN}Saved voice profile for {BOLD}{name}{RESET}")

    print(f"{DIM}{path}{RESET}")


def main():
    parser = argparse.ArgumentParser(description="On-device audio intelligence")
    parser.add_argument("--enroll", metavar="NAME", help="Enroll a voice (blends with existing)")
    parser.add_argument("-n", action="store_true", help="With --enroll: start fresh instead of blending")
    parser.add_argument("--remove", metavar="NAME", nargs="?", const="__list__",
                        help="Remove a voice profile (omit name to list)")
    args = parser.parse_args()

    if args.remove is not None:
        if args.remove == "__list__":
            voices = list_voices()
            if not voices:
                print(f"{DIM}No enrolled voices.{RESET}")
            else:
                print(f"\n{BOLD}Enrolled voices:{RESET}")
                for name, samples in voices:
                    print(f"  {CYAN}{name}{RESET} {DIM}({samples} sample{'s' if samples != 1 else ''}){RESET}")
                print(f"\n{DIM}Usage: --remove \"Name\"{RESET}")
            return

        if remove_voice(args.remove):
            print(f"{GREEN}Removed voice profile: {BOLD}{args.remove}{RESET}")
        else:
            print(f"{RED}No voice profile found for: {args.remove}{RESET}")
            voices = list_voices()
            if voices:
                print(f"{DIM}Known voices: {', '.join(n for n, _ in voices)}{RESET}")
        return

    if args.enroll:
        enroll_voice(args.enroll, fresh=args.n)
        return

    print(f"\n{BOLD}{'═' * 60}{RESET}")
    print(f"{BOLD}  On-Device Audio Intelligence Demo{RESET}")
    print(f"{DIM}  STT (Whisper) + Sounds (YAMNet) + Speakers (WeSpeaker){RESET}")
    print(f"{DIM}  Everything runs locally — nothing leaves your machine.{RESET}")
    print(f"{BOLD}{'═' * 60}{RESET}\n")

    # ── Load models ─────────────────────────────────────────────────
    print(f"{YELLOW}Loading models (first run downloads ~100MB)...{RESET}")

    print(f"  {DIM}Loading Whisper (tiny)...{RESET}", end=" ", flush=True)
    whisper = WhisperModel("tiny", device="cpu", compute_type="int8")
    print(f"{GREEN}ok{RESET}")

    print(f"  {DIM}Loading YAMNet...{RESET}", end=" ", flush=True)
    yamnet_path, class_names = download_yamnet()
    interpreter = tflite_interp.Interpreter(model_path=yamnet_path)
    interpreter.allocate_tensors()
    print(f"{GREEN}ok{RESET}")

    print(f"  {DIM}Loading WeSpeaker (ECAPA-TDNN)...{RESET}", end=" ", flush=True)
    speaker_model_path = download_speaker_model()
    speaker_session = ort.InferenceSession(speaker_model_path)
    print(f"{GREEN}ok{RESET}")

    # ── Load voice library ──────────────────────────────────────────
    known_voices = load_voice_library()
    if known_voices:
        print(f"\n  {CYAN}Known voices:{RESET} {', '.join(known_voices.keys())}")
    else:
        print(f"\n  {DIM}No enrolled voices (use --enroll \"Name\" to add){RESET}")

    noise_floor = NoiseFloor()

    # ── Real-time speaker identifier ─────────────────────────────
    speaker_id = SpeakerIdentifier(speaker_session, known_voices, noise_floor=noise_floor)

    print(f"\n{GREEN}Ready! Listening on your mic.{RESET}")
    print(f"{DIM}Chunks: {CHUNK_SECONDS}s | adaptive noise floor | sound threshold: {SOUND_CONFIDENCE:.0%}{RESET}")
    print(f"{DIM}Press Ctrl+C to stop.{RESET}\n")
    print(f"{'─' * 60}")

    # ── Audio capture loop ──────────────────────────────────────────
    audio_q = queue.Queue()
    silent_blocks = 0
    prev_chunk = np.zeros(OVERLAP_SAMPLES, dtype=np.float32)  # overlap for Whisper
    passage = ""               # current in-progress passage
    passages = []              # completed passages
    last_sounds = frozenset()  # deduplicate consecutive identical sound events
    sound_streak = {}          # label -> consecutive chunk count (for ambient suppression)
    AMBIENT_STREAK = 3         # if a sound appears this many chunks in a row, it's background
    current_speaker = None     # who the real-time ID thinks is talking right now

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

                # ── Real-time speaker identification ─────────────
                sid_event = speaker_id.feed(data)
                if sid_event and sid_event.event_type != "silence":
                    top = sid_event.speakers[0] if sid_event.speakers else None
                    if top and top != current_speaker:
                        sim = sid_event.similarities.get(top, 0.0)
                        parts = [f"{top} {sim:.0%}"]
                        # Show other detected voices (barge-in)
                        for name in sid_event.speakers[1:]:
                            parts.append(f"{name} {sid_event.similarities.get(name, 0.0):.0%}")
                        label = " | ".join(parts)
                        # Inject speaker tag into passage
                        tag = f"[{top}]"
                        if passage:
                            passage += f" {tag}"
                        else:
                            passage = tag
                        print(f"  {CYAN}[{label}]{RESET}")
                        current_speaker = top
                    elif not top and current_speaker is not None:
                        current_speaker = None

                # ── Check each 1s block for silence ──────────────
                block_level = rms(data)
                noise_floor.update(block_level)
                ts = time.strftime("%H:%M:%S")

                if block_level < noise_floor.level:
                    silent_blocks += 1
                    # Flush passage after 2 consecutive silent blocks (~2s)
                    if silent_blocks == 2 and passage:
                        passages.append(passage)
                        print(f"  {BOLD}{'·' * 50}{RESET}")
                        print_wrapped(f"  {GREEN}", passage, f"{RESET}", 70)
                        print(f"  {BOLD}{'·' * 50}{RESET}\n")

                        passage = ""
                        current_speaker = None
                        prev_chunk = np.zeros(OVERLAP_SAMPLES, dtype=np.float32)
                        buffer = np.empty((0,), dtype=np.float32)
                    if silent_blocks == 1:
                        print(f"  {DIM}Silence...{RESET}", flush=True)
                    continue
                else:
                    silent_blocks = 0

                if len(buffer) < CHUNK_SAMPLES:
                    continue

                chunk = buffer[:CHUNK_SAMPLES]
                buffer = buffer[CHUNK_SAMPLES:]

                # ── Bandpass filter: keep 300–3400 Hz (voice band) ──
                chunk = sosfilt(VOICE_BANDPASS, chunk).astype(np.float32)

                level = rms(chunk)

                # ── Sound classification (always run if above energy floor) ──
                sounds = classify_sounds(interpreter, chunk, class_names)
                notable_sounds = [
                    (label, score)
                    for label, score in sounds
                    if label not in BORING_SOUNDS
                    and score > SENSITIVE_SOUNDS.get(label, SOUND_CONFIDENCE)
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
                        whisper_input, beam_size=1, language="en",
                        initial_prompt=WHISPER_PROMPT,
                    )
                    text = " ".join(s.text.strip() for s in segments).strip()
                    for wrong, right in WORD_FIXES.items():
                        text = text.replace(wrong, right)
                    # Filter out prompt hallucinations
                    if text:
                        prompt_words = set(WHISPER_PROMPT.lower().split())
                        text_words = text.lower().replace(",", "").replace(".", "").split()
                        if text_words:
                            prompt_overlap = sum(1 for w in text_words if w in prompt_words)
                            if prompt_overlap / len(text_words) > 0.6:
                                text = ""

                # Save tail of this chunk as overlap for next round
                prev_chunk = chunk[-OVERLAP_SAMPLES:]

                # ── Inject sounds into passage as stage directions ────
                if notable_sounds:
                    sound_labels = frozenset(label for label, _ in notable_sounds)
                    # Only inject if these sounds are different from the last ones
                    if sound_labels != last_sounds:
                        stage_dir = "[" + ", ".join(
                            f"{label.lower()}" for label, score in notable_sounds
                        ) + "]"
                        if passage:
                            passage += f" {stage_dir}"
                        else:
                            passage = stage_dir
                    last_sounds = sound_labels
                else:
                    last_sounds = frozenset()

                # ── Build passage live ────────────────────────────────
                if text:
                    passage = stitch(passage, text)
                    print(f"  {DIM}▌{RESET} {GREEN}{text}{RESET}")

    except KeyboardInterrupt:
        # Flush any in-progress passage
        if passage:
            passages.append(passage)

        print(f"\n\n{YELLOW}Stopped.{RESET}")
        if passages:
            print(f"\n{BOLD}{'═' * 60}{RESET}")
            print(f"{BOLD}  All passages:{RESET}")
            for i, p in enumerate(passages):
                print(f"\n  {DIM}[{i+1}]{RESET}")
                print_wrapped(f"  {GREEN}", p, f"{RESET}", 70)
            print(f"\n{BOLD}{'═' * 60}{RESET}")


if __name__ == "__main__":
    main()
