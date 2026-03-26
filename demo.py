#!/usr/bin/env python3
"""
On-device STT + Environmental Sound Classification + Speaker Diarization Demo
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
ENERGY_FLOOR = 0.005       # RMS below this = silence, skip entirely
SOUND_CONFIDENCE = 0.15    # default threshold for most sounds

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
SPEAKER_MIN_AUDIO = int(SAMPLE_RATE * 1.5)  # need 1.5s of speech to diarize

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


def speaker_status_line(profiles, confidences):
    """Render the speaker status bar."""
    if not profiles:
        return f"  {DIM}Speakers: (none detected){RESET}"
    parts = []
    for sid in sorted(profiles.keys()):
        name = SPEAKER_NAMES[sid] if sid < len(SPEAKER_NAMES) else str(sid)
        conf = confidences.get(sid, 0.0)
        color = GREEN if conf > 0.6 else YELLOW if conf > 0.4 else RED
        parts.append(f"Person {name} {color}{conf:.0%}{RESET}")
    return f"  {BOLD}Speakers:{RESET} " + f" {DIM}|{RESET} ".join(parts)


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

    print(f"\n{GREEN}Ready! Listening on your mic.{RESET}")
    print(f"{DIM}Chunks: {CHUNK_SECONDS}s | silence gate: RMS < {ENERGY_FLOOR} | sound threshold: {SOUND_CONFIDENCE:.0%}{RESET}")
    print(f"{DIM}Press Ctrl+C to stop.{RESET}\n")
    print(f"{'─' * 60}")

    # ── Audio capture loop ──────────────────────────────────────────
    audio_q = queue.Queue()
    silent_blocks = 0
    prev_chunk = np.zeros(OVERLAP_SAMPLES, dtype=np.float32)  # overlap for Whisper
    passage = ""               # current in-progress passage
    passages = []              # completed passages
    lines_to_clear = 0        # lines to wipe for live passage redraw
    last_sounds = frozenset()  # deduplicate consecutive identical sound events

    # Speaker diarization state
    passage_audio_buf = np.empty((0,), dtype=np.float32)  # audio for current passage
    passage_embeddings = []    # one embedding per speech passage only
    passage_has_speech = []    # bool per passage — False = ambient only
    speaker_labels = []        # speaker ID per passage (None for ambient)
    speaker_profiles = {}      # speaker_id -> averaged embedding
    speaker_confidences = {}   # speaker_id -> avg cosine sim

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

                # ── Check each 1s block for silence ──────────────
                block_level = rms(data)
                ts = time.strftime("%H:%M:%S")

                if block_level < ENERGY_FLOOR:
                    silent_blocks += 1
                    # Flush passage after 2 consecutive silent blocks (~2s)
                    if silent_blocks == 2 and passage:
                        has_speech = len(passage_audio_buf) >= SPEAKER_MIN_AUDIO

                        if has_speech:
                            emb = extract_speaker_embedding(speaker_session, passage_audio_buf)
                            passage_embeddings.append(emb)

                        passages.append(passage)
                        passage_has_speech.append(has_speech)

                        # Re-cluster only speech passages
                        speaker_labels_speech, speaker_profiles, speaker_confidences = \
                            assign_speakers(passage_embeddings)

                        # Map back: speech passages get speaker labels, ambient gets None
                        speaker_labels = []
                        si = 0
                        for hs in passage_has_speech:
                            if hs:
                                speaker_labels.append(speaker_labels_speech[si])
                                si += 1
                            else:
                                speaker_labels.append(None)

                        # Wipe live draft and reprint finalized passage
                        if lines_to_clear > 0:
                            sys.stdout.write(f"\033[{lines_to_clear}A\033[J")

                        if has_speech:
                            cur_label = speaker_labels[-1]
                            name = SPEAKER_NAMES[cur_label] if cur_label < len(SPEAKER_NAMES) else str(cur_label)
                            label_str = f"  {CYAN}[Person {name}]{RESET}"
                        else:
                            label_str = f"  {DIM}[Ambient]{RESET}"
                        print(f"  {BOLD}{'·' * 50}{RESET}")
                        print(label_str)
                        print_wrapped(f"  {GREEN}", passage, f"{RESET}", 70)
                        print(f"  {BOLD}{'·' * 50}{RESET}\n")

                        passage = ""
                        passage_audio_buf = np.empty((0,), dtype=np.float32)
                        lines_to_clear = 0
                        prev_chunk = np.zeros(OVERLAP_SAMPLES, dtype=np.float32)
                        buffer = np.empty((0,), dtype=np.float32)
                    if silent_blocks == 1:
                        print(f"  {DIM}Silence...{RESET}", flush=True)
                    if len(buffer) < CHUNK_SAMPLES:
                        continue
                else:
                    silent_blocks = 0

                if len(buffer) < CHUNK_SAMPLES:
                    continue

                chunk = buffer[:CHUNK_SAMPLES]
                buffer = buffer[CHUNK_SAMPLES:]

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
                    # Skip Whisper if non-speech sounds dominate (prevents hallucinations)
                    speech_score = max(
                        (score for label, score in sounds if label in SPEECH_SOUNDS),
                        default=0.0,
                    )
                    nonspeech_score = max(
                        (score for label, score in sounds if label not in SPEECH_SOUNDS and label not in BORING_SOUNDS),
                        default=0.0,
                    )
                    if nonspeech_score > speech_score:
                        pass  # clapping/yelling/etc — let YAMNet handle it, skip Whisper
                    else:
                        whisper_input = np.concatenate([prev_chunk, chunk])
                        segments, info = whisper.transcribe(
                            whisper_input, beam_size=1, language="en",
                            vad_filter=True, initial_prompt=WHISPER_PROMPT,
                        )
                        text = " ".join(s.text.strip() for s in segments).strip()
                        for wrong, right in WORD_FIXES.items():
                            text = text.replace(wrong, right)
                        # Filter out prompt hallucinations
                        prompt_words = set(WHISPER_PROMPT.lower().split())
                        text_words = set(text.lower().replace(",", "").replace(".", "").split())
                        if text_words and text_words.issubset(prompt_words):
                            text = ""

                # Save tail of this chunk as overlap for next round
                prev_chunk = chunk[-OVERLAP_SAMPLES:]

                # Accumulate audio for speaker embedding
                if text:
                    passage_audio_buf = np.concatenate([passage_audio_buf, chunk])

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
                    display = wrap_text(passage, 70)
                    # Wipe previous draft
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

    except KeyboardInterrupt:
        # Flush any in-progress passage
        if passage:
            has_speech = len(passage_audio_buf) >= SPEAKER_MIN_AUDIO
            if has_speech:
                emb = extract_speaker_embedding(speaker_session, passage_audio_buf)
                passage_embeddings.append(emb)
            passages.append(passage)
            passage_has_speech.append(has_speech)

            speaker_labels_speech, speaker_profiles, speaker_confidences = \
                assign_speakers(passage_embeddings)
            speaker_labels = []
            si = 0
            for hs in passage_has_speech:
                if hs:
                    speaker_labels.append(speaker_labels_speech[si])
                    si += 1
                else:
                    speaker_labels.append(None)

        print(f"\n\n{YELLOW}Stopped.{RESET}")
        if passages:
            print(f"\n{BOLD}{'═' * 60}{RESET}")
            print(f"{BOLD}  All passages:{RESET}")
            for i, p in enumerate(passages):
                lbl = speaker_labels[i] if i < len(speaker_labels) else None
                if lbl is not None:
                    name = SPEAKER_NAMES[lbl] if lbl < len(SPEAKER_NAMES) else str(lbl)
                    label_str = f"{CYAN}Person {name}{RESET}"
                else:
                    label_str = f"{DIM}Ambient{RESET}"
                print(f"\n  {DIM}[{i+1}]{RESET} {label_str}")
                print_wrapped(f"  {GREEN}", p, f"{RESET}", 70)
            print(f"\n{speaker_status_line(speaker_profiles, speaker_confidences)}")
            print(f"\n{BOLD}{'═' * 60}{RESET}")


if __name__ == "__main__":
    main()
