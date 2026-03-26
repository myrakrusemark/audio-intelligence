# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Single-file real-time audio intelligence pipeline (`demo.py`) combining three on-device ML models:
- **Whisper** (faster-whisper/CTranslate2) for speech-to-text
- **YAMNet** (TFLite via ai-edge-litert) for environmental sound classification (521 categories)
- **WeSpeaker ECAPA-TDNN** (ONNX) for speaker diarization/identification

Everything runs locally on CPU. No cloud APIs.

## Setup and run

```bash
uv venv .venv --python 3.12
uv pip install --python .venv sounddevice numpy faster-whisper ai-edge-litert onnxruntime huggingface-hub
.venv/bin/python3 demo.py
```

Models auto-download to `.cache/` on first run (~100MB total).

## CLI modes

```bash
.venv/bin/python3 demo.py                    # Live listening mode
.venv/bin/python3 demo.py --enroll "Name"     # Enroll a voice (blends with existing)
.venv/bin/python3 demo.py --enroll "Name" -n  # Fresh enrollment (replaces existing)
.venv/bin/python3 demo.py --remove            # List enrolled voices
.venv/bin/python3 demo.py --remove "Name"     # Remove a voice profile
```

## Architecture

The pipeline processes audio in a single event loop:

1. **Audio capture**: 1-second blocks from mic via `sounddevice` callback into a queue
2. **Energy gating**: RMS check skips silent blocks entirely (< `ENERGY_FLOOR`)
3. **Chunk buffering**: Blocks accumulate into `CHUNK_SECONDS`-sized chunks for ML
4. **YAMNet classification**: Runs on every non-silent chunk, produces sound labels
5. **YAMNet-gated Whisper**: STT only runs when YAMNet detects speech labels, preventing hallucinations on non-speech audio
6. **Fuzzy stitching** (`stitch()`): Overlapping transcriptions merged at word boundaries via `SequenceMatcher`
7. **Sound injection**: Non-speech sounds inserted as `[stage directions]` inline in the transcript
8. **Speaker diarization**: Extracts embeddings per passage, clusters with iterative cosine similarity, matches against enrolled voices from `.voices/`
9. **Passage finalization**: 2 consecutive silent blocks trigger passage completion

Key design patterns:
- Voice profiles stored as JSON in `.voices/` with L2-normalized embeddings
- Speaker assignment re-runs from scratch each time (`assign_speakers`) so early passages benefit from later profile refinement (3-pass iterative clustering)
- `compute_fbank()` implements mel-filterbank feature extraction in pure numpy (no librosa/torchaudio dependency)
- Whisper receives `OVERLAP_SECONDS` of prior audio for cross-boundary context
- Post-processing: `WORD_FIXES` dict corrects known mishearings, prompt hallucination filter rejects text that mostly echoes `WHISPER_PROMPT`

## Tuning

All constants are at the top of `demo.py`. Key ones: `CHUNK_SECONDS`, `OVERLAP_SECONDS`, `ENERGY_FLOOR`, `SOUND_CONFIDENCE`, `SPEECH_ENERGY`, `SPEAKER_SIM_THRESHOLD`, `WHISPER_PROMPT`, `WORD_FIXES`, `SENSITIVE_SOUNDS`.
