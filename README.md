# audio-intelligence

On-device speech-to-text + environmental sound classification, running entirely locally. No cloud APIs, no data leaves your machine.

Combines **Whisper** (STT) and **YAMNet** (sound classification) into a single real-time pipeline that produces a unified transcript with environmental sounds woven in as inline `[stage directions]`.

## What it does

- Listens to your microphone in real-time
- Transcribes speech using Whisper (tiny model, ~75MB)
- Classifies 521 environmental sounds using YAMNet (~4MB TFLite model)
- Fuzzy-stitches overlapping chunks into coherent passages
- Injects detected sounds as `[bracketed stage directions]` in the transcript
- Finalizes passages on silence, prints a full session summary on exit

## Example output

```
[12:19:04] ████████░░░░░░░░░░░░
  speech: Okay, this is another test.

[12:19:07] █████████░░░░░░░░░░░
  speech: another test of unending stream of consciousness and

  ··················································
  Okay, this is another test of unending stream of consciousness
  and I just want to see if it works.
  ··················································
```

With environmental sounds interleaved:

```
  ··················································
  Okay now I'm talking but [hands, clapping] while talking and I'm
  really cute. if that will do anything. [hands, clapping] doing it
  again now. This is another unending passage and we'll just see how
  that turns out.
  ··················································
```

Sounds detected independently of speech:

```
[12:34:02] ██░░░░░░░░░░░░░░░░░░
  sounds: Typing 31% | Computer keyboard 28%
```

Animals, interruptions, and chaos:

```
  ··················································
  [cat, domestic animals, pets, meow] [cat] // Fuck! [bark, domestic
  animals, pets, animal, dog]
  ··················································
```

## Features

- **Energy-gated processing** -- silent chunks skip both models entirely
- **YAMNet-gated Whisper** -- STT only runs when YAMNet detects a human voice, preventing hallucinations on non-speech audio
- **Fuzzy transcript stitching** -- overlapping audio chunks are deduplicated at word boundaries using sequence matching
- **Per-sound confidence thresholds** -- reliably accurate sounds (cat, doorbell, alarm) use lower thresholds than the default
- **Whisper prompt biasing** -- `initial_prompt` steers the decoder toward custom vocabulary (names, jargon)
- **Post-processing word fixes** -- catches known misheard words
- **Live passage display** -- transcript builds in-place using ANSI escape codes
- **Passage finalization** -- silence triggers passage completion with clean formatting

## Requirements

- Python 3.12+
- A microphone
- ~80MB disk for model downloads (first run)

## Setup

```bash
uv venv .venv --python 3.12
uv pip install --python .venv sounddevice numpy faster-whisper ai-edge-litert
```

## Run

```bash
.venv/bin/python3 demo.py
```

Press `Ctrl+C` to stop. All collected passages are printed at exit.

## Configuration

Tunable constants at the top of `demo.py`:

| Constant | Default | Description |
|---|---|---|
| `CHUNK_SECONDS` | `3` | Audio chunk size for ML processing |
| `OVERLAP_SECONDS` | `1` | Prior audio fed to Whisper for cross-boundary context |
| `ENERGY_FLOOR` | `0.005` | RMS below this = silence, skip processing |
| `SOUND_CONFIDENCE` | `0.15` | Default YAMNet confidence threshold |
| `SPEECH_ENERGY` | `0.01` | RMS needed before running Whisper |
| `WHISPER_PROMPT` | `"Fathom..."` | Bias Whisper toward specific words/names |
| `WORD_FIXES` | `{"Adam": "Fathom", ...}` | Post-processing replacements for known mishearings |
| `SENSITIVE_SOUNDS` | `{"Cat": 0.06, ...}` | Per-label confidence overrides for accurate-but-quiet sounds |

## How it works

1. Audio is captured in 1-second blocks from the microphone
2. Each block is checked for energy -- silence is detected within ~2 seconds
3. Blocks are buffered into 3-second chunks for ML processing
4. YAMNet classifies the chunk across 521 sound categories
5. If YAMNet detects speech, Whisper transcribes (with 1s overlap from the previous chunk)
6. Sound events are injected as `[stage directions]` into the running passage
7. New speech is fuzzy-merged with the existing passage to avoid duplication
8. The passage is displayed live and finalized when silence is detected

## Models

Both models are downloaded automatically on first run and cached in `.cache/`.

- **Whisper tiny** (CTranslate2/int8) -- ~75MB, runs on CPU
- **YAMNet** (TFLite via ai-edge-litert) -- ~4MB, runs on CPU

For better transcription accuracy, change `"tiny"` to `"small"` (~460MB) or `"medium"` in the `WhisperModel()` call.

## License

MIT
