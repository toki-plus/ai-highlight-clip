# AI Highlight Clip

An AI-assisted workflow for turning long-form video into candidate highlight clips.

The project combines speech recognition, LLM-assisted content scoring, timeline selection, video clipping, and subtitle generation. It helps teams working with courses, interviews, and live recordings reduce the time required for initial review. Final editorial and publishing decisions remain human responsibilities.

## Context

Finding reusable moments in hours of video normally requires watching the full recording, noting timestamps, cutting clips, and recreating titles and subtitles. The process is time-consuming and editorial criteria can vary between reviewers.

AI Highlight Clip separates the initial review into observable stages while keeping keywords, target duration, and clip count configurable.

## Intended Users

- Course, interview, podcast, and live-content teams
- Operations teams processing large volumes of long-form video
- Developers studying AI-assisted video understanding and editing

## Capabilities

- Transcribe video with Whisper and preserve timestamps
- Score candidate segments using prompts, keywords, and surrounding context
- Generate candidate ranges and draft titles
- Clip media and render subtitles with FFmpeg
- Process a single file or a directory
- Display task progress and logs in a PyQt5 interface

## Workflow

```text
Long-form video
    -> Speech transcription
    -> Time-aligned transcript
    -> LLM-assisted scoring
    -> Candidate time ranges
    -> FFmpeg clipping
    -> Human review and export
```

## Technical Design

- `LLMClient`: model calls and structured result handling
- `SubtitleService`: transcription and subtitle processing
- `AIPipeline`: scoring, title, and clipping orchestration
- `Processor`: media-file processing
- `MainWindow`: configuration, task state, and output management

Core technologies: Python, OpenAI Whisper, DashScope, Pydantic, FFmpeg, and PyQt5.

## Quick Start

### Requirements

- Python 3.8+
- FFmpeg available on `PATH` (a compatible Windows binary is included)
- A DashScope API key

```bash
git clone https://github.com/toki-plus/ai-highlight-clip.git
cd ai-highlight-clip
python -m venv venv
```

After activating the virtual environment:

```bash
pip install -r requirements.txt
python yanqu_highlight_clip.py
```

The API key can be configured in the application. Never commit real credentials.

## Limitations

- Model scores are review suggestions, not replacements for editorial judgment.
- Output quality depends on recording quality, language, model behavior, and prompts.
- Long videos require significant compute and disk resources.
- The current version does not include comprehensive automated tests or CI.

## What This Project Demonstrates

The project validates an end-to-end path from unstructured long-form video to structured text, candidate clips, and human approval instead of attempting to remove the editor from the workflow.

## License

See [LICENSE](./LICENSE).
