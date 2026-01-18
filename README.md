# Pipecat Voice Assistant Quickstart

This project provides a robust foundation for building real-time voice AI assistants using the [Pipecat](https://github.com/pipecat-ai/pipecat) framework. It integrates industry-leading services for speech recognition, language modeling, and speech synthesis.

## Features

- **Real-time Voice Pipeline:** Seamless integration of STT, LLM, and TTS.
- **Advanced Voice Activity Detection (VAD):** Uses Silero VAD for accurate speech detection.
- **Smart Turn-Taking:** Local smart turn analyzer for natural conversation flow.
- **Multi-Transport Support:** Compatible with Daily and WebRTC for flexible deployment.
- **Modular Architecture:** Easily swap AI providers or customize the bot's behavior.

## AI Stack

- **STT:** [Deepgram](https://deepgram.com/) (Nova-2 model)
- **LLM:** [OpenAI](https://openai.com/) (GPT-4o)
- **TTS:** [Cartesia](https://cartesia.ai/) (Sonic-3 model)

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (recommended) or `pip`

## Installation

1. **Clone the repository:**
   ```bash
   git clone git@github.com:satyam844/pipecat-voicebot-basic.git
   cd pipecat-quickstart
   ```

2. **Install dependencies:**
   Using `uv`:
   ```bash
   uv sync
   ```
   Or using `pip`:
   ```bash
   pip install .
   ```

## Configuration

1. **Environment Variables:**
   Copy the example environment file:
   ```bash
   cp env.example .env
   ```
2. **API Keys:**
   Open `.env` and provide your credentials:
   - `DEEPGRAM_API_KEY`
   - `OPENAI_API_KEY`
   - `CARTESIA_API_KEY`
   - `DAILY_API_KEY` (if using Daily transport)

## Usage

Run the assistant using `uv`:

```bash
uv run bot.py
```

## Project Structure

- `bot.py`: Main application script containing the pipeline definition and bot logic.
- `pyproject.toml`: Project metadata and dependency management.
- `recordings/`: Directory where audio recordings of sessions are stored.

## License

This project is licensed under the BSD 2-Clause License. See the [LICENSE](LICENSE) file for details.