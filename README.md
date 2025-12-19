# Pipecat Tests

Quick project to test Pipecat updates against examples for Speechmatics.

## Setup

```shell
# Copy env file (and add your keys)
cp .env.example .env

# Install dependencies
uv sync

# Run an example (our TTS example)
uv run examples/bot-sarah.py

# Run an example (ElevenLabs TTS example)
uv run examples/bot-humphrey.py
```

Open a browser at `http://localhost:7860/client/` to connect to the bot.
