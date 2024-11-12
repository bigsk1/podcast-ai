# AI Podcast Generator

UNDER ACTIVATE DEVELOPMENT! Maybe have bugs 

A CLI tool that creates AI-generated podcast discussions from YouTube videos. It downloads videos, transcribes them, analyzes the content, and generates a natural conversation between two AI voices discussing the content.

## Features

- Downloads YouTube videos/audio using yt-dlp
- Transcribes audio using Faster Whisper
- Generates natural conversations using Claude AI
- Converts text to speech using ElevenLabs voices
- Fact-checks content using AI
- Generates audio files for each part of the conversation

## Prerequisites

- Python 3.10+
- Nvidia GPU for whisper
- FFmpeg installed and in PATH
- Nvidia cuDNN installed to path
- ElevenLabs API key
- Anthropic (Claude) API key
- voices.json configuration file

## Installation

1. Clone the repository
```bash
git clone https://github.com/bigsk1/podcast-ai.git
cd podcast-ai
```

2. Create and activate virtual environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Create .env file with your API keys:
```env
AI_PROVIDER=anthropic
ANTHROPIC_API_KEY=your_key
ELEVENLABS_API_KEY=your_key
VOICE1=your_voice_id_1
VOICE2=your_voice_id_2
MAX_AUDIO_LENGTH_SECONDS=120
WORDS_PER_MINUTE=400
MAX_CHARS_PER_VOICE=1000
```

## Usage

Basic usage:
```bash
python main.py "https://www.youtube.com/watch?v=video_id"
```

Skip audio generation:
```bash
python main.py --no-audio "https://www.youtube.com/watch?v=video_id"
```

## Output

The tool generates:
- Transcription of the video
- AI-generated conversation about the content
- Audio files for each part of the conversation
- Fact-checking analysis (if enabled)

Output files are saved in the `output` directory.

to merge all audio files to one audio file

```bash
python merge_audio.py output conversation.mp3 
```

## Configuration

### voices.json
Create a voices.json file with your ElevenLabs voice configurations:
```json
{
    "voices": [
        {
            "id": "voice_id_1",
            "name": "Voice 1 Name"
        },
        {
            "id": "voice_id_2",
            "name": "Voice 2 Name"
        }
    ]
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [yt-dlp](https://github.com/yt-dlp/yt-dlp) for video downloading
- [Faster Whisper](https://github.com/guillaumekln/faster-whisper) for transcription
- [ElevenLabs](https://elevenlabs.io/) for text-to-speech
- [Anthropic](https://www.anthropic.com/) for Claude AI

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request