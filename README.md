# AI Podcast Generator


A CLI tool that creates AI-generated podcast discussions from YouTube videos. It downloads videos, transcribes them, analyzes the content, and generates a natural conversation between two AI voices discussing the content.

## Features

- Downloads YouTube videos/audio using yt-dlp
- Transcribes audio using Faster Whisper
- Generates natural conversations styles using Claude AI
- Converts text to speech using ElevenLabs voices - have 2 voices talking about the youtube video
- Fact-checks content using AI if enabled in .env
- Generates audio files for each part of the conversation

## Prerequisites

Tested in Windows 

- Python 3.10+
- Nvidia GPU for whisper (optional)
- FFmpeg installed and in PATH
- Nvidia cuDNN installed to path (for nvidia gpu)
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
# ELEVENLABS VOICE ID'S
VOICE1=NYC9WEgkq1u4jiqBseQ9    # male
VOICE2=b0uJ9TWzQss61d8f2OWX   # female


# AI Model Settings
AI_PROVIDER=anthropic
ANTHROPIC_API_KEY=your_key_here
MODEL_NAME=claude-3-5-sonnet-20241022

# Podcast Generation Settings
MIN_EXCHANGES=4                    # Minimum number of back-and-forth exchanges - important setting
MAX_EXCHANGES=20                   # Maximum number of exchanges                - important setting
MIN_SENTENCES_PER_EXCHANGE=2       # Minimum sentences per speaker turn
MAX_SENTENCES_PER_EXCHANGE=4       # Maximum sentences per speaker turn
EXCHANGE_LENGTH_MIN_WORDS=20       # Minimum words per exchange
EXCHANGE_LENGTH_MAX_WORDS=150      # Maximum words per exchange

# Audio Length Control
TARGET_LENGTH_MINUTES=3            # Target length for final podcast (in minutes)  - important setting
LENGTH_FLEXIBILITY=0.2             # Allowed deviation from target (20% = ±36 seconds for 3 min target)
SOURCE_LENGTH_RATIO=0.2            # Target output length as ratio of source (0.2 = 20% of original)
MIN_PODCAST_LENGTH=2               # Minimum podcast length in minutes   - important setting
MAX_PODCAST_LENGTH=10              # Maximum podcast length in minutes   - important setting

# Audio Generation Settings
MAX_CHARS_PER_VOICE=2000          # Maximum characters per voice clip
PAUSE_BETWEEN_EXCHANGES=1          # Seconds of pause between exchanges

# Content Coverage
COVERAGE_STYLE=humor             # comprehensive, summary, or highlights, humor
FACT_CHECK_ENABLED=false         # Enable AI fact checking
FACT_CHECK_STYLE=balanced        # balanced, critical, or supportive

# Model Settings
TEMPERATURE=0.7
MAX_TOKENS=8192

LOGGING_LEVEL=DEBUG

# Output Directory
OUTPUT_DIR=output
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

Generate but don't merge:
```bash
python main.py --no-merge "https://www.youtube.com/watch?v=video_id"
```

## Output

The tool generates:
- Transcription of the video
- AI-generated conversation about the content with your selected voices
- Audio files for each part of the conversation + auto merge when finished
- Fact-checking analysis (if enabled)
- Min and Max length of audio podcast clips

Output files are saved in the `output` directory.

to merge all audio files to one audio file manually

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
## Examples


<audio controls>
    <source src="https://aicodelabs.io/silo.mp3" type="audio/mpeg">
    Your browser does not support the audio element.
</audio>

---

<audio controls>
    <source src="https://aicodelabs.io/merged.mp3" type="audio/mpeg">
    Your browser does not support the audio element.
</audio>


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

## In Progress

1. Getting audio clips longer
2. Audio covers more of the transcript and youtube conversation
3. Adding Openai 
4. Adding ollama
5. Add web search into fact checking of podcast

## Troubleshooting

### Could not locate cudnn_ops64_9.dll

```bash
Could not locate cudnn_ops64_9.dll. Please make sure it is in your library path!
Invalid handle. Cannot load symbol cudnnCreateTensorDescriptor
```

To resolve this:

Install cuDNN: Download cuDNN from the NVIDIA cuDNN page https://developer.nvidia.com/cudnn

Here’s how to add it to the PATH:

Open System Environment Variables:

Press Win + R, type sysdm.cpl, and hit Enter. Go to the Advanced tab, and click on Environment Variables. Edit the System PATH Variable:

In the System variables section, find the Path variable, select it, and click Edit. Click New and add the path to the bin directory where cudnn_ops64_9.dll is located. Based on your setup, you would add:

```bash
C:\Program Files\NVIDIA\CUDNN\v9.5\bin\12.6
```

Apply and Restart:

Click OK to close all dialog boxes, then restart your terminal (or any running applications) to apply the changes. Verify the Change:

Open a new terminal and run

```bash
where cudnn_ops64_9.dll
```

## pyaudio codec issue

Make sure you have ffmpeg inside and adding to PATH on windows terminal ( winget install ffmpeg )
