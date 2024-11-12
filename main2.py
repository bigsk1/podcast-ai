from dotenv import load_dotenv
import os
import re
import argparse
import json
import yt_dlp
import requests
from faster_whisper import WhisperModel
from anthropic import Anthropic
from openai import OpenAI
import logging
from datetime import datetime
import asyncio
from typing import Dict, List
from duckduckgo_search import DDGS
import newspaper
from urllib.parse import urlparse
import html2text
import subprocess


# Configure logging
logging.basicConfig(
    level=os.getenv('LOGGING_LEVEL', 'INFO'),
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('podcast_analyzer.log'),
        logging.StreamHandler()
    ]
)

# Add debug logging for important configuration
def log_config_debug(config):
    logging.debug("Configuration:")
    for key, value in config.items():
        if 'KEY' in key:  # Don't log API keys
            logging.debug(f"{key}: [REDACTED]")
        else:
            logging.debug(f"{key}: {value}")

def check_ffmpeg():
    """Check if FFmpeg is installed"""
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True)
        return True
    except FileNotFoundError:
        logging.error("FFmpeg not found. Please install FFmpeg and add it to your PATH.")
        return False

class PodcastAnalyzer:
    def __init__(self):
        if not check_ffmpeg():
            raise SystemExit("FFmpeg is required but not found. Please install FFmpeg and add it to your PATH.")
        
        load_dotenv()
        self.setup_config()
        self.load_voice_config()
        self.h2t = html2text.HTML2Text()
        self.h2t.ignore_links = True
        
        # Display initial quota
        self.display_elevenlabs_quota()
        
        # Initialize the Whisper model
        self.whisper_model = WhisperModel("base", device="auto", compute_type="auto")

    def display_elevenlabs_quota(self):
        """Display current ElevenLabs API quota"""
        try:
            response = requests.get(
                "https://api.elevenlabs.io/v1/user",
                headers={"xi-api-key": self.config['ELEVENLABS_API_KEY']},
                timeout=30
            )
            response.raise_for_status()
            user_data = response.json()
            character_count = user_data['subscription']['character_count']
            character_limit = user_data['subscription']['character_limit']
            
            # Calculate percentage used
            usage_percent = (character_count / character_limit) * 100 if character_limit > 0 else 0
            
            logging.info(f"ElevenLabs Character Usage: {character_count:,} / {character_limit:,} ({usage_percent:.1f}%)")
            
            # Also print to console in a visually appealing way
            print("\n" + "=" * 50)
            print("ElevenLabs API Usage:")
            print(f"Characters Used: {character_count:,}")
            print(f"Character Limit: {character_limit:,}")
            print(f"Usage: {usage_percent:.1f}%")
            print("=" * 50 + "\n")
            
            return character_count, character_limit
            
        except Exception as e:
            logging.error(f"Could not fetch ElevenLabs quota: {str(e)}")
            print("\nWarning: Could not fetch ElevenLabs quota")
            return None, None
        
    
        
    def setup_config(self):
        """Initialize configuration from environment variables"""
        self.config = {
            'AI_PROVIDER': os.getenv('AI_PROVIDER', 'anthropic'),
            'ANTHROPIC_API_KEY': os.getenv('ANTHROPIC_API_KEY'),
            'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
            'ELEVENLABS_API_KEY': os.getenv('ELEVENLABS_API_KEY'),
            'OLLAMA_HOST': os.getenv('OLLAMA_HOST', 'http://localhost:11434'),
            'MODEL_NAME': os.getenv('MODEL_NAME', 'claude-3-5-sonnet-20241022'),
            'OLLAMA_MODEL': os.getenv('OLLAMA_MODEL', 'llama2'),
            'OUTPUT_DIR': os.getenv('OUTPUT_DIR', 'output'),
            'VOICE1': os.getenv('VOICE1'),
            'VOICE2': os.getenv('VOICE2'),
            'MAX_SEARCH_RESULTS': int(os.getenv('MAX_SEARCH_RESULTS', '3')),
            'MAX_ARTICLE_LENGTH': int(os.getenv('MAX_ARTICLE_LENGTH', '5000')),
            'MAX_AUDIO_LENGTH_SECONDS': int(os.getenv('MAX_AUDIO_LENGTH_SECONDS', '30')),
            'WORDS_PER_MINUTE': int(os.getenv('WORDS_PER_MINUTE', '150')),  # Average speaking rate
            'MAX_CHARS_PER_VOICE': int(os.getenv('MAX_CHARS_PER_VOICE', '250'))  # Limit characters per voice line
        }

    def safe_str(self, obj) -> str:
        """Safely convert any object to a string"""
        try:
            if hasattr(obj, 'content'):
                return str(obj.content)
            elif hasattr(obj, 'text'):
                return str(obj.text)
            else:
                return str(obj)
        except Exception as e:
            logging.error(f"Error converting object to string: {str(e)}")
            return "[Error converting to string]"
    
    async def transcribe_audio(self, audio_path: str) -> Dict:
        """Transcribe audio using faster-whisper"""
        logging.info(f"Transcribing audio: {audio_path}")
        
        # Transcribe the audio
        segments, info = self.whisper_model.transcribe(audio_path, beam_size=5)
        
        # Format the results
        result = {
            "text": "",
            "segments": []
        }
        
        for segment in segments:
            result["text"] += f" {segment.text}"
            result["segments"].append({
                "id": len(result["segments"]),
                "text": segment.text,
                "start": segment.start,
                "end": segment.end
            })
        
        # Save transcription with timestamps
        transcript_path = f"{self.config['OUTPUT_DIR']}/transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(transcript_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
            
        return result

    async def extract_topics(self, transcript: Dict) -> List[str]:
        """Extract main topics from transcript using AI"""
        logging.info("Analyzing transcript for topics")
        
        prompt = f"""
        Extract 4-5 specific factual claims from this transcript.
        List each claim on a separate line.
        Focus only on clear, verifiable statements.
        Do not include any headers, notes, or formatting.

        Example format:
        The speaker uses Pop OS version 22.04
        They play Counter Strike twice per week
        They have been using Linux for 1.5 years

        Transcript:
        {transcript['text']}
        """
        
        try:
            client = Anthropic(api_key=self.config['ANTHROPIC_API_KEY'])
            response = client.messages.create(
                model=self.config['MODEL_NAME'],
                max_tokens=8192,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Clean up the response and extract individual topics
            content = str(response.content)
            
            # Split into lines and clean each one
            topics = []
            for line in content.split('\n'):
                line = line.strip()
                if line and not line.startswith(('Here', 'Note:', '-', '*', '•')):
                    # Clean up any remaining formatting
                    clean_line = re.sub(r'contentblock\(text=|\[|\]|\'|\"', '', line, flags=re.IGNORECASE)
                    if len(clean_line) > 10:
                        topics.append(clean_line)
            
            logging.info(f"Extracted topics: {topics}")
            return topics
            
        except Exception as e:
            logging.error(f"Error extracting topics: {str(e)}")
            return []

    async def fact_check_with_ai(self, transcript: Dict) -> List[Dict]:
        """Analyze transcript for any factual issues using AI knowledge"""
        logging.info("Analyzing transcript for factual accuracy")
        
        prompt = f"""
        Review this content and identify any factual claims that need correction or additional context based on your knowledge.
        Focus on significant factual errors or misleading statements, not minor details.

        Content to analyze:
        {transcript['text']}

        Provide response as a JSON list of objects, each containing:
        {{
            "claim": "the specific claim made",
            "correction": "the correct information",
            "context": "why this is important",
            "confidence": "high/medium/low"
        }}

        Only include claims that need correction. If everything is accurate, return an empty list.
        """
        
        try:
            client = Anthropic(api_key=self.config['ANTHROPIC_API_KEY'])
            response = client.messages.create(
                model=self.config['MODEL_NAME'],
                max_tokens=8192,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Extract and parse the JSON response
            content = str(response.content)
            try:
                corrections = json.loads(content)
                logging.info(f"Found {len(corrections)} items needing correction")
                return corrections
            except json.JSONDecodeError:
                # Try to extract JSON-like content if the response isn't perfect JSON
                import re
                json_match = re.search(r'\[(.*?)\]', content, re.DOTALL)
                if json_match:
                    corrections = json.loads(f"[{json_match.group(1)}]")
                    return corrections
                return []
                
        except Exception as e:
            logging.error(f"Error in fact checking: {str(e)}")
            return []

    def clean_conversation_text(self, text: str) -> str:
        """Clean and format conversation text for audio generation"""
        # Remove any partial sentences
        sentences = text.split('.')
        complete_sentences = [s.strip() + '.' for s in sentences if len(s.strip()) > 10]
        
        # Rejoin with proper spacing
        text = ' '.join(complete_sentences)
        
        # Remove any markdown or special characters
        text = re.sub(r'[*_~`]', '', text)
        
        # Ensure proper end punctuation
        if not text[-1] in '.!?':
            text += '.'
            
        return text

    def process_conversation(self, content: str) -> List[Dict]:
        """Parse AI response content into a structured conversation format."""
        conversation = []
        lines = content.split('\n')
        speaker_pattern = re.compile(r'^(Speaker\s*[12]):\s*(.*)', re.IGNORECASE)

        for line in lines:
            line = line.strip()
            if not line:
                continue
            match = speaker_pattern.match(line)
            if match:
                speaker_label = match.group(1)
                text = match.group(2).strip()
                speaker = "1" if '1' in speaker_label else "2"
                conversation.append({
                    "speaker": speaker,
                    "text": text,
                    "timestamp": f"0:{len(conversation):02d}"  # Example timestamp format
                })
            else:
                logging.warning(f"Line does not match expected format: {line}")

        return conversation

    async def generate_conversation(self, transcript: Dict, fact_checks: List[Dict]) -> Dict:
        """Generate an AI podcast-style discussion incorporating fact checks"""
        logging.info("Generating podcast discussion with fact checking")
        
        # Updated prompt to create a natural, engaging dialogue
        prompt = f"""
        Imagine you're two podcast hosts discussing a YouTube video on the topic. Make it an engaging back-and-forth dialogue. 
        Speak naturally, as if you're sharing insights with each other, and add reactions and follow-up questions to keep the conversation interesting.

        Key points to cover from the transcript:
        {transcript['text'][:1500]}

        Requirements:
        - Create a dynamic conversation, not a summary
        - Use emotions, natural expressions, and reactions like "That's surprising!" or "I hadn’t thought of that"
        - Avoid repeating exact phrases from the transcript
        - Structure as 10 exchanges, with each speaker contributing to the dialogue’s flow.
        - Each speaker should offer their perspective and ask questions to the other to expand the topic

        Format:
        Speaker 1: "Opening comment on the topic..."
        Speaker 2: "Response with additional insight or question..."
        (Alternate for exactly 10 exchanges)
        """
        
        try:
            client = Anthropic(api_key=self.config['ANTHROPIC_API_KEY'])
            response = client.messages.create(
                model=self.config['MODEL_NAME'],
                max_tokens=8192,
                temperature=0.8,  # Slightly increase temperature for more creative responses
                messages=[{"role": "user", "content": prompt}]
            )
            
            content = str(response.content).strip()
            conversation = self.process_conversation(content)  # Using process_conversation function to format the dialogue
            
            if len(conversation) < 10:  # Retry if the conversation is too short
                logging.warning("Generated conversation too short. Retrying with a simpler prompt.")
                return self.create_simple_conversation(transcript['text'][:500])
            
            return {"conversation": conversation}
        
        except Exception as e:
            logging.error(f"Error generating conversation: {str(e)}")
            return {"conversation": []}


    async def generate_audio(self, conversation: Dict) -> List[str]:
        """Generate audio for each line of conversation"""
        logging.info("Generating audio files")
        audio_files = []
        
        # Fixed voice settings within valid ranges
        VOICE_SETTINGS = {
            "default": {
                "stability": 0.85,
                "similarity_boost": 0.75,
                "style": 0.75,              # Fixed to be within 0-1 range
                "use_speaker_boost": True
            }
        }
        
        try:
            initial_count, _ = self.display_elevenlabs_quota()
            
            for i, entry in enumerate(conversation.get('conversation', [])):
                text = entry['text'].strip()
                speaker = entry['speaker']
                voice_id = self.config['VOICE1'] if speaker == "1" else self.config['VOICE2']
                
                # Ensure text is properly formatted
                if not text[-1] in '.!?':
                    text += '.'
                    
                logging.info(f"Generating audio {i+1} for speaker {speaker}: {text}")
                
                try:
                    response = requests.post(
                        f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
                        headers={"xi-api-key": self.config['ELEVENLABS_API_KEY']},
                        json={
                            "text": text,
                            "model_id": "eleven_monolingual_v1",
                            "voice_settings": VOICE_SETTINGS["default"]
                        }
                    )
                    
                    if response.status_code == 200:
                        filename = f"output/segment_{i:02d}_speaker{speaker}.mp3"
                        with open(filename, 'wb') as f:
                            f.write(response.content)
                        audio_files.append(filename)
                        logging.info(f"Generated {filename}: {text[:100]}")
                    else:
                        logging.error(f"Failed to generate audio: {response.text}")
                        
                except Exception as e:
                    logging.error(f"Error generating audio segment {i}: {str(e)}")
                    continue
                    
                # Small delay between requests
                await asyncio.sleep(0.5)
            
            # Log final results
            final_count, _ = self.display_elevenlabs_quota()
            if initial_count and final_count:
                used = final_count - initial_count
                logging.info(f"Generated {len(audio_files)} audio files using {used} characters")
            
            return audio_files
            
        except Exception as e:
            logging.error(f"Error in audio generation: {str(e)}")
            return []

    async def search_topic(self, topic: str) -> List[Dict]:
        """Search the web for information about a topic with rate limiting"""
        logging.info(f"Searching for topic: {topic}")
        results = []
        
        try:
            topic_str = str(topic).strip()
            if not topic_str:
                return results
                
            # Add delay to avoid rate limiting
            await asyncio.sleep(2)
            
            from duckduckgo_search import AsyncDDGS
            async with AsyncDDGS() as ddgs:
                try:
                    search_results = [r async for r in ddgs.text(
                        topic_str, 
                        max_results=2  # Reduced to avoid rate limiting
                    )]
                    
                    for result in search_results:
                        try:
                            article = newspaper.Article(result['link'])
                            article.download()
                            article.parse()
                            content = self.h2t.handle(article.text)[:2000]  # Limit content length
                            
                            results.append({
                                'title': result['title'],
                                'link': result['link'],
                                'content': content,
                                'source': urlparse(result['link']).netloc
                            })
                        except Exception as e:
                            logging.warning(f"Failed to process article {result['link']}: {str(e)}")
                            continue
                            
                except Exception as e:
                    logging.warning(f"Search failed for topic '{topic_str}': {str(e)}")
                    # Continue with AI fact-checking even if search fails
                    
        except Exception as e:
            logging.error(f"Error in search_topic: {str(e)}")
            
        return results

    def find_relevant_excerpt(self, text: str, topic: str) -> str:
        """Find the most relevant excerpt from the transcript for a given topic"""
        sentences = text.split('.')
        topic_words = set(topic.lower().split())
        
        # Find the sentence with the most matching words
        best_match = ""
        max_matches = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            sentence_words = set(sentence.lower().split())
            matches = len(topic_words.intersection(sentence_words))
            
            if matches > max_matches:
                max_matches = matches
                best_match = sentence
        
        if not best_match:
            return "No direct reference found"
        
        return best_match

    async def analyze_search_results(self, topic: str, search_results: List[Dict], transcript_excerpt: str) -> Dict:
        """Compare search results with transcript claims using AI"""
        prompt = f"""
        Compare the following transcript excerpt with web search results about the topic.
        Identify agreements, disagreements, and additional context.
        
        Topic: {topic}
        
        Transcript excerpt:
        {transcript_excerpt}
        
        Search Results:
        {json.dumps(search_results, indent=2)}
        
        Provide analysis in JSON format:
        {{
            "agreement": "what matches between sources",
            "disagreement": "what conflicts between sources",
            "additional_context": "important additional information",
            "confidence": "high/medium/low based on source quality and consistency",
            "sources": ["list of most relevant sources"]
        }}
        """
        
        if self.config['AI_PROVIDER'] == 'anthropic':
            client = Anthropic(api_key=self.config['ANTHROPIC_API_KEY'])
            response = client.messages.create(
                model=self.config['MODEL_NAME'],
                max_tokens=8192,
                temperature=0.7,
                messages=[{"role": "user", "content": prompt}]
            )
            return json.loads(response.content)
        # Add similar handlers for OpenAI and Ollama


    async def generate_extended_conversation(self, summary: str, fact_checks: List[Dict]) -> Dict:
        """Generate a longer conversation when the first attempt is too short"""
        prompt = f"""
        Create a detailed back-and-forth dialogue discussing this content.
        Make it natural and engaging, like two podcast hosts having a real conversation. Do not just repeat the transcript and speak in first person, summarize yourself.

        The dialogue should include:
        1. Introduction of the topic
        2. Discussion of main points
        3. Sharing of perspectives
        4. Interesting observations
        5. Relevant fact-checks or corrections
        6. Final thoughts and conclusions

        Generate EXACTLY 10 exchanges (20 lines total, alternating speakers).
        Keep each line between 15-30 words.

        Content:
        {summary}

        Format as:
        Speaker 1: [First observation or point]
        Speaker 2: [Response and new point]
        (continue alternating 10 times)
        """
        
        try:
            client = Anthropic(api_key=self.config['ANTHROPIC_API_KEY'])
            response = client.messages.create(
                model=self.config['MODEL_NAME'],
                max_tokens=8192,
                temperature=0.7,
                messages=[{"role": "user", "content": prompt}]
            )
            
            content = str(response.content)
            lines = content.split('\n')
            conversation = []
            speaker = "1"
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Clean up the line
                text = re.sub(r'^(?:Speaker\s*)?[12]:\s*', '', line).strip()
                
                if text and len(text) >= 10 and len(text) <= 150:
                    conversation.append({
                        "speaker": speaker,
                        "text": text,
                        "timestamp": f"0:{len(conversation):02d}"
                    })
                    speaker = "2" if speaker == "1" else "1"
                    
            return {
                "conversation": conversation,
                "summary": {
                    "content": summary,
                    "factual_corrections": [
                        f.get('analysis', {}).get('corrections', '')
                        for f in fact_checks
                        if f.get('analysis', {}).get('corrections')
                    ]
                }
            }
            
        except Exception as e:
            logging.error(f"Error generating extended conversation: {str(e)}")
            return self.create_simple_conversation(summary)
    
    def create_simple_conversation(self, text: str) -> Dict:
        """Create a basic conversation about the content"""
        # Extract first meaningful sentence
        first_sentence = next((s.strip() for s in text.split('.') if len(s.strip()) > 20), text[:300])
        
        return {
            "conversation": [
                {
                    "speaker": "1",
                    "text": f"I just listened to this interesting content about {first_sentence}",
                    "timestamp": "0:00"
                },
                {
                    "speaker": "2",
                    "text": "What were the main points that stood out to you?",
                    "timestamp": "0:05"
                },
                {
                    "speaker": "1",
                    "text": f"Well, one key point was about {text[200:400].split('.')[0]}",
                    "timestamp": "0:10"
                },
                {
                    "speaker": "2",
                    "text": "That's interesting. What else did they discuss?",
                    "timestamp": "0:15"
                }
            ]
        }


    def load_voice_config(self):
        """Load voice configurations from voices.json"""
        try:
            with open('voices.json', 'r') as f:
                self.voices = json.load(f)
            
            # Validate selected voices exist in config
            if self.config['VOICE1'] and self.config['VOICE2']:
                voice1_exists = any(v['id'] == self.config['VOICE1'] for v in self.voices['voices'])
                voice2_exists = any(v['id'] == self.config['VOICE2'] for v in self.voices['voices'])
                
                if not (voice1_exists and voice2_exists):
                    raise ValueError("One or both selected voices not found in voices.json")
            else:
                # Default to first two voices if not specified
                self.config['VOICE1'] = self.voices['voices'][0]['id']
                self.config['VOICE2'] = self.voices['voices'][1]['id']
                logging.warning("No voices specified in .env, using first two voices from voices.json")
                
            # Get voice names for logging
            self.voice1_name = next(v['name'] for v in self.voices['voices'] if v['id'] == self.config['VOICE1'])
            self.voice2_name = next(v['name'] for v in self.voices['voices'] if v['id'] == self.config['VOICE2'])
            logging.info(f"Using voices: Speaker 1: {self.voice1_name}, Speaker 2: {self.voice2_name}")
            
        except FileNotFoundError:
            logging.error("voices.json not found")
            raise
        except json.JSONDecodeError:
            logging.error("Invalid voices.json format")
            raise
        except Exception as e:
            logging.error(f"Error loading voice config: {str(e)}")
            raise

    def sanitize_filename(self, filename: str) -> str:
        """Remove invalid characters from filename"""
        # Remove or replace invalid characters
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '_')
        return filename

    async def download_podcast(self, url: str) -> str:
        """Download podcast audio using yt-dlp"""
        logging.info(f"Downloading podcast from: {url}")
        
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'm4a',
            }],
            'outtmpl': f"{self.config['OUTPUT_DIR']}/%(title).100s.%(ext)s",  # Limit title length
            'restrictfilenames': True,  # Replace invalid characters
            'windowsfilenames': True,   # Ensure Windows compatibility
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Get video info first
                info = ydl.extract_info(url, download=False)
                # Sanitize the filename
                safe_title = self.sanitize_filename(info['title'])
                # Update output template with sanitized title
                ydl_opts['outtmpl'] = f"{self.config['OUTPUT_DIR']}/{safe_title}.%(ext)s"
                
                # Now download with sanitized filename
                with yt_dlp.YoutubeDL(ydl_opts) as ydl2:
                    info = ydl2.extract_info(url, download=True)
                    output_path = f"{self.config['OUTPUT_DIR']}/{safe_title}.m4a"
                    logging.info(f"Downloaded to: {output_path}")
                    return output_path
                    
        except Exception as e:
            logging.error(f"Download failed: {str(e)}")
            # Check if the file was partially downloaded
            if 'info' in locals() and 'title' in info:
                safe_title = self.sanitize_filename(info['title'])
                m4a_path = f"{self.config['OUTPUT_DIR']}/{safe_title}.m4a"
                if os.path.exists(m4a_path):
                    logging.info(f"Using partially downloaded file: {m4a_path}")
                    return m4a_path
            raise

    def display_audio_sequence(self, audio_files: List[str]):
        """Display the sequence of audio files and how to play them"""
        if not audio_files:
            print("\nNo audio files were generated.")
            return
            
        print("\n" + "=" * 50)
        print("Audio Sequence Generated:")
        print("=" * 50)
        
        for audio_file in audio_files:
            filename = os.path.basename(audio_file)
            speaker_num = re.search(r'speaker(\d)', filename).group(1)
            voice_name = self.voice1_name if speaker_num == "1" else self.voice2_name
            print(f"- {voice_name}: {filename}")
        
        print("\nThe files are numbered in sequence and should be played in order.")
        print("Each file represents one part of the conversation.")
        print("=" * 50 + "\n")

async def main():
    parser = argparse.ArgumentParser(description='Podcast Analysis CLI')
    parser.add_argument('url', help='URL of the podcast to analyze')
    parser.add_argument('--no-audio', action='store_true', help='Skip audio generation')
    args = parser.parse_args()
    
    analyzer = PodcastAnalyzer()
    os.makedirs(analyzer.config['OUTPUT_DIR'], exist_ok=True)
    
    try:
        # Download and transcribe
        logging.info("Starting podcast analysis...")
        audio_path = await analyzer.download_podcast(args.url)
        logging.info(f"Downloaded audio to: {audio_path}")
        
        # In main():
        transcript = await analyzer.transcribe_audio(audio_path)
        fact_checks = await analyzer.fact_check_with_ai(transcript)
        conversation = await analyzer.generate_conversation(transcript, fact_checks)
        
        # Validate conversation
        if not conversation or not conversation.get('conversation'):
            logging.error("Failed to generate valid conversation")
            return
        
        if len(conversation['conversation']) < 2:
            logging.error("Generated conversation too short")
            return
        
        # Generate audio files
        if not args.no_audio:
            logging.info("Generating audio files...")
            audio_files = await analyzer.generate_audio(conversation)
            
            if audio_files:
                logging.info(f"Successfully generated {len(audio_files)} audio files:")
                for audio_file in audio_files:
                    print(f"- {os.path.basename(audio_file)}")
                
                # Display information about the generated files
                print("\nAudio files have been generated in the output directory.")
                print("Play them in numerical order for the complete conversation.")
            else:
                logging.error("No audio files were generated")
        
        logging.info("Analysis complete!")
        
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())