from dotenv import load_dotenv
import os
import re
import argparse
import json
import yt_dlp
import requests
from faster_whisper import WhisperModel
from anthropic import Anthropic
import logging
from datetime import datetime
import asyncio
from typing import Dict, List
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
            # Core Settings
            'AI_PROVIDER': os.getenv('AI_PROVIDER', 'anthropic'),
            'ANTHROPIC_API_KEY': os.getenv('ANTHROPIC_API_KEY'),
            'XAI_API_KEY': os.getenv('XAI_API_KEY'),
            'XAI_BASE_URL': os.getenv('XAI_BASE_URL', 'https://api.x.ai'),
            'MODEL_NAME': os.getenv('MODEL_NAME', 'claude-3-5-sonnet-20241022'),
            'OUTPUT_DIR': os.getenv('OUTPUT_DIR', 'output'),
            
            # Model Settings
            'TEMPERATURE': float(os.getenv('TEMPERATURE', '0.7')),
            'MAX_TOKENS': int(os.getenv('MAX_TOKENS', '8192')),
            
            # Voice Settings
            'VOICE1': os.getenv('VOICE1'),
            'VOICE2': os.getenv('VOICE2'),
            'ELEVENLABS_API_KEY': os.getenv('ELEVENLABS_API_KEY'),
            
            # Podcast Generation Settings
            'MIN_EXCHANGES': int(os.getenv('MIN_EXCHANGES', '4')),
            'MAX_EXCHANGES': int(os.getenv('MAX_EXCHANGES', '20')),
            'MIN_SENTENCES_PER_EXCHANGE': int(os.getenv('MIN_SENTENCES_PER_EXCHANGE', '2')),
            'MAX_SENTENCES_PER_EXCHANGE': int(os.getenv('MAX_SENTENCES_PER_EXCHANGE', '4')),
            'EXCHANGE_LENGTH_MIN_WORDS': int(os.getenv('EXCHANGE_LENGTH_MIN_WORDS', '20')),
            'EXCHANGE_LENGTH_MAX_WORDS': int(os.getenv('EXCHANGE_LENGTH_MAX_WORDS', '100')),
            
            # Length Control
            'TARGET_LENGTH_MINUTES': float(os.getenv('TARGET_LENGTH_MINUTES', '3')),
            'SOURCE_LENGTH_RATIO': float(os.getenv('SOURCE_LENGTH_RATIO', '0.2')),
            'MIN_PODCAST_LENGTH': float(os.getenv('MIN_PODCAST_LENGTH', '2')),
            'MAX_PODCAST_LENGTH': float(os.getenv('MAX_PODCAST_LENGTH', '10')),
            
            # Audio Settings
            'MAX_CHARS_PER_VOICE': int(os.getenv('MAX_CHARS_PER_VOICE', '2000')),
            'PAUSE_BETWEEN_EXCHANGES': float(os.getenv('PAUSE_BETWEEN_EXCHANGES', '1')),
            
            # Content Settings
            'COVERAGE_STYLE': os.getenv('COVERAGE_STYLE', 'comprehensive').lower(),
            'FACT_CHECK_ENABLED': os.getenv('FACT_CHECK_ENABLED', 'true').lower() == 'true',
            'FACT_CHECK_STYLE': os.getenv('FACT_CHECK_STYLE', 'balanced').lower()
        
        }

    def log_ai_provider_info(self):
        """Log current AI provider configuration"""
        provider = self.config['AI_PROVIDER'].lower()
        model = self.config['MODEL_NAME']
        logging.info(f"Using AI Provider: {provider.upper()}")
        logging.info(f"Model: {model}")
        if provider == 'xai':
            logging.info(f"XAI Base URL: {self.config['XAI_BASE_URL']}")

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

    def _validate_dialogue(self, text: str) -> bool:
        """Validate a single line of dialogue"""
        if not text or len(text) < 20:
            return False
            
        # Check for complete sentences
        sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 10]
        if len(sentences) < 2:
            return False
            
        # Check for conversation markers
        conversation_markers = ['well', 'yes', 'no', 'indeed', 'however', 'but', 'and', 'so']
        has_marker = any(marker in text.lower() for marker in conversation_markers)
        
        return has_marker and all(s[-1] in '.!?' for s in sentences)


    def _log_content(self, prefix: str, content: str, max_length: int = 200):
        """Helper to safely log content snippets"""
        if content:
            preview = content[:max_length] + ('...' if len(content) > max_length else '')
            logging.debug(f"{prefix}: {preview}")
        else:
            logging.debug(f"{prefix}: <empty>")

    def _process_raw_response(self, raw_content: str) -> str:
        """Process raw response with improved ContentBlock handling"""
        try:
            # First try to extract from ContentBlock if present
            if 'ContentBlock' in raw_content:
                matches = re.findall(r'text=["\'](.*?)["\'](?:,|\))', raw_content, re.DOTALL)
                if matches:
                    content = matches[0].replace('\\n', '\n').replace('\\"', '"').replace("\\'", "'")
                    return content
            
            # If not in ContentBlock format or extraction failed, use raw content
            content = raw_content.strip()
            
            # Ensure content has proper speaker formatting
            if not any(line.strip().startswith(('Speaker 1:', 'Speaker 2:')) for line in content.split('\n')):
                logging.warning("Content missing speaker format, attempting to fix...")
                lines = content.split('\n')
                formatted_lines = []
                for i, line in enumerate(lines):
                    if line.strip():
                        speaker = "Speaker 1:" if i % 2 == 0 else "Speaker 2:"
                        if not line.startswith(('Speaker 1:', 'Speaker 2:')):
                            formatted_lines.append(f"{speaker} {line.strip()}")
                        else:
                            formatted_lines.append(line)
                content = '\n'.join(formatted_lines)
            
            return content
            
        except Exception as e:
            logging.error(f"Error processing response: {e}")
            return raw_content

    def _extract_exchanges(self, cleaned_content: str) -> List[Dict]:
        """Extract exchanges with improved validation"""
        exchanges = []
        lines = cleaned_content.split('\n')
        max_exchanges = self.config['MAX_EXCHANGES']
        
        logging.info(f"Processing content for up to {max_exchanges} exchanges")
        
        for line in lines:
            if len(exchanges) >= max_exchanges:
                logging.info(f"Reached max exchanges limit ({max_exchanges})")
                break
                
            line = line.strip()
            if not line:
                continue
                
            # Try multiple speaker patterns
            speaker_match = re.match(r'^(?:Speaker\s*([12]):|HOST\s*([12]):|(SPEAKER\s*[12]:))\s*(.+)', line, re.IGNORECASE)
            if speaker_match:
                # Get the first non-None group for speaker number
                speaker = next(g for g in speaker_match.groups()[:3] if g is not None)
                if speaker.isdigit():
                    speaker_num = speaker
                else:
                    speaker_num = '1' if '1' in speaker else '2'
                
                text = speaker_match.group(4).strip()
                text = self.clean_conversation_text(text)
                
                if text and len(text.split()) >= self.config['EXCHANGE_LENGTH_MIN_WORDS']:
                    exchanges.append({
                        "speaker": speaker_num,
                        "text": text,
                        "timestamp": f"{len(exchanges)//2}:00"
                    })
                    logging.debug(f"Added exchange {len(exchanges)}: Speaker {speaker_num}")
        
        exchange_count = len(exchanges)
        if exchange_count > 0:
            logging.info(f"Successfully generated {exchange_count} exchanges")
        else:
            logging.warning("No valid exchanges were generated")
        
        return exchanges

    async def calculate_target_length(self, source_length: float) -> float:
        """Calculate appropriate podcast length based on source content"""
        
        # Get settings
        target_minutes = self.config['TARGET_LENGTH_MINUTES']
        source_ratio = self.config['SOURCE_LENGTH_RATIO']
        min_length = self.config['MIN_PODCAST_LENGTH']
        max_length = self.config['MAX_PODCAST_LENGTH']
        
        # Calculate base length using ratio
        source_minutes = source_length / 60
        ratio_based_length = source_minutes * source_ratio
        
        # Adjust based on source length
        if source_minutes < 20:
            # For short content, use close to target length
            target_length = target_minutes
        elif source_minutes < 60:
            # For medium content, scale gradually
            target_length = min(target_minutes * 1.5, ratio_based_length)
        else:
            # For long content, use ratio but ensure good coverage
            target_length = min(max_length, max(target_minutes * 2, ratio_based_length))
        
        # Ensure within bounds
        target_length = max(min_length, min(target_length, max_length))
        
        # Use ASCII characters for logging
        logging.info(f"Source: {source_minutes:.1f}m -> Target: {target_length:.1f}m (Ratio: {source_ratio:.2f})")
        return target_length

    async def generate_comprehensive_discussion(self, transcript: Dict) -> Dict:
        """Generate a thorough podcast discussion with correct settings handling"""
        
        try:
            # Calculate target length and exchanges
            source_length = max(s['end'] for s in transcript['segments'])
            target_length = await self.calculate_target_length(source_length)
            
            # Get settings from config instead of directly from env
            coverage_style = os.getenv('COVERAGE_STYLE', 'comprehensive').lower()
            fact_check_enabled = os.getenv('FACT_CHECK_ENABLED', 'true').lower() == 'true'
            fact_check_style = os.getenv('FACT_CHECK_STYLE', 'balanced').lower()
            
            logging.info(f"Source length: {source_length/60:.1f}m | Target length: {target_length:.1f}m")
            logging.info(f"Using {coverage_style} coverage style with fact checking {'enabled' if fact_check_enabled else 'disabled'}")
            if fact_check_enabled:
                logging.info(f"Fact check style: {fact_check_style}")
            
            # Calculate target exchanges
            target_exchanges = min(
                self.config['MAX_EXCHANGES'],
                max(self.config['MIN_EXCHANGES'], 
                    int((target_length * 150) / self.config['EXCHANGE_LENGTH_MAX_WORDS'])
                )
            )
            
            logging.info(f"Planning {target_exchanges} exchanges (max: {self.config['MAX_EXCHANGES']})")
            
            # Adjust content analysis based on coverage style
            content_analysis_prompt = {
                'emotional': f"""
                    Have the speakers discuss the content with emotional depth and expressiveness:
                    1. Use empathetic, passionate, or intense language to convey engagement with the content
                    2. Infuse excitement, concern, or other strong feelings as relevant to the subject matter
                    3. Let the speakers' tone reflect emotions that enhance connection with the audience
                    4. Emphasize points that are likely to stir emotions in listeners
                    5. Aim to make the discussion feel genuine and heartfelt, you are personally moved by the content
                    
                    This is not an analysis of emotional elements in the content but rather an emotionally expressive delivery about it.
                    Content length: {source_length/60:.1f} minutes
                    
                    Content to discuss:
                    {transcript['text']}
                """,
                'debate': f"""
                    Analyze the content from a debate perspective:
                    1. Identify key arguments and counterarguments
                    2. Outline both supporting using Speaker 1 and opposing evidence using Speaker 2
                    3. Present potential points of agreement or compromise
                    4. Highlight any contradictions or controversial points
                    5. Evaluate the strength of each argument and provide a final verdict
                    
                    Focus on creating a balanced yet critical examination.
                    Content length: {source_length/60:.1f} minutes
                    
                    Content to analyze:
                    {transcript['text']}
                """,
                'humor': f"""
                    Analyze this content with a light-hearted, entertaining approach:
                    1. Find amusing angles or funny observations
                    2. Identify potential jokes or witty remarks
                    3. Look for ironic or absurd elements
                    4. Spot opportunities for playful commentary
                    5. Note any unintentionally funny moments
                    
                    Make it entertaining while still being informative.
                    Focus on finding humor without being disrespectful.
                    Content length: {source_length/60:.1f} minutes
                    
                    Content to analyze:
                    {transcript['text']}
                """,
                'summary': f"""
                    Provide a concise summary of the main points:
                    1. Core message or thesis
                    2. 3-4 key supporting points
                    3. Most important evidence
                    4. Main conclusions
                    
                    Focus on the most significant elements.
                    Content length: {source_length/60:.1f} minutes
                    
                    Content to analyze:
                    {transcript['text']}
                """,
                'highlights': f"""
                    Extract the most interesting or notable points:
                    1. Surprising or unique insights
                    2. Strongest arguments
                    3. Memorable examples
                    4. Stand-out moments
                    
                    Focus on what makes this content distinctive.
                    Content length: {source_length/60:.1f} minutes
                    
                    Content to analyze:
                    {transcript['text']}
                """,
                'comprehensive': f"""
                    Provide a detailed analysis of the entire content:
                    1. All major topics and themes
                    2. Key arguments and evidence
                    3. Important details and examples
                    4. Chronological progression
                    5. Main conclusions
                    
                    Cover the entire content thoroughly, not just highlights.
                    Content length: {source_length/60:.1f} minutes
                    
                    Content to analyze:
                    {transcript['text']}
                """
            }.get(coverage_style, 'comprehensive')  # Fallback to comprehensive if invalid style
            
            # Generate content analysis
            content_analysis = await self._generate_content_analysis(transcript, content_analysis_prompt)
            
            # Generate fact checks only if enabled
            fact_checks = ""
            if fact_check_enabled:
                logging.info(f"Generating fact checks with {fact_check_style} style")
                fact_check_prompt = {
                    'balanced': """
                        Analyze the content's accuracy objectively:
                        - Verify key factual claims
                        - Note both accurate and inaccurate statements
                        - Provide corrections where needed
                        - Add relevant context
                        Balance praise and criticism.
                    """,
                    'critical': """
                        Thoroughly examine all claims for accuracy:
                        - Scrutinize every major assertion
                        - Identify potential errors or misunderstandings
                        - Provide detailed corrections
                        - Note missing context
                        Focus on improving accuracy and explain what is false and what is truth.
                    """,
                    'supportive': """
                        Review content accuracy constructively:
                        - Verify main factual claims
                        - Highlight accurate information
                        - Gently note any needed corrections
                        - Add helpful context
                        Maintain a positive tone.
                    """
                }.get(fact_check_style, 'balanced')
                
                fact_checks = await self._generate_fact_checks(transcript['text'], fact_check_prompt)
            else:
                logging.info("Fact checking disabled")
            
            # Calculate target exchanges
            SPEECH_RATE = 150
            words_per_exchange = (self.config['EXCHANGE_LENGTH_MIN_WORDS'] + self.config['EXCHANGE_LENGTH_MAX_WORDS']) // 2
            target_exchanges = max(20, int((target_length * SPEECH_RATE) / words_per_exchange))
            
            logging.info(f"Source length: {source_length/60:.1f}m | Target length: {target_length:.1f}m")
            logging.info(f"Using {coverage_style} coverage style")
            logging.info(f"Planning {target_exchanges} exchanges for target length")
            
            # Generate discussion with style-appropriate prompt
            discussion_prompt = f"""
            Create an informative discussion about this content.
            Use {coverage_style} coverage style with {fact_check_style} approach to accuracy.

            Important Rules:
            - NO podcast names or introductions
            - Start directly with content discussion
            - Provide a summary on the content DO NOT repeat word for word
            - Focus on the actual content
            - Natural dialogue style
            - {coverage_style.capitalize()} coverage of the material
                
                Required Format:
                Speaker 1: [First point about the content]
                Speaker 2: [Response and additional insight]
                (continue alternating speakers)

                Required Structure:
                1. Content Discussion (90% of exchanges)
                - Cover key points chronologically
                - Include insights and analysis
                - Address any corrections or clarifications
                2. Brief Conclusion (10% of exchanges)
                - Summarize main takeaways
                - Final thoughts

                Generate exactly {target_exchanges} total exchanges.
                Each exchange should be {self.config['EXCHANGE_LENGTH_MIN_WORDS']}-{self.config['EXCHANGE_LENGTH_MAX_WORDS']} words.

                Content to Discuss:
                {content_analysis}

                Fact Checks to Include:
                {fact_checks}
                """
            
            response = await self._make_ai_request(discussion_prompt, temperature=0.7)
            cleaned_content = self._process_raw_response(str(response.content))
            exchanges = self._extract_exchanges(cleaned_content)
            
            if len(exchanges) >= target_exchanges * 0.8:  # Allow some flexibility
                return {"conversation": exchanges}
                
            # If first attempt didn't produce enough exchanges, try again with simpler prompt
            logging.warning(f"First attempt only produced {len(exchanges)} exchanges, trying again")
            
            backup_prompt = f"""
            Create a detailed discussion about this content.
            Format exactly like this, with {target_exchanges} total exchanges:

            Speaker 1: [Discussion point about the content]
            Speaker 2: [Response and additional insight]
            Speaker 1: [Follow-up point with new information]
            Speaker 2: [Analysis and connection to other points]

            Important:
            - NO podcast names or introductions
            - Start directly with content discussion
            - Provide a summary on the content DO NOT repeat word for word
            - Be specific and detailed
            - Cover the entire source content
            - Natural conversation style

            Content:
            {content_analysis}
            """
            
            backup_response = await self._make_ai_request(backup_prompt, temperature=0.7)
            backup_content = self._process_raw_response(str(backup_response.content))
            backup_exchanges = self._extract_exchanges(backup_content)
            
            if len(backup_exchanges) >= target_exchanges * 0.8:
                return {"conversation": backup_exchanges}
                
            # Use whichever attempt produced more exchanges
            return {"conversation": exchanges if len(exchanges) > len(backup_exchanges) else backup_exchanges}

        except Exception as e:
            logging.error(f"Error in discussion generation: {str(e)}")
            raise

    async def _generate_content_analysis(self, transcript: Dict, prompt: str) -> str:
        """Generate a focused content analysis"""
        try:
            response = await self._make_ai_request(prompt, temperature=0.3)
            return str(response.content)
        except Exception as e:
            logging.error(f"Error generating content analysis: {str(e)}")
            raise

    async def _generate_fact_checks(self, content: str, fact_check_prompt: str) -> str:
        """Generate fact checks with configurable style"""
        try:
            prompt = f"""
            {fact_check_prompt}
            
            Content to analyze:
            {content}
            """
            
            response = await self._make_ai_request(prompt, temperature=0.3)
            return str(response.content)
            
        except Exception as e:
            logging.error(f"Error generating fact checks: {e}")
            return "Fact checking unavailable."

    async def _make_ai_request(self, prompt: str, temperature: float = 0.7) -> any:
        """Make an AI request with system message and robust error handling"""
        try:
            # Initialize client based on provider
            if self.config['AI_PROVIDER'].lower() == 'xai':
                client = Anthropic(
                    api_key=self.config['XAI_API_KEY'],
                    base_url=self.config['XAI_BASE_URL']
                )
            else:
                client = Anthropic(api_key=self.config['ANTHROPIC_API_KEY'])
            
            # Initial request with system message
            response = client.messages.create(
                model=self.config['MODEL_NAME'],
                max_tokens=self.config['MAX_TOKENS'],
                temperature=temperature,
                system="Generate podcast discussion with alternating Speaker 1 and Speaker 2 lines.",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            content = str(response.content)
            if not any(line.strip().startswith(('Speaker 1:', 'Speaker 2:')) for line in content.split('\n')):
                logging.warning("Response does not contain proper speaker format, retrying...")
                # Retry with more explicit prompt
                response = client.messages.create(
                    model=self.config['MODEL_NAME'],
                    max_tokens=self.config['MAX_TOKENS'],
                    temperature=temperature,
                    system="You must format every line as 'Speaker 1:' or 'Speaker 2:' followed by their dialogue.",
                    messages=[
                        {
                            "role": "user",
                            "content": f"""
                            Format EXACTLY like this:
                            Speaker 1: [First line]
                            Speaker 2: [Response]
                            Speaker 1: [Next line]
                            
                            Content to discuss:
                            {prompt}
                            """
                        }
                    ]
                )
            
            return response
                
        except Exception as e:
            provider = self.config['AI_PROVIDER'].lower()
            logging.error(f"AI request failed for {provider}: {str(e)}")
            raise

    async def _make_simple_ai_request(self, prompt: str, temperature: float = 0.7) -> any:
        """Make a basic AI request without system message"""
        try:
            # Initialize client based on provider
            if self.config['AI_PROVIDER'].lower() == 'xai':
                client = Anthropic(
                    api_key=self.config['XAI_API_KEY'],
                    base_url=self.config['XAI_BASE_URL']
                )
            else:
                client = Anthropic(api_key=self.config['ANTHROPIC_API_KEY'])
                
            response = client.messages.create(
                model=self.config['MODEL_NAME'],
                max_tokens=self.config['MAX_TOKENS'],
                temperature=temperature,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            return response
                
        except Exception as e:
            provider = self.config['AI_PROVIDER'].lower()
            logging.error(f"Simple AI request failed for {provider}: {str(e)}")
            raise

    def clean_conversation_text(self, text: str) -> str:
        """Clean and validate conversation text with better handling"""
        if not text:
            return ""
        
        try:
            # Remove common formatting artifacts
            text = text.replace('\\"', '"').replace('\\n', ' ')
            text = re.sub(r'[*_~`]', '', text)
            
            # Split into sentences
            sentences = []
            for sentence in text.split('.'):
                cleaned = sentence.strip()
                if len(cleaned) >= 10:  # Minimum sentence length
                    if not cleaned[-1] in '.!?':
                        cleaned += '.'
                    sentences.append(cleaned)
            
            if not sentences:
                return ""
            
            # Join sentences with proper spacing
            text = ' '.join(sentences)
            
            # Additional validation
            words = text.split()
            if len(words) < self.config['EXCHANGE_LENGTH_MIN_WORDS']:
                logging.debug(f"Text too short ({len(words)} words): {text[:50]}...")
                return ""
                
            if len(words) > self.config['EXCHANGE_LENGTH_MAX_WORDS']:
                logging.debug(f"Trimming long text from {len(words)} words")
                text = ' '.join(words[:self.config['EXCHANGE_LENGTH_MAX_WORDS']])
                if not text[-1] in '.!?':
                    text += '.'
            
            return text
            
        except Exception as e:
            logging.error(f"Error cleaning text: {e}")
            return ""

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
                        timeout=60,
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
    parser.add_argument('--no-merge', action='store_true', help='Skip audio merging')
    args = parser.parse_args()
    
    analyzer = PodcastAnalyzer()
    os.makedirs(analyzer.config['OUTPUT_DIR'], exist_ok=True)
    
    # Log AI provider info
    analyzer.log_ai_provider_info()
    
    try:
        # Download and transcribe
        logging.info("Starting podcast analysis...")
        audio_path = await analyzer.download_podcast(args.url)
        logging.info(f"Downloaded audio to: {audio_path}")
        
        # Get full transcript
        transcript = await analyzer.transcribe_audio(audio_path)
        logging.info("Transcription complete")
        
        # Generate comprehensive discussion
        logging.info("Generating podcast discussion...")
        try:
            conversation = await analyzer.generate_comprehensive_discussion(transcript)
        except Exception as e:
            logging.error(f"Error in conversation generation: {str(e)}")
            raise

        # Validate conversation
        if not conversation or not conversation.get('conversation'):
            logging.error("Failed to generate valid conversation")
            return
        
        if len(conversation['conversation']) < 2:
            logging.error("Generated conversation too short")
            return
        
        # Generate audio files
        audio_files = []
        if not args.no_audio:
            logging.info("Generating audio files...")
            audio_files = await analyzer.generate_audio(conversation)
            
            if audio_files:
                logging.info(f"Successfully generated {len(audio_files)} audio files:")
                analyzer.display_audio_sequence(audio_files)
                
                # Merge audio files if generation was successful
                if not args.no_merge and len(audio_files) > 0:
                    logging.info("Merging audio files...")
                    try:
                        from merge_audio import merge_audio_files
                        output_filename = f"merged_podcast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3"
                        merge_audio_files(analyzer.config['OUTPUT_DIR'], output_filename)
                        logging.info(f"Successfully merged audio files into: {output_filename}")
                    except Exception as e:
                        logging.error(f"Error merging audio files: {str(e)}")
            else:
                logging.error("No audio files were generated")
        
        logging.info("Analysis complete!")
        
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())