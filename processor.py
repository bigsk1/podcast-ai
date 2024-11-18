import os
import asyncio
import logging
import tempfile
import shutil
from typing import Dict
from pathlib import Path
from podcast_analyzer import PodcastAnalyzer, check_ffmpeg
from merge_audio import merge_audio_files

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def process_youtube_url(job_id: str, url: str, jobs: Dict[str, dict]) -> None:
    """Process a YouTube URL and create an AI podcast"""
    
    output_dir = os.path.join("public", "audio")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create output directory for intermediate files
    temp_output_dir = os.path.join("output")
    os.makedirs(temp_output_dir, exist_ok=True)
    
    analyzer = None

    try:
        # Update job status
        jobs[job_id].update({
            "status": "checking requirements",
            "audio_url": None,
            "error": None
        })

        if not check_ffmpeg():
            raise Exception("FFmpeg is required but not found")

        jobs[job_id]["status"] = "initializing"
        
        # Initialize the podcast analyzer
        analyzer = PodcastAnalyzer()
        analyzer.config['OUTPUT_DIR'] = temp_output_dir
        
        jobs[job_id]["status"] = "downloading"
        # Download the audio
        audio_path = await analyzer.download_podcast(url)
        
        jobs[job_id]["status"] = "transcribing"
        # Get transcript
        transcript = await analyzer.transcribe_audio(audio_path)
        
        jobs[job_id]["status"] = "generating discussion"
        # Generate discussion
        conversation = await analyzer.generate_comprehensive_discussion(transcript)
        
        if not conversation or not conversation.get('conversation'):
            raise Exception("Failed to generate valid conversation")
            
        jobs[job_id]["status"] = "generating audio"
        # Generate audio files
        audio_files = await analyzer.generate_audio(conversation)
        
        if not audio_files:
            raise Exception("No audio files were generated")
            
        jobs[job_id]["status"] = "merging audio"
        # Merge audio files
        merged_filename = f"{job_id}.mp3"
        merge_audio_files(temp_output_dir, merged_filename)
        
        # Move the merged file to public directory
        source_path = os.path.join(temp_output_dir, merged_filename)
        dest_path = os.path.join(output_dir, merged_filename)
        
        if os.path.exists(source_path):
            shutil.move(source_path, dest_path)
            jobs[job_id].update({
                "status": "completed",
                "audio_url": f"/audio/{merged_filename}"
            })
        else:
            raise Exception(f"Merged audio file not found at {source_path}")
            
    except Exception as e:
        logging.error(f"Error processing job {job_id}: {str(e)}")
        jobs[job_id].update({
            "status": "failed",
            "error": str(e)
        })
        
    finally:
        # Cleanup intermediate files but keep the merged output
        try:
            for file in os.listdir(temp_output_dir):
                if file.startswith('segment_') and file.endswith('.mp3'):
                    os.remove(os.path.join(temp_output_dir, file))
        except Exception as e:
            logging.error(f"Cleanup error: {str(e)}")