from pydub import AudioSegment
import os
import sys
import glob
import re
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def merge_audio_files(directory: str, output_file: str = "merged_conversation.mp3"):
    """Merge all audio segments in order"""
    try:
        # Get all mp3 files
        audio_files = glob.glob(os.path.join(directory, "segment_*_speaker*.mp3"))
        
        if not audio_files:
            print("No audio files found in directory!")
            return
            
        # Sort files by segment number
        audio_files.sort(key=lambda x: int(re.search(r'segment_(\d+)_', os.path.basename(x)).group(1)))
        
        print(f"Found {len(audio_files)} audio files to merge")
        print("\nProcessing sequence:")
        
        # Create merged audio
        merged = AudioSegment.empty()
        
        for file in audio_files:
            filename = os.path.basename(file)
            segment_num = re.search(r'segment_(\d+)_', filename).group(1)
            speaker_num = re.search(r'speaker(\d)', filename).group(1)
            
            print(f"Adding segment {segment_num} (Speaker {speaker_num}): {filename}")
            
            # Load audio segment
            audio = AudioSegment.from_mp3(file)
            
            # Add a small pause between segments (500ms = 0.5 seconds)
            if len(merged) > 0:
                merged = merged + AudioSegment.silent(duration=500)
            
            # Add the audio segment
            merged = merged + audio
        
        # Add a longer pause at the end
        merged = merged + AudioSegment.silent(duration=1000)
        
        # Export the merged file
        output_path = os.path.join(directory, output_file)
        merged.export(output_path, format="mp3")
        
        print(f"\nSuccessfully created merged audio file:")
        print(f"- Output: {output_file}")
        print(f"- Total segments: {len(audio_files)}")
        print(f"- Total length: {len(merged)/1000:.1f} seconds")
        
    except Exception as e:
        print(f"Error merging audio files: {str(e)}")
        logging.error(f"Error details: {str(e)}", exc_info=True)

def main():
    """Main function to handle command line arguments"""
    if len(sys.argv) < 2:
        print("Usage: python merge_audio.py <directory> [output_filename]")
        print("Example: python merge_audio.py output conversation.mp3")
        return
        
    directory = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "merged_conversation.mp3"
    
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return
        
    print(f"\nMerging audio files from: {directory}")
    print(f"Output file will be: {output_file}")
    merge_audio_files(directory, output_file)

if __name__ == "__main__":
    main()