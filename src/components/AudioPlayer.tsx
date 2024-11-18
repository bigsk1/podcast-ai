import { useState, useRef } from 'react';
import { Play, Pause, Volume2, VolumeX } from 'lucide-react';

interface AudioPlayerProps {
  audioUrl: string;
}

export default function AudioPlayer({ audioUrl }: AudioPlayerProps) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const audioRef = useRef<HTMLAudioElement>(null);

  const togglePlay = () => {
    if (audioRef.current) {
      if (isPlaying) {
        audioRef.current.pause();
      } else {
        audioRef.current.play();
      }
      setIsPlaying(!isPlaying);
    }
  };

  const toggleMute = () => {
    if (audioRef.current) {
      audioRef.current.muted = !isMuted;
      setIsMuted(!isMuted);
    }
  };

  return (
    <div className="w-full max-w-md mx-auto bg-gray-800 p-4 rounded-lg shadow-lg">
      <audio
        ref={audioRef}
        src={audioUrl}
        className="hidden"
        onEnded={() => setIsPlaying(false)}
      />
      <div className="flex items-center justify-center space-x-4">
        <button
          onClick={togglePlay}
          className="p-2 rounded-full hover:bg-gray-700 transition-colors"
        >
          {isPlaying ? (
            <Pause className="w-8 h-8 text-purple-500" />
          ) : (
            <Play className="w-8 h-8 text-purple-500" />
          )}
        </button>
        <button
          onClick={toggleMute}
          className="p-2 rounded-full hover:bg-gray-700 transition-colors"
        >
          {isMuted ? (
            <VolumeX className="w-6 h-6 text-gray-400" />
          ) : (
            <Volume2 className="w-6 h-6 text-gray-400" />
          )}
        </button>
      </div>
    </div>
  );
}