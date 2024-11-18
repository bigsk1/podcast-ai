import { useState, useRef, useEffect } from 'react';
import { Play, Pause, Volume2, VolumeX, RotateCcw, SkipBack, SkipForward, Download } from 'lucide-react';
import WaveSurfer from 'wavesurfer.js';

interface AudioPlayerProps {
  audioUrl: string;
}

export default function AudioPlayer({ audioUrl }: AudioPlayerProps) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [volume, setVolume] = useState(1);
  const [waveform, setWaveform] = useState<WaveSurfer | null>(null);
  const audioRef = useRef<HTMLAudioElement>(null);
  const waveformRef = useRef<HTMLDivElement>(null);
  const sliderRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (waveformRef.current) {
      const wavesurfer = WaveSurfer.create({
        container: waveformRef.current,
        waveColor: '#8B5CF6',
        progressColor: '#EC4899',
        cursorColor: '#EC4899',
        barWidth: 2,
        barGap: 1,
        height: 60,
        responsive: true,
        normalize: true,
        backend: 'WebAudio',
      });

      wavesurfer.load(audioUrl);
      
      wavesurfer.on('ready', () => {
        setWaveform(wavesurfer);
        setDuration(wavesurfer.getDuration());
      });

      wavesurfer.on('audioprocess', () => {
        setCurrentTime(wavesurfer.getCurrentTime());
      });

      wavesurfer.on('finish', () => {
        setIsPlaying(false);
      });

      return () => {
        wavesurfer.destroy();
      };
    }
  }, [audioUrl]);

  const formatTime = (time: number) => {
    const minutes = Math.floor(time / 60);
    const seconds = Math.floor(time % 60);
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  };

  const handleDownload = async () => {
    try {
      const response = await fetch(audioUrl);
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `podcast_${new Date().getTime()}.mp3`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (error) {
      console.error('Download failed:', error);
    }
  };

  const togglePlay = () => {
    if (waveform) {
      waveform.playPause();
      setIsPlaying(!isPlaying);
    }
  };

  const toggleMute = () => {
    if (waveform) {
      if (isMuted) {
        waveform.setVolume(volume);
      } else {
        waveform.setVolume(0);
      }
      setIsMuted(!isMuted);
    }
  };

  const handleVolumeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = parseFloat(e.target.value);
    setVolume(value);
    if (waveform) {
      waveform.setVolume(value);
      setIsMuted(value === 0);
    }
  };

  const skipSeconds = (seconds: number) => {
    if (waveform) {
      const newTime = Math.min(
        Math.max(waveform.getCurrentTime() + seconds, 0),
        waveform.getDuration()
      );
      waveform.seekTo(newTime / waveform.getDuration());
    }
  };

  const restart = () => {
    if (waveform) {
      waveform.seekTo(0);
      if (!isPlaying) {
        waveform.play();
        setIsPlaying(true);
      }
    }
  };

  return (
    <div className="w-full max-w-3xl mx-auto bg-gray-800 p-6 rounded-lg shadow-lg space-y-4">
      {/* Waveform */}
      <div ref={waveformRef} className="w-full h-16 bg-gray-900 rounded" />

      {/* Time display */}
      <div className="flex justify-between text-sm text-gray-400">
        <span>{formatTime(currentTime)}</span>
        <span>{formatTime(duration)}</span>
      </div>

      {/* Controls */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4">
          {/* Main controls */}
          <button
            onClick={restart}
            className="p-2 hover:bg-gray-700 rounded-full transition-colors"
            title="Restart"
          >
            <RotateCcw className="w-5 h-5 text-gray-400" />
          </button>
          <button
            onClick={() => skipSeconds(-10)}
            className="p-2 hover:bg-gray-700 rounded-full transition-colors"
            title="Back 10 seconds"
          >
            <SkipBack className="w-5 h-5 text-gray-400" />
          </button>
          <button
            onClick={togglePlay}
            className="p-3 bg-purple-500 hover:bg-purple-600 rounded-full transition-colors"
            title={isPlaying ? 'Pause' : 'Play'}
          >
            {isPlaying ? (
              <Pause className="w-6 h-6 text-white" />
            ) : (
              <Play className="w-6 h-6 text-white" />
            )}
          </button>
          <button
            onClick={() => skipSeconds(10)}
            className="p-2 hover:bg-gray-700 rounded-full transition-colors"
            title="Forward 10 seconds"
          >
            <SkipForward className="w-5 h-5 text-gray-400" />
          </button>
          <button
            onClick={handleDownload}
            className="p-2 hover:bg-gray-700 rounded-full transition-colors"
            title="Download MP3"
          >
            <Download className="w-5 h-5 text-gray-400" />
          </button>
        </div>

        {/* Volume control */}
        <div className="flex items-center space-x-2">
          <button
            onClick={toggleMute}
            className="p-2 hover:bg-gray-700 rounded-full transition-colors"
          >
            {isMuted || volume === 0 ? (
              <VolumeX className="w-5 h-5 text-gray-400" />
            ) : (
              <Volume2 className="w-5 h-5 text-gray-400" />
            )}
          </button>
          <input
            type="range"
            min={0}
            max={1}
            step={0.1}
            value={volume}
            onChange={handleVolumeChange}
            className="w-24 h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-purple-500"
          />
        </div>
      </div>
    </div>
  );
}