import { useState, useCallback, useEffect } from 'react';
import { toast, Toaster } from 'react-hot-toast';
import { LinkIcon, RotateCcw, Headphones, Play } from 'lucide-react';
import Header from './components/Header';
import PodcastForm from './components/PodcastForm';
import ProcessingStatus from './components/ProcessingStatus';
import AudioPlayer from './components/AudioPlayer';
import { podcastService } from './services/api';

interface ProcessedPodcast {
  url: string;
  audioUrl: string;
  timestamp: string;
  videoId: string;
}

function App() {
  const [isProcessing, setIsProcessing] = useState(false);
  const [status, setStatus] = useState('');
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [youtubeUrl, setYoutubeUrl] = useState('');
  const [pollCount, setPollCount] = useState(0);
  const [processedPodcasts, setProcessedPodcasts] = useState<ProcessedPodcast[]>([]);

  const pollStatus = useCallback(async (id: string) => {
    try {
      const response = await podcastService.checkStatus(id);
      
      if (response.status === 'completed' && response.audioUrl) {
        setIsProcessing(false);
        setStatus('');
        setAudioUrl(response.audioUrl);
        setPollCount(0);
        
        // Save to processed podcasts
        const videoId = getYoutubeVideoId(youtubeUrl);
        if (videoId) {
          const newPodcast: ProcessedPodcast = {
            url: youtubeUrl,
            audioUrl: response.audioUrl,
            timestamp: new Date().toISOString(),
            videoId
          };
          setProcessedPodcasts(prev => {
            const newProcessed = [newPodcast, ...prev.filter(p => p.videoId !== videoId)];
            localStorage.setItem('processedPodcasts', JSON.stringify(newProcessed));
            return newProcessed;
          });
        }
        toast.success('Podcast generated successfully!');
      } else if (response.status === 'failed') {
        throw new Error(response.error || 'Generation failed');
      } else {
        setStatus(response.status);
        const delay = Math.min(1000 * Math.pow(1.5, pollCount), 10000);
        setPollCount(prev => prev + 1);
        setTimeout(() => pollStatus(id), delay);
      }
    } catch (error) {
      handleError(error);
    }
  }, [pollCount, youtubeUrl]);

  const handleError = (error: unknown) => {
    const message = error instanceof Error ? error.message : 'An unexpected error occurred';
    setStatus('Error: ' + message);
    toast.error(message);
    setTimeout(() => {
      setIsProcessing(false);
      setStatus('');
    }, 3000);
  };

  const clearAll = () => {
    setYoutubeUrl('');
    setAudioUrl(null);
    setStatus('');
    setJobId(null);
    setIsProcessing(false);
  };

  const playExistingPodcast = (podcast: ProcessedPodcast) => {
    setYoutubeUrl(podcast.url);
    setAudioUrl(podcast.audioUrl);
    setIsProcessing(false);
    setStatus('');
  };

  const handleSubmit = async (url: string) => {
    // Check if we already have this podcast processed
    const videoId = getYoutubeVideoId(url);
    const existingPodcast = processedPodcasts.find(p => p.videoId === videoId);
    
    if (existingPodcast) {
      playExistingPodcast(existingPodcast);
      return;
    }

    // Process new podcast
    setIsProcessing(true);
    setStatus('Initializing...');
    setAudioUrl(null);
    setYoutubeUrl(url);
    setPollCount(0);

    try {
      const response = await podcastService.generatePodcast(url);
      setJobId(response.status);
      await pollStatus(response.status);
    } catch (error) {
      handleError(error);
    }
  };

  const getYoutubeVideoId = (url: string) => {
    const match = url.match(/[?&]v=([^&]+)/);
    return match ? match[1] : null;
  };

  // Load processed podcasts on mount
  useEffect(() => {
    const savedPodcasts = localStorage.getItem('processedPodcasts');
    if (savedPodcasts) {
      setProcessedPodcasts(JSON.parse(savedPodcasts));
    }
  }, []);

  return (
    <div className="min-h-screen bg-gray-900 text-gray-100">
      <Toaster position="top-right" />
      <Header />
      
      <main className="max-w-4xl mx-auto px-4 py-8">
        <div className="space-y-8">
          <div className="text-center space-y-4">
            <h2 className="text-3xl font-bold">Transform YouTube Videos into AI Podcasts</h2>
            <p className="text-gray-400 max-w-2xl mx-auto">
              Simply paste a YouTube URL and let our AI create an engaging podcast review. 
              Perfect for content creators, researchers, and podcast enthusiasts.
            </p>
          </div>

          <PodcastForm 
            onSubmit={handleSubmit} 
            disabled={isProcessing}
            value={youtubeUrl}
            onChange={setYoutubeUrl}
          />

          {/* Processing Status */}
          {isProcessing && (
            <ProcessingStatus status={status} />
          )}

          {/* YouTube Video */}
          {youtubeUrl && getYoutubeVideoId(youtubeUrl) && (
            <div className="rounded-lg overflow-hidden bg-gray-800 aspect-video w-full max-w-3xl mx-auto">
              <iframe
                width="100%"
                height="100%"
                src={`https://www.youtube.com/embed/${getYoutubeVideoId(youtubeUrl)}`}
                title="YouTube video player"
                frameBorder="0"
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                allowFullScreen
              ></iframe>
            </div>
          )}

          {/* Audio Player */}
          {audioUrl && !isProcessing && (
            <div className="space-y-4 max-w-xl mx-auto">
              <AudioPlayer audioUrl={audioUrl} />
              <button
                onClick={clearAll}
                className="w-full py-2 px-4 bg-gray-800 hover:bg-gray-700 rounded-lg transition-colors flex items-center justify-center gap-2"
              >
                <RotateCcw size={16} />
                Clear & Start Over
              </button>
            </div>
          )}

          {/* Previously Processed Videos */}
          {processedPodcasts.length > 0 && !isProcessing && (
            <div className="mt-8 bg-gray-800 rounded-lg p-4">
              <h3 className="text-lg font-semibold mb-4">Previously Processed Videos</h3>
              <div className="space-y-2">
                {processedPodcasts.map((podcast, index) => (
                  <div key={index} className="flex items-center gap-2">
                    <button
                      onClick={() => playExistingPodcast(podcast)}
                      className="flex items-center gap-2 text-purple-400 hover:text-purple-300"
                      title="Play saved podcast"
                    >
                      <Play size={14} />
                      <span className="truncate">{podcast.url}</span>
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

export default App;