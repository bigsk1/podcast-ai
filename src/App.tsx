import { useState, useCallback, useEffect } from 'react';
import { toast, Toaster } from 'react-hot-toast';
import Header from './components/Header';
import PodcastForm from './components/PodcastForm';
import ProcessingStatus from './components/ProcessingStatus';
import AudioPlayer from './components/AudioPlayer';
import { podcastService } from './services/api';

function App() {
  const [isProcessing, setIsProcessing] = useState(false);
  const [status, setStatus] = useState('');
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [pollCount, setPollCount] = useState(0);

  const pollStatus = useCallback(async (id: string) => {
    try {
      const response = await podcastService.checkStatus(id);
      
      if (response.status === 'completed' && response.audioUrl) {
        setIsProcessing(false);
        setStatus('');
        setAudioUrl(response.audioUrl);
        setPollCount(0);
        toast.success('Podcast generated successfully!');
      } else if (response.status === 'failed') {
        throw new Error(response.error || 'Generation failed');
      } else {
        setStatus(response.status);
        // Continue polling with exponential backoff
        const delay = Math.min(1000 * Math.pow(1.5, pollCount), 10000); // Max 10 seconds
        setPollCount(prev => prev + 1);
        setTimeout(() => pollStatus(id), delay);
      }
    } catch (error) {
      if (pollCount < 20) { // Try for about 5 minutes before giving up
        // If error is connection-related, keep trying
        const delay = Math.min(1000 * Math.pow(1.5, pollCount), 10000);
        setPollCount(prev => prev + 1);
        setTimeout(() => pollStatus(id), delay);
      } else {
        handleError(error);
      }
    }
  }, [pollCount]);

  const handleError = (error: unknown) => {
    const message = error instanceof Error ? error.message : 'An unexpected error occurred';
    setStatus('Error: ' + message);
    toast.error(message);
    setTimeout(() => {
      setIsProcessing(false);
      setStatus('');
    }, 3000);
  };

  const handleSubmit = async (url: string) => {
    setIsProcessing(true);
    setStatus('Initializing...');
    setAudioUrl(null);
    setPollCount(0);

    try {
      const response = await podcastService.generatePodcast(url);
      setJobId(response.status);
      await pollStatus(response.status);
    } catch (error) {
      handleError(error);
    }
  };

  // Resume polling on page load if there was an active job
  useEffect(() => {
    const savedJobId = localStorage.getItem('activeJobId');
    if (savedJobId) {
      setJobId(savedJobId);
      setIsProcessing(true);
      pollStatus(savedJobId);
    }
  }, []);

  // Save active job ID to localStorage
  useEffect(() => {
    if (jobId && isProcessing) {
      localStorage.setItem('activeJobId', jobId);
    } else {
      localStorage.removeItem('activeJobId');
    }
  }, [jobId, isProcessing]);

  return (
    <div className="min-h-screen bg-gray-900 text-gray-100">
      <Toaster position="top-right" />
      <Header />
      
      <main className="max-w-4xl mx-auto px-4 py-12 space-y-8">
        <div className="text-center space-y-4">
          <h2 className="text-3xl font-bold">Transform YouTube Videos into AI Podcasts</h2>
          <p className="text-gray-400 max-w-2xl mx-auto">
            Simply paste a YouTube URL and let our AI create an engaging podcast review. 
            Perfect for content creators, researchers, and podcast enthusiasts.
          </p>
        </div>

        <PodcastForm onSubmit={handleSubmit} disabled={isProcessing} />

        {isProcessing && (
          <ProcessingStatus status={status} />
        )}

        {audioUrl && !isProcessing && (
          <AudioPlayer audioUrl={audioUrl} />
        )}
      </main>
    </div>
  );
}

export default App;