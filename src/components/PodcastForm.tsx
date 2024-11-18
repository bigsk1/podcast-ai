import { useState } from 'react';
import { Youtube } from 'lucide-react';

interface PodcastFormProps {
  onSubmit: (url: string) => void;
  disabled?: boolean;
  value: string;
  onChange: (value: string) => void;
}

export default function PodcastForm({ onSubmit, disabled, value, onChange }: PodcastFormProps) {
  const [error, setError] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!value.trim()) {
      setError('Please enter a YouTube URL');
      return;
    }

    if (!value.includes('youtube.com/watch?v=') && !value.includes('youtu.be/')) {
      setError('Please enter a valid YouTube URL');
      return;
    }

    setError('');
    onSubmit(value);
  };

  return (
    <form onSubmit={handleSubmit} className="w-full max-w-2xl mx-auto space-y-4">
      <div className="relative">
        <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
          <Youtube className="h-5 w-5 text-gray-400" />
        </div>
        <input
          type="url"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder="Paste YouTube URL here..."
          className="w-full pl-10 pr-4 py-3 bg-gray-800 border border-gray-700 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent outline-none transition-all"
          disabled={disabled}
        />
      </div>
      {error && (
        <p className="text-red-400 text-sm">{error}</p>
      )}
      <button
        type="submit"
        disabled={disabled}
        className="w-full py-3 px-4 bg-gradient-to-r from-purple-500 to-pink-500 text-white font-medium rounded-lg hover:opacity-90 transition-opacity disabled:opacity-50"
      >
        Generate AI Podcast Review
      </button>
    </form>
  );
}