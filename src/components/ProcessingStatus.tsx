import { Loader2 } from 'lucide-react';

export default function ProcessingStatus({ status }: { status: string }) {
  const getStatusMessage = (status: string) => {
    switch (status.toLowerCase()) {
      case 'downloading':
        return 'Downloading audio from YouTube...';
      case 'processing audio':
        return 'Processing audio segments...';
      case 'merging segments':
        return 'Merging audio segments...';
      default:
        return status;
    }
  };

  return (
    <div className="flex flex-col items-center justify-center space-y-4 p-8 bg-gray-800/50 rounded-lg">
      <Loader2 className="w-8 h-8 text-purple-500 animate-spin" />
      <p className="text-gray-300">{getStatusMessage(status)}</p>
    </div>
  );
}