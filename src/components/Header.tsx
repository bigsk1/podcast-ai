import { Headphones } from 'lucide-react';

export default function Header() {
  return (
    <header className="w-full py-6 px-4 border-b border-gray-800">
      <div className="max-w-4xl mx-auto flex items-center gap-3">
        <Headphones className="w-8 h-8 text-purple-500" />
        <h1 className="text-2xl font-bold bg-gradient-to-r from-purple-500 to-pink-500 bg-clip-text text-transparent">
          AI Podcast Review
        </h1>
      </div>
    </header>
  );
}