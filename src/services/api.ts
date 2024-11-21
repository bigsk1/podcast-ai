import axios from 'axios';

// Dynamically determine API URL based on current hostname
const API_BASE_URL = import.meta.env.VITE_API_URL || 
  `http://${window.location.hostname}:5000`;

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export interface PodcastResponse {
  status: string;
  audioUrl?: string;
  error?: string;
}

export const podcastService = {
  async generatePodcast(youtubeUrl: string): Promise<PodcastResponse> {
    try {
      const response = await api.post('/generate-podcast', { url: youtubeUrl });
      return response.data;
    } catch (error) {
      if (axios.isAxiosError(error)) {
        throw new Error(error.response?.data?.error || 'Failed to generate podcast');
      }
      throw error;
    }
  },

  async checkStatus(jobId: string): Promise<PodcastResponse> {
    try {
      const response = await api.get(`/status/${jobId}`);
      return response.data;
    } catch (error) {
      if (axios.isAxiosError(error)) {
        throw new Error(error.response?.data?.error || 'Failed to check status');
      }
      throw error;
    }
  },
};