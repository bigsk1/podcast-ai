/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        gray: {
          900: '#111827',
          800: '#1F2937',
          700: '#374151',
          400: '#9CA3AF',
          300: '#D1D5DB',
          100: '#F3F4F6',
        },
        purple: {
          500: '#8B5CF6',
        },
        pink: {
          500: '#EC4899',
        },
      },
    },
  },
  plugins: [],
};