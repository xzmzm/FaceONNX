import React, { useRef, useEffect } from 'react';

interface EmbeddingHeatmapProps {
  embedding: number[];
  width?: number;
  height?: number;
}

// Helper function for linear interpolation (can be moved to utils if used elsewhere)
const lerp = (a: number, b: number, t: number): number => a + (b - a) * t;

// Helper to convert RGB components to hex string (can be moved to utils)
const rgbToHex = (r: number, g: number, b: number): string => {
  const toHex = (c: number) => {
    const hex = Math.round(Math.max(0, Math.min(255, c))).toString(16); // Clamp to 0-255
    return hex.length === 1 ? "0" + hex : hex;
  };
  return `#${toHex(r)}${toHex(g)}${toHex(b)}`;
};

// Color mapping function for embedding values (Blue to Red scale)
// Duplicated here for simplicity, could be passed as prop or imported from utils
const getColorForEmbeddingValue = (value: number): string => {
  // Define the approximate range of embedding values. Adjust if needed based on actual data.
  const minVal = -1;
  const maxVal = 1;

  // Define colors: Blue for minVal, Red for maxVal
  const colorMin = { r: 0, g: 0, b: 0 };   // Blue
  const colorMax = { r: 255, g: 255, b: 255 };   // Red

  // Clamp value to the defined range
  const clampedValue = Math.max(minVal, Math.min(value, maxVal));

  // Calculate interpolation factor (t) from 0 to 1
  const range = maxVal - minVal;
  // Handle potential division by zero if range is 0
  const t = range === 0 ? 0.5 : (clampedValue - minVal) / range;

  // Interpolate RGB components
  const r = lerp(colorMin.r, colorMax.r, t);
  const g = lerp(colorMin.g, colorMax.g, t);
  const b = lerp(colorMin.b, colorMax.b, t);

  return rgbToHex(r, g, b);
};


const EmbeddingHeatmap: React.FC<EmbeddingHeatmapProps> = ({
  embedding,
  width = 512, // Default width
  height = 50   // Default height
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !embedding || embedding.length === 0) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Ensure canvas size matches embedding length if different from default
    const actualWidth = embedding.length;
    canvas.width = actualWidth; // Use actual embedding length for width
    canvas.height = height;

    // Clear canvas
    ctx.clearRect(0, 0, actualWidth, height);

    // Draw heatmap lines
    for (let i = 0; i < actualWidth; i++) {
      const value = embedding[i];
      ctx.fillStyle = getColorForEmbeddingValue(value);
      // Draw a 1-pixel wide vertical line
      ctx.fillRect(i, 0, 1, height);
    }

  }, [embedding, width, height]); // Redraw if embedding or dimensions change

  // Render the canvas element, setting display dimensions via style
  // The actual drawing resolution is set via canvas.width/height in useEffect
  return (
    <canvas
      ref={canvasRef}
      style={{ width: `${width}px`, height: `${height}px`, display: 'block', maxWidth: '100%' }}
      aria-label="Embedding heatmap visualization"
    />
  );
};

export default EmbeddingHeatmap;