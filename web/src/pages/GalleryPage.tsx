import React, { useState, useEffect } from 'react';
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/card';
import { toast } from 'sonner';

// Define the expected structure of the data from the backend
interface GalleryData {
  [label: string]: number[][]; // Label maps to a list of embeddings (each embedding is a list of numbers)
}

const GalleryPage: React.FC = () => {
  const [galleryData, setGalleryData] = useState<GalleryData | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      setIsLoading(true);
      setError(null);
      try {
        // TODO: Replace with actual backend URL
        const backendUrl = 'http://localhost:8000/gallery_data';
        const response = await fetch(backendUrl);

        if (!response.ok) {
          const errorData = await response.json().catch(() => ({ detail: `HTTP error! status: ${response.status}` }));
          throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
        }

        const data: GalleryData = await response.json();
        setGalleryData(data);
      } catch (err) {
        console.error('Failed to fetch gallery data:', err);
        const errorMessage = err instanceof Error ? err.message : String(err);
        setError(errorMessage);
        toast.error(`Failed to load gallery: ${errorMessage}`);
      } finally {
        setIsLoading(false);
      }
    };

    fetchData();
  }, []); // Empty dependency array means this runs once on mount

  return (
    <Card className="w-full max-w-2xl"> {/* Wider card for gallery */}
      <CardHeader>
        <CardTitle>Registered Faces Gallery</CardTitle>
        <CardDescription>View all registered faces and the number of embeddings for each.</CardDescription>
      </CardHeader>
      <CardContent>
        {isLoading && <p>Loading gallery...</p>}
        {error && <p className="text-destructive">Error loading gallery: {error}</p>}
        {!isLoading && !error && galleryData && Object.keys(galleryData).length > 0 && (
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
            {Object.entries(galleryData).map(([label, embeddings]) => (
              <Card key={label} className="flex flex-col">
                <CardHeader className="pb-2">
                  <CardTitle className="text-lg">{label}</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-muted-foreground">
                    Embeddings: {embeddings?.length || 0}
                  </p>
                  {/* Optional: Could add a small representative image if stored/retrievable */}
                </CardContent>
                {/* Optional: Add delete button per label */}
              </Card>
            ))}
          </div>
        )}
        {!isLoading && !error && (!galleryData || Object.keys(galleryData).length === 0) && (
          <p>No faces registered yet.</p>
        )}
      </CardContent>
    </Card>
  );
};

export default GalleryPage;