import React, { useState, useEffect } from 'react';
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/card';
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { toast } from 'sonner';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts'; // Added recharts components

// Define the expected structure for a single registered entry
interface GalleryEntry {
  embedding: number[]; // List of 512 floats
  image_filename: string; // Just the filename like "unique_image.jpg"
}

// Define the expected structure of the data from the backend
interface GalleryData {
  [label: string]: GalleryEntry[]; // Label maps to a list of entries
}

// Helper function to format embedding data for recharts
const formatEmbeddingDataForChart = (embedding: number[]) => {
  if (!embedding) return [];
  return embedding.map((value, index) => ({
    index: index, // Index of the value (0-511)
    value: value   // The actual float value
  }));
};

const GalleryPage: React.FC = () => {
  const [galleryData, setGalleryData] = useState<GalleryData | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      setIsLoading(true);
      setError(null);
      try {
        // Use environment variable for backend URL
        const backendBaseUrl = import.meta.env.VITE_BACKEND_URL;
        if (!backendBaseUrl) {
            throw new Error("Backend URL is not configured in environment variables (VITE_BACKEND_URL).");
        }
        const galleryUrl = `${backendBaseUrl}/gallery_data`;
        const response = await fetch(galleryUrl);

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
    <Card className="w-full max-w-4xl"> {/* Even wider card */}
      <CardHeader>
        <CardTitle>Registered Faces Gallery</CardTitle>
        <CardDescription>View registered faces, images, and their embedding vectors.</CardDescription>
      </CardHeader>
      <CardContent>
        {isLoading && <p>Loading gallery...</p>}
        {error && <p className="text-destructive">Error loading gallery: {error}</p>}
        {!isLoading && !error && galleryData && Object.keys(galleryData).length > 0 && (
          <Accordion type="multiple" className="w-full">
            {Object.entries(galleryData).map(([label, entries]) => (
              <AccordionItem value={label} key={label}>
                <AccordionTrigger className="text-xl font-semibold">{label} ({entries?.length || 0} entries)</AccordionTrigger>
                <AccordionContent>
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 p-4">
                    {entries.map((entry, index) => {
                      // Construct image URL using backend base URL and filename
                      const backendBaseUrl = import.meta.env.VITE_BACKEND_URL;
                      // Use image_filename now
                      const imageUrl = entry.image_filename ? `${backendBaseUrl}/images/${entry.image_filename}` : undefined;

                      return (
                        <Card key={`${label}-${index}`} className="overflow-hidden">
                          <CardHeader className="p-0">
                            {imageUrl ? (
                              <img
                                src={imageUrl}
                                alt={`Registered face for ${label} #${index + 1}`}
                                className="w-full h-48 object-cover" // Fixed height image
                                onError={(e) => { e.currentTarget.src = 'placeholder.png'; e.currentTarget.alt = 'Image not found'; }} // Basic error handling
                              />
                            ) : (
                              <div className="w-full h-48 bg-muted flex items-center justify-center text-muted-foreground">No Image</div>
                            )}
                          </CardHeader>
                          <CardContent className="p-4 space-y-2">
                             <CardTitle className="text-sm mb-2">Entry #{index + 1}</CardTitle>
                             {/* Accordion for Embedding Chart */}
                             <Accordion type="single" collapsible className="w-full">
                               <AccordionItem value="embedding-chart">
                                 <AccordionTrigger className="text-xs">Show Embedding Chart</AccordionTrigger>
                                 <AccordionContent>
                                   {entry.embedding && entry.embedding.length > 0 ? (
                                     <div style={{ width: '100%', height: 150 }}> {/* Fixed height container */}
                                       <ResponsiveContainer>
                                         <BarChart
                                           data={formatEmbeddingDataForChart(entry.embedding)}
                                           margin={{ top: 5, right: 5, left: -30, bottom: 5 }} // Adjust margins
                                           barGap={0} // No gap between bars for dense view
                                           barCategoryGap={0}
                                         >
                                           <CartesianGrid strokeDasharray="3 3" vertical={false} />
                                           <XAxis dataKey="index" hide /> {/* Hide X-axis labels (too many) */}
                                           <YAxis domain={['auto', 'auto']} tick={{ fontSize: 10 }} />
                                           <Tooltip
                                             contentStyle={{ fontSize: 10, padding: '2px 5px' }}
                                             labelFormatter={(label) => `Index: ${label}`}
                                             formatter={(value: number) => [value.toFixed(4), 'Value']}
                                           />
                                           <Bar dataKey="value" fill="#8884d8" />
                                         </BarChart>
                                       </ResponsiveContainer>
                                     </div>
                                   ) : (
                                     <p className="text-xs text-muted-foreground">Embedding data not available.</p>
                                   )}
                                 </AccordionContent>
                               </AccordionItem>
                             </Accordion>
                           </CardContent>
                        </Card>
                      );
                    })}
                  </div>
                </AccordionContent>
              </AccordionItem>
            ))}
          </Accordion>
        )}
        {!isLoading && !error && (!galleryData || Object.keys(galleryData).length === 0) && (
          <p>No faces registered yet.</p>
        )}
      </CardContent>
    </Card>
  );
};

export default GalleryPage;