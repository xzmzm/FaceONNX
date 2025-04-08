import React, { useState, useEffect } from 'react';
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/card';
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { toast } from 'sonner';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts';
import { Button } from '@/components/ui/button'; // Import Button
import { Trash2 } from 'lucide-react'; // Import an icon for delete
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog" // Import Alert Dialog

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

        // The backend returns { "data": { ...gallery... } }
        const responseData = await response.json();
        if (!responseData || typeof responseData.data !== 'object') {
            throw new Error("Invalid data structure received from gallery endpoint.");
        }
        const actualGalleryData: GalleryData = responseData.data;
        setGalleryData(actualGalleryData);
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
  }, []);

  // --- Delete Handler ---
  const handleDeleteEntry = async (labelToDelete: string, filenameToDelete: string) => {
    // Find the entry to confirm deletion details (optional but good UX)
    const entryToDelete = galleryData?.[labelToDelete]?.find(entry => entry.image_filename === filenameToDelete);
    if (!entryToDelete) {
      toast.error("Could not find entry to delete.");
      return;
    }

    // Confirmation is handled by AlertDialog, this function is called on confirm

    toast.info(`Deleting entry for ${labelToDelete}...`);
    try {
      const backendBaseUrl = import.meta.env.VITE_BACKEND_URL;
      const deleteUrl = new URL(`${backendBaseUrl}/delete_entry`);
      deleteUrl.searchParams.append('label', labelToDelete);
      deleteUrl.searchParams.append('filename', filenameToDelete);

      const response = await fetch(deleteUrl.toString(), {
        method: 'DELETE',
      });

      const result = await response.json();

      if (!response.ok) {
        throw new Error(result.detail || `HTTP error! status: ${response.status}`);
      }

      toast.success(`Successfully deleted entry for ${labelToDelete} (${filenameToDelete})`);

      // Update local state for immediate feedback
      setGalleryData(currentData => {
        if (!currentData) return null;

        const newData = { ...currentData };
        const entriesForLabel = newData[labelToDelete];

        if (entriesForLabel) {
          const updatedEntries = entriesForLabel.filter(entry => entry.image_filename !== filenameToDelete);

          if (updatedEntries.length === 0) {
            // If no entries left for this label, remove the label itself
            delete newData[labelToDelete];
          } else {
            // Otherwise, update the entries for the label
            newData[labelToDelete] = updatedEntries;
          }
        }
        return newData;
      });

    } catch (error) {
      console.error('Deletion failed:', error);
      toast.error(`Deletion failed: ${error instanceof Error ? error.message : String(error)}`);
    }
  };
  // --- End Delete Handler ---

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
                          <CardContent className="p-4 space-y-2 relative"> {/* Added relative positioning */}
                             {/* Delete Button Top Right */}
                             <AlertDialog>
                               <AlertDialogTrigger asChild>
                                  <Button
                                    variant="ghost"
                                    size="icon"
                                    className="absolute top-1 right-1 h-6 w-6 text-destructive hover:bg-destructive/10" // Positioned top-right
                                    aria-label="Delete entry"
                                  >
                                    <Trash2 className="h-4 w-4" />
                                  </Button>
                               </AlertDialogTrigger>
                               <AlertDialogContent>
                                 <AlertDialogHeader>
                                   <AlertDialogTitle>Are you sure?</AlertDialogTitle>
                                   <AlertDialogDescription>
                                     This action cannot be undone. This will permanently delete the face entry
                                     for <span className="font-semibold">{label}</span> (Image: <span className="font-mono text-xs">{entry.image_filename}</span>)
                                     and remove its data from the system.
                                   </AlertDialogDescription>
                                 </AlertDialogHeader>
                                 <AlertDialogFooter>
                                   <AlertDialogCancel>Cancel</AlertDialogCancel>
                                   <AlertDialogAction
                                      onClick={() => handleDeleteEntry(label, entry.image_filename)}
                                      className="bg-destructive text-destructive-foreground hover:bg-destructive/90" // Destructive style
                                   >
                                      Delete
                                   </AlertDialogAction>
                                 </AlertDialogFooter>
                               </AlertDialogContent>
                             </AlertDialog>

                             <CardTitle className="text-sm mb-2 pt-2">Entry #{index + 1}</CardTitle> {/* Added padding top */}
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