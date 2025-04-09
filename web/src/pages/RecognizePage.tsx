import React, { useState, useRef, useCallback, useEffect } from 'react'; // Added useEffect
import Webcam from 'react-webcam';
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Switch } from "@/components/ui/switch"; // Added Switch
import { toast } from 'sonner';
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion"; // Added Accordion
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Cell, ComposedChart, Line } from 'recharts'; // Added recharts components

// Updated interface to match backend response
interface RecognitionResult {
  label: string;
  similarity: number;
  query_embedding: number[] | null;
  matched_embedding: number[] | null;
  matched_image_filename: string | null;
  query_landmarks_5pt: [number, number][] | null; // Renamed: 5-point landmarks from detector
  query_landmarks_68pt?: [number, number][] | null; // Optional: 68-point landmarks from PFLD
}

// Helper function to calculate difference and prepare data for comparison chart
const prepareComparisonData = (query: number[] | null, matched: number[] | null) => {
  if (!query || !matched || query.length !== matched.length) return [];
  return query.map((qVal, index) => {
    const mVal = matched[index];
    const diff = Math.abs(qVal - mVal);
    return {
      index: index,
      queryValue: qVal,
      matchedValue: mVal,
      difference: diff,
    };
  });
};

// Helper function for color coding based on difference (adjust scale as needed)
// Returns HSL color string: green (low diff) to red (high diff)
const getColorForDifference = (diff: number, maxDiff = 1.0): string => { // Assuming max possible diff is around 1.0-2.0 for normalized vectors, adjust if needed
  const normalizedDiff = Math.min(diff / maxDiff, 1.0); // Normalize diff to 0-1 range
  // Hue: 120 (green) for 0 diff, 0 (red) for max diff
  const hue = 120 * (1 - normalizedDiff);
  return `hsl(${hue}, 100%, 50%)`; // Full saturation, 50% lightness
};

const RecognizePage: React.FC = () => {
  const [imageSrc, setImageSrc] = useState<string | null>(null); // For uploaded image preview or webcam snapshot (base64)
  const [activeTab, setActiveTab] = useState('upload'); // Track active tab
  const webcamRef = useRef<Webcam>(null);
  const [isManualRecognizing, setIsManualRecognizing] = useState(false); // Renamed for clarity
  const [recognitionResult, setRecognitionResult] = useState<RecognitionResult | null>(null);
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string | undefined>(undefined);
  // Landmarks are now always requested and shown if available
  // const [showKeypoints, setShowKeypoints] = useState(false); // Removed: Landmarks always enabled
  const queryImageRef = useRef<HTMLImageElement>(null); // Ref to get query image dimensions
  const [isAutoRecognizing, setIsAutoRecognizing] = useState(false);
  // const [isProcessingFrame, setIsProcessingFrame] = useState(false); // Using ref instead
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const recognitionInProgressRef = useRef(false); // Ref to track if a recognition call is active
  const autoRecognizeInterval = 1000; // ms, e.g., 1 FPS. Adjust as needed.

  const handleRecognize = async (imageDataUrl: string | null = null) => {
    const currentImage = imageDataUrl ?? imageSrc; // Use provided image or state
    if (!currentImage) {
      // This case should ideally not happen in auto-mode if screenshot is always provided
      // For manual mode, the button is disabled if !imageSrc
      toast.error('No image data available for recognition.');
      return;
    }

    // Use ref to prevent overlapping calls more reliably
    if (recognitionInProgressRef.current) {
        console.log("Recognition already in progress, skipping frame.");
        return;
    }
    recognitionInProgressRef.current = true;

    // Don't clear results immediately in auto mode, let it update
    // setRecognitionResult(null);
    // Only show toast for manual clicks
    if (!imageDataUrl) {
        toast.info('Recognizing face...');
    }

    try {
      const fetchRes = await fetch(currentImage);
      const blob = await fetchRes.blob();
      const formData = new FormData();
      formData.append('file', blob, 'face_image.jpg');

      const url = new URL(`${import.meta.env.VITE_BACKEND_URL}/recognize`);
      url.searchParams.append('extract_landmarks', 'true');
      // console.log("Requesting URL:", url.toString()); // Less verbose logging

      const response = await fetch(url.toString(), {
        method: 'POST',
        body: formData,
      });

      const result = await response.json();

      if (!response.ok) {
        throw new Error(result.detail || `HTTP error! status: ${response.status}`);
      }

      setRecognitionResult(result);
      // Only update imageSrc if it was from auto-recognize to show the *processed* frame
      if (imageDataUrl) {
          setImageSrc(imageDataUrl);
      }
      // Don't toast success in auto mode, too noisy
      // toast.success(`Recognition complete.`);

    } catch (error) {
      console.error('Recognition failed:', error);
      // Avoid flooding toasts in auto mode
      if (!isAutoRecognizing) {
          toast.error(`Recognition failed: ${error instanceof Error ? error.message : String(error)}`);
      }
      setRecognitionResult(null); // Clear result on error
    } finally {
      recognitionInProgressRef.current = false;
    }
  };

  // Wrapper for the manual recognize button
  const handleManualRecognize = () => {
      if (isAutoRecognizing || recognitionInProgressRef.current) return; // Don't allow manual trigger if auto is on or processing
      setIsManualRecognizing(true); // Show loading state on button
      handleRecognize().finally(() => {
          setIsManualRecognizing(false); // Hide loading state
      });
  };

  // --- Fetch devices ---
  useEffect(() => {
    const getDevices = async () => {
      try {
        await navigator.mediaDevices.getUserMedia({ video: true }); // Request permission first
        const mediaDevices = await navigator.mediaDevices.enumerateDevices();
        const videoDevices = mediaDevices.filter(({ kind }) => kind === 'videoinput');
        setDevices(videoDevices);
        // Optionally set the first device as default, or let the user choose
        // if (videoDevices.length > 0) {
        //   setSelectedDeviceId(videoDevices[0].deviceId);
        // }
        if (videoDevices.length > 0 && !selectedDeviceId) {
             // Set default if none selected and devices are available
             setSelectedDeviceId(videoDevices[0].deviceId);
        }
      } catch (err) {
        console.error("Error fetching media devices:", err);
        toast.error("Could not access camera devices. Please check permissions.");
      }
    };
    getDevices();
  }, [selectedDeviceId]); // Re-run if selectedDeviceId changes? No, device fetch is once. Keep []
  // --- End Fetch devices ---

  // --- Auto Recognition Loop ---
  useEffect(() => {
    if (isAutoRecognizing && activeTab === 'webcam') {
      intervalRef.current = setInterval(async () => {
        if (recognitionInProgressRef.current) {
          // console.log("Skipping frame due to ongoing recognition.");
          return; // Don't capture if the previous one is still processing
        }
        if (webcamRef.current) {
          const screenshot = webcamRef.current.getScreenshot();
          if (screenshot) {
            // Don't await here, let it run in the background
            handleRecognize(screenshot);
          }
        }
      }, autoRecognizeInterval); // Interval time
    } else {
      // Clear interval if auto-recognizing stops or tab changes
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    }

    // Cleanup function
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
        recognitionInProgressRef.current = false; // Reset flag on cleanup
      }
    };
    // Depend on isAutoRecognizing and activeTab to start/stop the interval correctly
  }, [isAutoRecognizing, activeTab, autoRecognizeInterval]);
  // --- End Auto Recognition Loop ---

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setRecognitionResult(null); // Clear result on new file
    setIsAutoRecognizing(false); // Stop auto if user uploads file
    const file = event.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setImageSrc(reader.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleWebcamCapture = useCallback(() => {
    setRecognitionResult(null); // Clear result on new capture
    setIsAutoRecognizing(false); // Stop auto if user takes manual snapshot
    if (webcamRef.current) {
      const image = webcamRef.current.getScreenshot();
      if (image) {
        setImageSrc(image);
        toast.success('Snapshot captured!');
      } else {
        toast.error('Could not capture snapshot.');
      }
    }
  }, [webcamRef]);

  return (
    <Card className="w-full"> {/* Removed max-w-md */}
      <CardHeader>
        <CardTitle>Recognize Face</CardTitle>
        <CardDescription>Provide an image via upload or webcam to recognize a face.</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Clear image preview when switching tabs */}
        <Tabs value={activeTab} onValueChange={(value) => { setImageSrc(null); setRecognitionResult(null); setIsAutoRecognizing(false); setActiveTab(value); }}>
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="upload">Upload File</TabsTrigger>
            <TabsTrigger value="webcam">Use Webcam</TabsTrigger>
          </TabsList>
          <TabsContent value="upload" className="mt-4 space-y-4">
             <Label htmlFor="picture">Upload Picture</Label>
             <Input id="picture" type="file" accept="image/*" onChange={handleFileChange} disabled={isManualRecognizing} />
             {/* Query Image Preview Area */}
             {activeTab === 'upload' && imageSrc && (
                <div className="relative mt-2 mx-auto max-h-60 w-fit"> {/* Container for image and overlay */}
                   <img
                      ref={queryImageRef}
                      src={imageSrc}
                      alt="Upload Preview"
                      className="block max-h-60 rounded border" // Ensure image is block for layout
                      onLoad={() => { /* Could trigger redraw if needed */ }}
                   />
                   {/* Landmark Overlay - Always shown if landmarks available */}
                   {queryImageRef.current && (recognitionResult?.query_landmarks_5pt || recognitionResult?.query_landmarks_68pt) && ( // Removed showKeypoints condition
                     <svg
                       className="absolute top-0 left-0 w-full h-full pointer-events-none"
                       viewBox={`0 0 ${queryImageRef.current.naturalWidth} ${queryImageRef.current.naturalHeight}`}
                       preserveAspectRatio="none"
                     >
                       {/* Draw 68-point landmark connections (mesh) */}
                       {recognitionResult?.query_landmarks_68pt && recognitionResult.query_landmarks_68pt.length === 68 && (
                         <g stroke="lime" strokeWidth="0.75" fill="none"> {/* Group for lines */}
                           {/* Jaw line */}
                           <polyline points={recognitionResult.query_landmarks_68pt.slice(0, 17).map(p => p.join(',')).join(' ')} />
                           {/* Left eyebrow */}
                           <polyline points={recognitionResult.query_landmarks_68pt.slice(17, 22).map(p => p.join(',')).join(' ')} />
                           {/* Right eyebrow */}
                           <polyline points={recognitionResult.query_landmarks_68pt.slice(22, 27).map(p => p.join(',')).join(' ')} />
                           {/* Nose bridge */}
                           <polyline points={recognitionResult.query_landmarks_68pt.slice(27, 31).map(p => p.join(',')).join(' ')} />
                           {/* Lower nose */}
                           <polyline points={recognitionResult.query_landmarks_68pt.slice(31, 36).map(p => p.join(',')).join(' ')} />
                           {/* Left eye (closed loop) */}
                           <polyline points={[...recognitionResult.query_landmarks_68pt.slice(36, 42), recognitionResult.query_landmarks_68pt[36]].map(p => p.join(',')).join(' ')} />
                           {/* Right eye (closed loop) */}
                           <polyline points={[...recognitionResult.query_landmarks_68pt.slice(42, 48), recognitionResult.query_landmarks_68pt[42]].map(p => p.join(',')).join(' ')} />
                           {/* Outer lip (closed loop) */}
                           <polyline points={[...recognitionResult.query_landmarks_68pt.slice(48, 60), recognitionResult.query_landmarks_68pt[48]].map(p => p.join(',')).join(' ')} />
                           {/* Inner lip (closed loop) */}
                           <polyline points={[...recognitionResult.query_landmarks_68pt.slice(60, 68), recognitionResult.query_landmarks_68pt[60]].map(p => p.join(',')).join(' ')} />
                         </g>
                       )}

                       {/* Draw 5-point landmark points (blue) */}
                       {recognitionResult?.query_landmarks_5pt?.map(([x, y], index) => (
                         <circle
                           key={`landmark-5pt-${index}`}
                           cx={x}
                           cy={y}
                           r="2.5"
                           fill="blue"
                           stroke="white"
                           strokeWidth="0.5"
                         />
                       ))}
                       {/* Draw 68-point landmark points (red) */}
                       {recognitionResult?.query_landmarks_68pt?.map(([x, y], index) => (
                         <circle
                           key={`landmark-68pt-${index}`}
                           cx={x}
                           cy={y}
                           r="1.5"
                           fill="red"
                           stroke="white"
                           strokeWidth="0.5"
                         />
                       ))}
                     </svg>
                   )}
                </div>
             )}
          </TabsContent>
          <TabsContent value="webcam" className="mt-4 space-y-4 flex flex-col items-center">
             {/* Webcam Selector */}
             {devices.length > 1 && ( // Only show selector if multiple cameras exist
               <div className="w-full max-w-xs space-y-1">
                 <Label htmlFor="webcam-select-recognize">Select Camera</Label>
                 <select
                   id="webcam-select-recognize" // Unique ID
                   value={selectedDeviceId || ''}
                   onChange={(e) => setSelectedDeviceId(e.target.value)}
                   className="flex h-10 w-full items-center justify-between rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50" // Basic styling similar to shadcn Select
                   disabled={isAutoRecognizing || recognitionInProgressRef.current}
                 >
                   {devices.map((device) => (
                     <option key={device.deviceId} value={device.deviceId}>
                       {device.label || `Camera ${devices.indexOf(device) + 1}`}
                     </option>
                   ))}
                 </select>
               </div>
             )}
             {/* End Webcam Selector */}

             <Webcam
                audio={false}
                ref={webcamRef}
                screenshotFormat="image/jpeg"
                className="rounded border w-full max-w-xs" // Keep webcam size constrained
                videoConstraints={{ deviceId: selectedDeviceId ? { exact: selectedDeviceId } : undefined }}
             />
             {/* Query Image Preview Area (Snapshot) */}
             {activeTab === 'webcam' && imageSrc && (
                <div className="relative mt-2 mx-auto max-h-60 w-fit"> {/* Container for image and overlay */}
                   <img
                      ref={queryImageRef}
                      src={imageSrc}
                      alt="Snapshot Preview"
                      className="block max-h-60 rounded border"
                      onLoad={() => { /* Could trigger redraw if needed */ }}
                   />
                   {/* Landmark Overlay - Always shown if landmarks available */}
                   {queryImageRef.current && (recognitionResult?.query_landmarks_5pt || recognitionResult?.query_landmarks_68pt) && ( // Removed showKeypoints condition
                     <svg
                       className="absolute top-0 left-0 w-full h-full pointer-events-none"
                       viewBox={`0 0 ${queryImageRef.current.naturalWidth} ${queryImageRef.current.naturalHeight}`}
                       preserveAspectRatio="none"
                     >
                       {/* Draw 68-point landmark connections (mesh) */}
                       {recognitionResult?.query_landmarks_68pt && recognitionResult.query_landmarks_68pt.length === 68 && (
                         <g stroke="lime" strokeWidth="0.75" fill="none"> {/* Group for lines */}
                           {/* Jaw line */}
                           <polyline points={recognitionResult.query_landmarks_68pt.slice(0, 17).map(p => p.join(',')).join(' ')} />
                           {/* Left eyebrow */}
                           <polyline points={recognitionResult.query_landmarks_68pt.slice(17, 22).map(p => p.join(',')).join(' ')} />
                           {/* Right eyebrow */}
                           <polyline points={recognitionResult.query_landmarks_68pt.slice(22, 27).map(p => p.join(',')).join(' ')} />
                           {/* Nose bridge */}
                           <polyline points={recognitionResult.query_landmarks_68pt.slice(27, 31).map(p => p.join(',')).join(' ')} />
                           {/* Lower nose */}
                           <polyline points={recognitionResult.query_landmarks_68pt.slice(31, 36).map(p => p.join(',')).join(' ')} />
                           {/* Left eye (closed loop) */}
                           <polyline points={[...recognitionResult.query_landmarks_68pt.slice(36, 42), recognitionResult.query_landmarks_68pt[36]].map(p => p.join(',')).join(' ')} />
                           {/* Right eye (closed loop) */}
                           <polyline points={[...recognitionResult.query_landmarks_68pt.slice(42, 48), recognitionResult.query_landmarks_68pt[42]].map(p => p.join(',')).join(' ')} />
                           {/* Outer lip (closed loop) */}
                           <polyline points={[...recognitionResult.query_landmarks_68pt.slice(48, 60), recognitionResult.query_landmarks_68pt[48]].map(p => p.join(',')).join(' ')} />
                           {/* Inner lip (closed loop) */}
                           <polyline points={[...recognitionResult.query_landmarks_68pt.slice(60, 68), recognitionResult.query_landmarks_68pt[60]].map(p => p.join(',')).join(' ')} />
                         </g>
                       )}

                       {/* Draw 5-point landmark points (blue) */}
                       {recognitionResult?.query_landmarks_5pt?.map(([x, y], index) => (
                         <circle
                           key={`landmark-snap-5pt-${index}`}
                           cx={x}
                           cy={y}
                           r="2.5"
                           fill="blue"
                           stroke="white"
                           strokeWidth="0.5"
                         />
                       ))}
                       {/* Draw 68-point landmark points (red) */}
                       {recognitionResult?.query_landmarks_68pt?.map(([x, y], index) => (
                         <circle
                           key={`landmark-snap-68pt-${index}`}
                           cx={x}
                           cy={y}
                           r="1.5"
                           fill="red"
                           stroke="white"
                           strokeWidth="0.5"
                         />
                       ))}
                     </svg>
                   )}
                </div>
             )}
            <div className="flex space-x-2"> {/* Container for buttons */}
               <Button onClick={handleWebcamCapture} disabled={isAutoRecognizing || recognitionInProgressRef.current}>Take Snapshot</Button>
               <Button
                  onClick={() => setIsAutoRecognizing(!isAutoRecognizing)}
                  disabled={recognitionInProgressRef.current}
                  variant={isAutoRecognizing ? "destructive" : "default"} // Style change when active
                >
                  {isAutoRecognizing ? 'Stop Auto Recognize' : 'Start Auto Recognize'}
                </Button>
            </div>
          </TabsContent>
        </Tabs>

        {/* Recognize Button (Only for Upload Tab) */}
        {activeTab === 'upload' && (
            <div className="flex items-center justify-center pt-2">
                <Button onClick={handleManualRecognize} disabled={!imageSrc || isManualRecognizing} className="w-full">
                    {isManualRecognizing ? 'Recognizing...' : 'Recognize Face'}
                </Button>
            </div>
        )}
        {/* Keypoints Toggle Removed */}
           {/* <div className="flex items-center space-x-2 flex-shrink-0"> */}
              {/* <Switch */}
                 {/* id="show-keypoints-toggle" */}
                 {/* checked={showKeypoints} */}
                 {/* onCheckedChange={setShowKeypoints} */}
                 {/* disabled={isRecognizing} */}
              {/* /> */}
              {/* <Label htmlFor="show-keypoints-toggle" className="text-sm whitespace-nowrap">Show Keypoints</Label> */}
           {/* </div> */}
        {/* Removed stray closing div */}

        {/* --- Recognition Result Display --- */}
        {recognitionResult && (
          <Card className="mt-4 w-full bg-secondary/50"> {/* Removed max-w-4xl */}
            <CardHeader> {/* Removed flex layout and toggle from header */}
              <CardTitle>Recognition Result</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <p>Label: <span className="font-semibold">{recognitionResult.label}</span></p>
              <p>Similarity: <span className="font-semibold">{(recognitionResult.similarity * 100).toFixed(2)}%</span></p>

              {/* Display matched image and comparison chart if match found */}
              {recognitionResult.label !== 'unknown' && recognitionResult.matched_image_filename && recognitionResult.matched_embedding && recognitionResult.query_embedding && (
                <>
                  <div>
                    <h4 className="text-sm font-medium mb-2">Matched Image:</h4>
                    <img
                      src={`${import.meta.env.VITE_BACKEND_URL}/images/${recognitionResult.matched_image_filename}`}
                      alt={`Matched face for ${recognitionResult.label}`}
                      className="rounded border max-h-40 mx-auto"
                      onError={(e) => { e.currentTarget.src = 'placeholder.png'; e.currentTarget.alt = 'Image not found'; }}
                    />
                  </div>

                  <Accordion type="single" collapsible className="w-full">
                    <AccordionItem value="comparison-chart">
                      <AccordionTrigger className="text-xs">Show Embedding Comparison Chart</AccordionTrigger>
                      <AccordionContent>
                        <div style={{ width: '100%', height: 200 }}> {/* Adjust height as needed */}
                          <ResponsiveContainer>
                            <ComposedChart
                              data={prepareComparisonData(recognitionResult.query_embedding, recognitionResult.matched_embedding)}
                              margin={{ top: 5, right: 5, left: -30, bottom: 5 }}
                            >
                              <CartesianGrid strokeDasharray="3 3" vertical={false} />
                              <XAxis dataKey="index" hide />
                              <YAxis tick={{ fontSize: 10 }} />
                              <Tooltip
                                contentStyle={{ fontSize: 10, padding: '2px 5px' }}
                                labelFormatter={(label) => `Index: ${label}`}
                                formatter={(value: number, name: string, props: any) => {
                                   // Show query, matched, and difference values
                                   const data = props.payload;
                                   return [
                                       `Query: ${data.queryValue.toFixed(4)}`,
                                       `Matched: ${data.matchedValue.toFixed(4)}`,
                                       `Difference: ${data.difference.toFixed(4)}`
                                   ];
                                }}
                              />
                              {/* Bar chart showing the difference, color-coded */}
                              <Bar dataKey="difference" name="Difference" barSize={1}>
                                {prepareComparisonData(recognitionResult.query_embedding, recognitionResult.matched_embedding).map((entry) => (
                                  <Cell key={`cell-${entry.index}`} fill={getColorForDifference(entry.difference)} />
                                ))}
                              </Bar>
                               {/* Optional: Add lines for query/matched values if desired */}
                               {/* <Line type="monotone" dataKey="queryValue" stroke="#8884d8" dot={false} strokeWidth={0.5} name="Query" /> */}
                               {/* <Line type="monotone" dataKey="matchedValue" stroke="#82ca9d" dot={false} strokeWidth={0.5} name="Matched" /> */}
                            </ComposedChart>
                          </ResponsiveContainer>
                        </div>
                        <p className="text-xs text-muted-foreground mt-1">Bar color indicates difference (Green=Low, Red=High)</p>
                      </AccordionContent>
                    </AccordionItem>
                  </Accordion>
                </>
              )}
            </CardContent>
          </Card>
        )}
        {/* --- End Recognition Result Display --- */}
      </CardContent>
    </Card>
  );
};

export default RecognizePage;