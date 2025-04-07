import React, { useState, useRef, useCallback, useEffect } from 'react'; // Added useEffect
import Webcam from 'react-webcam';
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { toast } from 'sonner';

interface RecognitionResult {
  label: string;
  similarity: number;
}

const RecognizePage: React.FC = () => {
  const [imageSrc, setImageSrc] = useState<string | null>(null); // For uploaded image preview or webcam snapshot (base64)
  const [activeTab, setActiveTab] = useState('upload'); // Track active tab
  const webcamRef = useRef<Webcam>(null);
  const [isRecognizing, setIsRecognizing] = useState(false);
  const [recognitionResult, setRecognitionResult] = useState<RecognitionResult | null>(null);
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string | undefined>(undefined);

  const handleRecognize = async () => {
    if (!imageSrc) {
      toast.error('Please provide an image/snapshot to recognize.');
      return;
    }
    setIsRecognizing(true);
    setRecognitionResult(null); // Clear previous result
    toast.info('Recognizing face...');

    try {
      // Convert base64 image to Blob
      const fetchRes = await fetch(imageSrc);
      const blob = await fetchRes.blob();

      const formData = new FormData();
      // Use a consistent filename, the backend doesn't rely on it
      formData.append('file', blob, 'face_image.jpg');

      // Use environment variable for backend URL
      const backendUrl = `${import.meta.env.VITE_BACKEND_URL}/recognize`;

      const response = await fetch(backendUrl, {
        method: 'POST',
        body: formData,
      });

      const result = await response.json();

      if (!response.ok) {
        throw new Error(result.detail || `HTTP error! status: ${response.status}`);
      }

      setRecognitionResult(result); // Store the result {label: string, similarity: float}
      toast.success(`Recognition complete.`);

    } catch (error) {
      console.error('Recognition failed:', error);
      toast.error(`Recognition failed: ${error instanceof Error ? error.message : String(error)}`);
      setRecognitionResult(null);
    } finally {
      setIsRecognizing(false);
    }
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
  }, []); // Run once on mount
  // --- End Fetch devices ---

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setRecognitionResult(null); // Clear result on new file
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
    <Card className="w-full max-w-md">
      <CardHeader>
        <CardTitle>Recognize Face</CardTitle>
        <CardDescription>Provide an image via upload or webcam to recognize a face.</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Clear image preview when switching tabs */}
        <Tabs value={activeTab} onValueChange={(value) => { setImageSrc(null); setRecognitionResult(null); setActiveTab(value); }}>
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="upload">Upload File</TabsTrigger>
            <TabsTrigger value="webcam">Use Webcam</TabsTrigger>
          </TabsList>
          <TabsContent value="upload" className="mt-4 space-y-4">
             <Label htmlFor="picture">Upload Picture</Label>
             <Input id="picture" type="file" accept="image/*" onChange={handleFileChange} disabled={isRecognizing} />
             {activeTab === 'upload' && imageSrc && <img src={imageSrc} alt="Upload Preview" className="mt-2 max-h-40 rounded border" />}
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
                   disabled={isRecognizing}
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
                className="rounded border w-full max-w-xs"
                videoConstraints={{ deviceId: selectedDeviceId ? { exact: selectedDeviceId } : undefined }} // Use selected device
             />
             {activeTab === 'webcam' && imageSrc && <img src={imageSrc} alt="Snapshot Preview" className="mt-2 max-h-40 rounded border" />}
            <Button onClick={handleWebcamCapture} disabled={isRecognizing}>Take Snapshot</Button>
          </TabsContent>
        </Tabs>

        <Button onClick={handleRecognize} disabled={!imageSrc || isRecognizing} className="w-full">
          {isRecognizing ? 'Recognizing...' : 'Recognize Face'}
        </Button>

        {recognitionResult && (
          <Card className="mt-4 bg-secondary/50">
            <CardHeader>
              <CardTitle>Recognition Result</CardTitle>
            </CardHeader>
            <CardContent>
              <p>Label: <span className="font-semibold">{recognitionResult.label}</span></p>
              <p>Similarity: <span className="font-semibold">{(recognitionResult.similarity * 100).toFixed(2)}%</span></p>
            </CardContent>
          </Card>
        )}
      </CardContent>
    </Card>
  );
};

export default RecognizePage;