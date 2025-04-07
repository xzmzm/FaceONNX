import React, { useState, useRef, useCallback, useEffect } from 'react'; // Added useEffect
import Webcam from 'react-webcam';
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { toast } from 'sonner';

const RegisterPage: React.FC = () => {
  const [label, setLabel] = useState('');
  const [imageSrc, setImageSrc] = useState<string | null>(null); // For uploaded image preview or webcam snapshot (base64)
  const [activeTab, setActiveTab] = useState('upload'); // Track active tab
  const webcamRef = useRef<Webcam>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string | undefined>(undefined);

  const handleRegister = async () => {
    if (!label || !imageSrc) {
      toast.error('Please provide a label and an image/snapshot.');
      return;
    }
    setIsSubmitting(true);
    toast.info(`Registering face for: ${label}...`);

    try {
      // Convert base64 image to Blob
      const fetchRes = await fetch(imageSrc);
      const blob = await fetchRes.blob();

      const formData = new FormData();
      formData.append('label', label);
      // Use a consistent filename, the backend doesn't rely on it
      formData.append('file', blob, 'face_image.jpg');

      // Use environment variable for backend URL
      const backendUrl = `${import.meta.env.VITE_BACKEND_URL}/register`;

      const response = await fetch(backendUrl, {
        method: 'POST',
        body: formData, // Send as FormData
      });

      const result = await response.json();

      if (!response.ok) {
        throw new Error(result.detail || `HTTP error! status: ${response.status}`);
      }

      toast.success(result.message || `Successfully registered face for ${label}.`);
      setLabel('');
      setImageSrc(null); // Clear preview
      // TODO: Clear file input or stop webcam if needed

    } catch (error) {
      console.error('Registration failed:', error);
      toast.error(`Registration failed: ${error instanceof Error ? error.message : String(error)}`);
    } finally {
      setIsSubmitting(false);
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
    if (webcamRef.current) {
      const image = webcamRef.current.getScreenshot();
      if (image) {
        setImageSrc(image); // Set base64 image source
        toast.success('Snapshot captured!');
      } else {
        toast.error('Could not capture snapshot.');
      }
    }
  }, [webcamRef]);

  return (
    <Card className="w-full max-w-md">
      <CardHeader>
        <CardTitle>Register New Face</CardTitle>
        <CardDescription>Enter a label and provide an image via upload or webcam.</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-2">
          <Label htmlFor="label">Label / Name</Label>
          <Input
            id="label"
            placeholder="Enter a unique name"
            value={label}
            onChange={(e) => setLabel(e.target.value)}
            disabled={isSubmitting}
          />
        </div>

        {/* Clear image preview when switching tabs */}
        <Tabs value={activeTab} onValueChange={(value) => { setImageSrc(null); setActiveTab(value); }}>
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="upload">Upload File</TabsTrigger>
            <TabsTrigger value="webcam">Use Webcam</TabsTrigger>
          </TabsList>
          <TabsContent value="upload" className="mt-4 space-y-4">
             <Label htmlFor="picture">Upload Picture</Label>
             <Input id="picture" type="file" accept="image/*" onChange={handleFileChange} disabled={isSubmitting} />
            {activeTab === 'upload' && imageSrc && <img src={imageSrc} alt="Upload Preview" className="mt-2 max-h-40 rounded border" />}
         </TabsContent>
         <TabsContent value="webcam" className="mt-4 space-y-4 flex flex-col items-center">
            {/* Webcam Selector */}
            {devices.length > 1 && ( // Only show selector if multiple cameras exist
              <div className="w-full max-w-xs space-y-1">
                <Label htmlFor="webcam-select">Select Camera</Label>
                <select
                  id="webcam-select"
                  value={selectedDeviceId || ''}
                  onChange={(e) => setSelectedDeviceId(e.target.value)}
                  className="flex h-10 w-full items-center justify-between rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50" // Basic styling similar to shadcn Select
                  disabled={isSubmitting}
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
               className="rounded border w-full max-w-xs" // Adjust size as needed
               videoConstraints={{ deviceId: selectedDeviceId ? { exact: selectedDeviceId } : undefined }} // Use selected device
            />
            {activeTab === 'webcam' && imageSrc && <img src={imageSrc} alt="Snapshot Preview" className="mt-2 max-h-40 rounded border" />}
            <Button onClick={handleWebcamCapture} disabled={isSubmitting}>Take Snapshot</Button>
          </TabsContent>
        </Tabs>

        <Button onClick={handleRegister} disabled={!label || !imageSrc || isSubmitting} className="w-full">
          {isSubmitting ? 'Registering...' : 'Register Face'}
        </Button>
      </CardContent>
    </Card>
  );
};

export default RegisterPage;