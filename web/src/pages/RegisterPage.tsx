import React, { useState, useRef, useCallback, useEffect } from 'react';
import Webcam from 'react-webcam';
import { useDropzone } from 'react-dropzone';
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input'; // Need Input now
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { toast } from 'sonner';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
  DialogClose,
} from "@/components/ui/dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"; // Need Select now
import { Loader2 } from 'lucide-react';
// Removed Popover/Command related imports

const RegisterPage: React.FC = () => {
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState('webcam'); // Tab for image source
  const webcamRef = useRef<Webcam>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string | undefined>(undefined);
  const [detailedDevices, setDetailedDevices] = useState<DetailedMediaDeviceInfo[]>([]);

  // --- State for User Name Confirmation Dialog ---
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [existingLabels, setExistingLabels] = useState<{ label: string; count: number }[]>([]);
  const [isLoadingLabels, setIsLoadingLabels] = useState(false);
  const [dialogLabelInput, setDialogLabelInput] = useState(''); // Holds the final user name (from Input or Select)
  const [fetchError, setFetchError] = useState<string | null>(null);
  const [userNameMode, setUserNameMode] = useState<'new' | 'existing'>('new'); // State for New/Existing tabs
  // --- End Dialog State ---

  interface DetailedMediaDeviceInfo {
    deviceId: string;
    label: string;
    width?: number;
    height?: number;
  }

  const performRegistration = async (confirmedUserName: string) => {
    if (!confirmedUserName || !imageSrc) {
      toast.error('User Name and image are required.');
      setIsSubmitting(false);
      return;
    }
    setIsSubmitting(true);
    toast.info(`Registering face for: ${confirmedUserName}...`);

    try {
      const fetchRes = await fetch(imageSrc);
      const blob = await fetchRes.blob();
      const formData = new FormData();
      formData.append('label', confirmedUserName);
      formData.append('file', blob, 'face_image.jpg');
      const backendUrl = `${import.meta.env.VITE_BACKEND_URL}/register`;
      const response = await fetch(backendUrl, { method: 'POST', body: formData });
      const result = await response.json();
      if (!response.ok) {
        throw new Error(result.detail || `HTTP error! status: ${response.status}`);
      }
      toast.success(result.message || `Successfully registered face for ${confirmedUserName}.`);
      setImageSrc(null);
      setDialogLabelInput('');
      setUserNameMode('new'); // Reset mode on success
    } catch (error) {
      console.error('Registration failed:', error);
      toast.error(`Registration failed: ${error instanceof Error ? error.message : String(error)}`);
    } finally {
      setIsSubmitting(false);
    }
  };

  useEffect(() => {
    const getDevicesAndDetails = async () => {
      let permissionGranted = false;
      try {
        const permStream = await navigator.mediaDevices.getUserMedia({ video: true });
        permissionGranted = true;
        permStream.getTracks().forEach(track => track.stop());
      } catch (err) {
        console.error("Error requesting camera permission:", err);
        toast.error("Camera permission denied. Please allow camera access in your browser settings.");
        return;
      }

      if (!permissionGranted) return;

      try {
        const mediaDevices = await navigator.mediaDevices.enumerateDevices();
        const videoDevices = mediaDevices.filter(({ kind }) => kind === 'videoinput');
        const detailedDevicePromises = videoDevices.map(async (device, index) => {
          let width: number | undefined = undefined;
          let height: number | undefined = undefined;
          let stream: MediaStream | null = null;
          const defaultLabel = `Camera ${index + 1}`;
          try {
            stream = await navigator.mediaDevices.getUserMedia({ video: { deviceId: { exact: device.deviceId } } });
            const track = stream.getVideoTracks()[0];
            if (track) {
              const settings = track.getSettings();
              width = settings.width;
              height = settings.height;
            }
          } catch (err) {
             console.warn(`Could not get settings for ${device.label || defaultLabel} (${device.deviceId}):`, err);
          } finally {
            stream?.getTracks().forEach(track => track.stop());
          }
          return { deviceId: device.deviceId, label: device.label || defaultLabel, width, height };
        });
        const resolvedDevices = await Promise.all(detailedDevicePromises);
        setDetailedDevices(resolvedDevices);
        if (resolvedDevices.length > 0 && !selectedDeviceId) {
          setSelectedDeviceId(resolvedDevices[0].deviceId);
        }
      } catch (err) {
        console.error("Error fetching or processing media devices:", err);
        toast.error("Could not list camera devices. Please check permissions or refresh.");
      }
    };
    getDevicesAndDetails();
  }, []);

  const handleFileSelect = (file: File | null) => {
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => { setImageSrc(reader.result as string); };
      reader.readAsDataURL(file);
    } else {
      setImageSrc(null);
    }
  };

  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles && acceptedFiles.length > 0) {
      handleFileSelect(acceptedFiles[0]);
    } else {
      toast.error("Invalid file type. Please upload an image.");
      handleFileSelect(null);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/jpeg': [], 'image/png': [], 'image/webp': [], 'image/gif': [], 'image/bmp': [] },
    multiple: false
  });

  const handlePaste = useCallback(async (event: ClipboardEvent) => {
    if (isSubmitting || activeTab !== 'upload-image') return;
    const items = event.clipboardData?.items;
    if (!items) return;
    for (let i = 0; i < items.length; i++) {
      if (items[i].type.indexOf('image') !== -1) {
        const file = items[i].getAsFile();
        if (file) {
          event.preventDefault();
          toast.info("Pasting image...");
          handleFileSelect(file);
          return;
        }
      }
    }
  }, [isSubmitting, activeTab, handleFileSelect]);

  useEffect(() => {
    document.addEventListener('paste', handlePaste);
    return () => { document.removeEventListener('paste', handlePaste); };
  }, [handlePaste]);

  const handleWebcamCapture = useCallback((): string | null => {
    if (webcamRef.current) {
      const image = webcamRef.current.getScreenshot();
      if (image) {
        setImageSrc(image); // Still set state for preview in dialog
        return image; // Return the captured image data
      } else {
        return null; // Indicate failure
      }
    }
    return null; // Indicate failure if no webcam ref
  }, [webcamRef]);

  const handleRegisterClick = async () => {
    let currentImageSrc: string | null = imageSrc; // Start with current state

    // If webcam tab is active, capture snapshot first
    if (activeTab === 'webcam') {
      const capturedSrc = handleWebcamCapture(); // Returns string | null
      if (!capturedSrc) {
        toast.error('Could not capture snapshot. Please ensure the camera is working and try again.');
        return;
      }
      currentImageSrc = capturedSrc; // Use the directly returned source
    } else if (!currentImageSrc) { // Check only if not webcam tab (upload/paste)
      toast.error('Please provide an image first (upload or paste).');
      return;
    }

    // Now check the definitive image source *before* opening dialog
    if (!currentImageSrc) {
        // This message should ideally not be hit if the logic above is correct, but acts as a safeguard.
        toast.error('Image source is missing. Please capture or upload an image.');
        return;
    }
    // Note: The imageSrc state *is* set by handleWebcamCapture/handleFileSelect
    // and will be used by the Dialog's preview, which renders after this function completes.
    setDialogLabelInput(''); // Reset input
    setUserNameMode('new'); // Default to new user tab
    setIsDialogOpen(true);
    setIsLoadingLabels(true);
    setFetchError(null);
    setExistingLabels([]);

    try {
      const backendUrl = `${import.meta.env.VITE_BACKEND_URL}/gallery_data`;
      const response = await fetch(backendUrl);
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Failed to fetch gallery data' }));
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
      }
      const responseData = await response.json();
      if (!responseData || typeof responseData.data !== 'object' || responseData.data === null) {
        console.error("Invalid data structure received:", responseData);
        throw new Error("Invalid data structure received from gallery_data endpoint.");
      }
      const actualGalleryData: { [label: string]: any[] } = responseData.data;
      const formattedLabels = Object.entries(actualGalleryData).map(([label, entries]) => ({
        label,
        count: Array.isArray(entries) ? entries.length : 0,
      }));
      setExistingLabels(formattedLabels);
    } catch (error) {
      console.error('Failed to fetch or process gallery_data:', error);
      const errorMessage = error instanceof Error ? error.message : String(error);
      toast.error(`Failed to load existing user names: ${errorMessage}`);
      setFetchError(errorMessage);
      // If fetch fails, keep mode as 'new' as 'existing' list is unavailable
      setUserNameMode('new');
    } finally {
      setIsLoadingLabels(false);
    }
  };

  const handleDialogConfirm = () => {
    const finalUserName = dialogLabelInput.trim();
    if (!finalUserName) {
      toast.error('User Name cannot be empty.');
      return;
    }
    setIsDialogOpen(false);
    performRegistration(finalUserName);
  };

  // Handler for Select component change
  const handleSelectChange = (value: string) => {
    setDialogLabelInput(value);
  };

  return (
    <Card className="w-full"> {/* Removed max-w-md */}
      <CardHeader>
        <CardTitle>Register New Face</CardTitle>
        <CardDescription>Provide an image via upload or webcam, then confirm the user name.</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <Tabs value={activeTab} onValueChange={(value) => { setImageSrc(null); setActiveTab(value); }}>
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="webcam">Use Webcam</TabsTrigger>
            <TabsTrigger value="upload-image">Upload Image</TabsTrigger>
          </TabsList>
          <TabsContent value="upload-image" className="mt-4 space-y-4">
             <div
               {...getRootProps()}
               className={`relative flex flex-col items-center justify-center p-2 border-2 border-dashed rounded-md cursor-pointer text-center transition-colors group overflow-hidden
                 ${isDragActive ? 'border-primary bg-primary/10' : 'border-input hover:border-primary/50'}
                 ${isSubmitting ? 'opacity-50 cursor-not-allowed' : ''}
                 ${imageSrc ? 'max-h-[250px]' : 'h-32'}`
               }
             >
               <input {...getInputProps()} disabled={isSubmitting} />
               {imageSrc && (
                 <img src={imageSrc} alt="Upload Preview" className="block w-full max-h-[250px] object-contain rounded-md" />
               )}
               <div className={`absolute inset-0 z-10 flex flex-col items-center justify-center space-y-2 p-4 rounded-md transition-opacity duration-200 ${imageSrc ? 'opacity-0 group-hover:opacity-100 bg-black/50' : 'opacity-100'}`}>
                 {isDragActive ? (
                   <p className={`text-sm ${imageSrc ? 'text-white' : 'text-primary'}`}>Drop the image here ...</p>
                 ) : (
                   <p className={`text-sm ${imageSrc ? 'text-white' : 'text-muted-foreground'}`}>Drag 'n' drop, click to select, or Ctrl+V to paste</p>
                 )}
               </div>
             </div>
          </TabsContent>
          <TabsContent value="webcam" className="mt-4 space-y-4 flex flex-col items-center">
             {detailedDevices.length > 1 && (
               <div className="w-full max-w-xs space-y-1">
                 <Label htmlFor="webcam-select-register">Select Camera</Label>
                 <select
                   id="webcam-select-register"
                   value={selectedDeviceId || ''}
                   onChange={(e) => setSelectedDeviceId(e.target.value)}
                   className="flex h-10 w-full items-center justify-between rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
                   disabled={isSubmitting}
                 >
                   {detailedDevices.map((device) => (
                     <option key={device.deviceId} value={device.deviceId}>
                       {device.label} {device.width && device.height ? `(${device.width}x${device.height})` : ''}
                     </option>
                   ))}
                 </select>
               </div>
             )}
             <Webcam
                audio={false}
                ref={webcamRef}
                screenshotFormat="image/jpeg"
                className="rounded border w-full max-w-xs"
                videoConstraints={{ deviceId: selectedDeviceId ? { exact: selectedDeviceId } : undefined }}
             />
             {/* Snapshot button and preview removed */}
           </TabsContent>
         </Tabs>

        <div className="flex justify-center"> {/* Center the button */}
          <Button onClick={handleRegisterClick} disabled={(activeTab === 'upload-image' && !imageSrc) || isSubmitting}> {/* Removed w-full */}
            {isSubmitting ? <><Loader2 className="mr-2 h-4 w-4 animate-spin" /> Registering...</> : 'Register Face'}
          </Button>
        </div>

        <Dialog open={isDialogOpen} onOpenChange={setIsDialogOpen}>
          <DialogContent className="sm:max-w-[425px]">
            <DialogHeader>
              <DialogTitle>Confirm User Name</DialogTitle>
              <DialogDescription>
                Enter a new user name or select an existing one to add this face to their gallery.
              </DialogDescription>
            </DialogHeader>
            <div className="py-4 space-y-4"> {/* Added space-y-4 */}
              {/* Image Preview */}
              {imageSrc && (
                <div className="flex justify-center">
                  <img src={imageSrc} alt="Face to register" className="max-h-40 rounded border object-contain" />
                </div>
              )}
              {isLoadingLabels ? (
                <div className="flex items-center justify-center space-x-2 h-24"> {/* Added height for loading state */}
                  <Loader2 className="h-5 w-5 animate-spin" />
                  <span>Loading existing user names...</span>
                </div>
              ) : fetchError ? (
                 // Wrap adjacent elements in a Fragment
                 <>
                   <div className="text-red-600 text-center mb-4">
                      Error loading user names: {fetchError} <br/> You can only enter a new user name.
                   </div>
                   {/* Force 'new' mode UI if fetch failed */}
                   <div className="grid grid-cols-4 items-center gap-4">
                     <Label htmlFor="new-user-name-input-error" className="text-right col-span-1">
                       New Name
                     </Label>
                     <Input
                       id="new-user-name-input-error" // Use different id
                       value={dialogLabelInput}
                       onChange={(e) => setDialogLabelInput(e.target.value)}
                       placeholder="Enter new user name"
                       className="col-span-3"
                       disabled={isSubmitting}
                     />
                   </div>
                 </> // Close Fragment
              ) : (
                // Render Tabs if loading is complete and no error
                <Tabs value={userNameMode} onValueChange={(value) => {
                  setUserNameMode(value as 'new' | 'existing');
                  setDialogLabelInput(''); // Clear input when switching tabs
                }}>
                  <TabsList className="grid w-full grid-cols-2">
                    <TabsTrigger value="new">New User</TabsTrigger>
                    <TabsTrigger value="existing" disabled={existingLabels.length === 0}>Existing User</TabsTrigger>
                  </TabsList>
                  <TabsContent value="new" className="mt-4">
                    <div className="grid grid-cols-4 items-center gap-4">
                      <Label htmlFor="new-user-name-input" className="text-right col-span-1">
                        New Name
                      </Label>
                      <Input
                        id="new-user-name-input"
                        value={dialogLabelInput}
                        onChange={(e) => setDialogLabelInput(e.target.value)}
                        placeholder="Enter new user name"
                        className="col-span-3"
                        disabled={isSubmitting}
                      />
                    </div>
                  </TabsContent>
                  <TabsContent value="existing" className="mt-4">
                    <div className="grid grid-cols-4 items-center gap-4">
                      <Label htmlFor="existing-user-select" className="text-right col-span-1">
                        Select Name
                      </Label>
                      <Select
                        value={dialogLabelInput}
                        onValueChange={handleSelectChange} // Use handler to update state
                        disabled={isSubmitting}
                      >
                        <SelectTrigger id="existing-user-select" className="col-span-3">
                          <SelectValue placeholder="Select existing user..." />
                        </SelectTrigger>
                        <SelectContent>
                          {existingLabels.map((item) => (
                            <SelectItem key={item.label} value={item.label}>
                              {item.label} ({item.count} {item.count === 1 ? 'entry' : 'entries'})
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                  </TabsContent>
                </Tabs>
              )}
            </div>
            <DialogFooter>
              <DialogClose asChild>
                <Button type="button" variant="outline" disabled={isSubmitting}>
                  Cancel
                </Button>
              </DialogClose>
              <Button
                type="button"
                onClick={handleDialogConfirm}
                // Disable confirm if no input, or submitting, or still loading initial labels
                disabled={!dialogLabelInput.trim() || isSubmitting || isLoadingLabels}
              >
                {isSubmitting ? <><Loader2 className="mr-2 h-4 w-4 animate-spin" /> Confirming...</> : 'Confirm & Register'}
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </CardContent>
    </Card>
  );
};

export default RegisterPage;