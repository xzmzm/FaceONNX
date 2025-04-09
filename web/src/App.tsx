import { Routes, Route, Link, Navigate } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import RegisterPage from '@/pages/RegisterPage'; // Use alias
import RecognizePage from '@/pages/RecognizePage'; // Use alias
import GalleryPage from '@/pages/GalleryPage'; // Use alias
// import './App.css'; // Removed as it might be missing and we use Tailwind/shadcn

function App() {
  return (
    <div className="min-h-screen flex flex-col items-center p-4 bg-background text-foreground">
      <header className="w-full max-w-4xl mb-8">
        <h1 className="text-3xl font-bold text-center mb-4">Elsoft Face Recognition</h1>
        <nav className="flex justify-center space-x-4">
          <Button variant="link" asChild>
            <Link to="/register">Register</Link>
          </Button>
          <Button variant="link" asChild>
            <Link to="/recognize">Recognize</Link>
          </Button>
          <Button variant="link" asChild>
            <Link to="/gallery">Gallery</Link>
          </Button>
        </nav>
      </header>

      <main className="w-full max-w-4xl">
        <Routes>
          <Route path="/" element={<Navigate to="/register" replace />} />
          <Route path="/register" element={<RegisterPage />} />
          <Route path="/recognize" element={<RecognizePage />} />
          <Route path="/gallery" element={<GalleryPage />} />
          {/* Add a fallback route if needed */}
          {/* <Route path="*" element={<div>Page Not Found</div>} /> */}
        </Routes>
      </main>
    </div>
  );
}

export default App;
