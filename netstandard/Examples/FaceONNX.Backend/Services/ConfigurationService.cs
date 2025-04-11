using System.IO;
using System.Reflection;

namespace FaceONNX.Backend.Services
{
    /// <summary>
    /// Provides configuration settings for the backend application.
    /// </summary>
    public class ConfigurationService
    {
        // --- Paths ---
        // Base directory of the current assembly (FaceONNX.Backend)
        private static readonly string _baseDirectory = AppContext.BaseDirectory ?? Directory.GetCurrentDirectory();

        // Gallery directory relative to the backend project's output directory
        public string WebAppDirectory { get; } = Path.Combine(_baseDirectory, "webapp");
        // Gallery directory relative to the backend project's output directory
        public string GalleryDirectory { get; } = Path.Combine(_baseDirectory, "GalleryData");

        // Embeddings file path
        public string EmbeddingsFilePath { get; } = Path.Combine(_baseDirectory, "GalleryData", "embeddings.json");

        // Model paths (relative to the main FaceONNX project structure)
        // Adjust these paths if your model locations differ.
        private static readonly string _modelsBaseDirectory = Path.GetFullPath(Path.Combine(_baseDirectory, "..", "..", "..", "FaceONNX.Models", "models"));

        // --- Settings ---
        public float SimilarityThreshold { get; } = 0.3f; // Threshold for considering a match

        /// <summary>
        /// Ensures the gallery directory exists. Creates it if it doesn't.
        /// </summary>
        public void EnsureGalleryDirectoryExists()
        {
            Directory.CreateDirectory(GalleryDirectory);
        }
    }
}