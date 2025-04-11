using FaceONNX.Backend.Models;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;

namespace FaceONNX.Backend.Services
{
    /// <summary>
    /// Handles loading and saving registered face embeddings to a JSON file.
    /// Implements basic file locking to prevent race conditions during concurrent writes.
    /// </summary>
    public class PersistenceService
    {
        private readonly string _filePath;
        // Removed _jsonOptions, options are now in PersistenceJsonContext
        private static readonly SemaphoreSlim _fileLock = new SemaphoreSlim(1, 1); // Semaphore for file access

        public PersistenceService(ConfigurationService config)
        {
            _filePath = config.EmbeddingsFilePath;
            // _jsonOptions initialization removed

            // Ensure the directory exists on initialization
            config.EnsureGalleryDirectoryExists();
        }

        /// <summary>
        /// Loads the gallery data from the JSON file.
        /// Returns an empty dictionary if the file doesn't exist or is empty/invalid.
        /// </summary>
        public async Task<Dictionary<string, List<GalleryEntry>>> LoadEmbeddingsAsync()
        {
            await _fileLock.WaitAsync(); // Acquire lock
            try
            {
                if (!File.Exists(_filePath))
                {
                    return new Dictionary<string, List<GalleryEntry>>();
                }

                var json = await File.ReadAllTextAsync(_filePath);
                if (string.IsNullOrWhiteSpace(json))
                {
                    return new Dictionary<string, List<GalleryEntry>>();
                }

                try
                {
                    // Use source generator context for deserialization
                    var data = JsonSerializer.Deserialize(json, PersistenceJsonContext.Default.DictionaryStringListGalleryEntry);
                    return data ?? new Dictionary<string, List<GalleryEntry>>();
                }
                catch (JsonException ex)
                {
                    // Log the error or handle corrupted file scenario
                    Console.Error.WriteLine($"Error deserializing embeddings file '{_filePath}': {ex.Message}");
                    // Return empty or throw, depending on desired behavior for corrupted data
                    return new Dictionary<string, List<GalleryEntry>>();
                }
            }
            finally
            {
                _fileLock.Release(); // Release lock
            }
        }

        /// <summary>
        /// Saves the gallery data to the JSON file.
        /// Overwrites the existing file.
        /// </summary>
        public async Task SaveEmbeddingsAsync(Dictionary<string, List<GalleryEntry>> embeddings)
        {
            await _fileLock.WaitAsync(); // Acquire lock
            try
            {
                // Use source generator context for serialization
                var json = JsonSerializer.Serialize(embeddings, PersistenceJsonContext.Default.DictionaryStringListGalleryEntry);
                await File.WriteAllTextAsync(_filePath, json);
            }
            finally
            {
                _fileLock.Release(); // Release lock
            }
        }
    }
}