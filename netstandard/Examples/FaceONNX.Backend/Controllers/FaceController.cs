using FaceONNX.Backend.Models;
using FaceONNX.Backend.Services;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Logging;
using SixLabors.ImageSharp; // For Image.Load
using SixLabors.ImageSharp.PixelFormats; // For Rgb24
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;

namespace FaceONNX.Backend.Controllers
{
    [ApiController]
    [Route("api")]
    public class FaceController : ControllerBase
    {
        private readonly ConfigurationService _configService;
        private readonly PersistenceService _persistenceService;
        private readonly FaceProcessingService _faceProcessingService;
        private readonly ILogger<FaceController> _logger;

        // In-memory cache of the gallery data. Loaded once at startup.
        private static Dictionary<string, List<GalleryEntry>> _registeredEmbeddings = new Dictionary<string, List<GalleryEntry>>();
        private static bool _isGalleryLoaded = false;
        private static readonly object _loadLock = new object(); // Lock for initial load

        public FaceController(
            ConfigurationService configService,
            PersistenceService persistenceService,
            FaceProcessingService faceProcessingService,
            ILogger<FaceController> logger)
        {
            _configService = configService;
            _persistenceService = persistenceService;
            _faceProcessingService = faceProcessingService;
            _logger = logger;

            // Load gallery data on first request (or consider moving to startup)
            EnsureGalleryLoaded();
        }

        // Helper to load gallery data once
        private void EnsureGalleryLoaded()
        {
            if (!_isGalleryLoaded)
            {
                lock (_loadLock)
                {
                    if (!_isGalleryLoaded) // Double-check lock
                    {
                        _logger.LogInformation("Loading initial gallery data...");
                        try
                        {
                            // Use .Result here as it's part of the constructor/initialization path
                            // Consider async initialization pattern if this becomes complex
                            _registeredEmbeddings = _persistenceService.LoadEmbeddingsAsync().Result;
                            _isGalleryLoaded = true;
                            _logger.LogInformation($"Loaded {_registeredEmbeddings.Sum(kv => kv.Value.Count)} embeddings for {_registeredEmbeddings.Count} labels.");
                        }
                        catch (Exception ex)
                        {
                            _logger.LogError(ex, "Failed to load initial gallery data.");
                            // Depending on requirements, either continue with empty data or throw
                            // Throwing might be safer to indicate a critical startup failure.
                            throw new ApplicationException("Failed to load gallery data on startup.", ex);
                        }
                    }
                }
            }
        }

        // POST: api/Face/register
        [HttpPost("register")]
        [ProducesResponseType(StatusCodes.Status200OK)]
        [ProducesResponseType(StatusCodes.Status400BadRequest)]
        [ProducesResponseType(StatusCodes.Status500InternalServerError)]
        public async Task<IActionResult> RegisterFace([FromForm] string label, [FromForm] IFormFile file)
        {
            if (string.IsNullOrWhiteSpace(label))
            {
                return BadRequest(new { message = "Label cannot be empty." });
            }
            if (file == null || file.Length == 0)
            {
                return BadRequest(new { message = "Image file is required." });
            }

            string? savedFilename = null;
            string? imageSavePath = null;

            try
            {
                // 1. Save uploaded image
                var (uniqueFilename, imageBytes) = await Utils.SaveUploadedImageAsync(file, label, _configService.GalleryDirectory);
                savedFilename = uniqueFilename;
                imageSavePath = Path.Combine(_configService.GalleryDirectory, savedFilename);
                _logger.LogInformation($"Saved uploaded image as: {savedFilename}");

                // 2. Decode image
                using var image = Utils.DecodeImage(imageBytes);
                if (image == null)
                {
                    Utils.CleanUpImage(imageSavePath); // Clean up saved file
                    return BadRequest(new { message = "Could not decode uploaded image." });
                }

                // 3. Process image for embedding (needs alignment, so landmarks are extracted internally)
                var processingResults = _faceProcessingService.ProcessImage(image, extractLandmarks: false); // Landmarks needed for alignment internally

                // Find the first valid embedding
                float[]? embedding = null;
                foreach (var result in processingResults)
                {
                    if (result.Embedding != null && result.Embedding.Length > 0) // Check if embedding is valid
                    {
                        embedding = result.Embedding;
                        _logger.LogInformation($"Using embedding from face @ {result.DetectionBox} for registration.");
                        break;
                    }
                }

                if (embedding == null)
                {
                    Utils.CleanUpImage(imageSavePath); // Clean up saved file
                    _logger.LogWarning($"No valid embedding found for image {savedFilename}.");
                    return BadRequest(new { message = "Could not detect face or extract embedding from the image." });
                }

                // 4. Add to registered embeddings (thread-safe update needed if loaded lazily)
                var newEntry = new GalleryEntry { Embedding = embedding, ImageFilename = savedFilename };

                bool saveNeeded = false;
                lock (_loadLock) // Use the same lock for modifying the shared dictionary
                {
                    if (_registeredEmbeddings.TryGetValue(label, out var entriesList))
                    {
                        entriesList.Add(newEntry);
                    }
                    else
                    {
                        _registeredEmbeddings[label] = new List<GalleryEntry> { newEntry };
                    }
                    _logger.LogInformation($"Registered '{label}'. Total embeddings for label: {_registeredEmbeddings[label].Count}");
                    saveNeeded = true; // Mark that a save is needed after releasing the lock
                }

                // 5. Save updated embeddings to file (outside the lock)
                if (saveNeeded)
                {
                    try
                    {
                        await _persistenceService.SaveEmbeddingsAsync(_registeredEmbeddings);
                        _logger.LogInformation("Successfully saved updated embeddings.");
                    }
                    catch (Exception saveEx)
                    {
                        // Log the save error, but the registration might still be considered successful in memory
                        _logger.LogError(saveEx, "Error saving embeddings after registration.");
                        // Optionally return a specific status or message indicating save failure
                    }
                }


                return Ok(new { message = $"Face for '{label}' registered successfully.", filename = savedFilename });
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error during registration.");
                // Clean up image if saving succeeded but processing failed later
                if (imageSavePath != null)
                {
                    Utils.CleanUpImage(imageSavePath);
                }
                return StatusCode(StatusCodes.Status500InternalServerError, new { message = $"An internal error occurred: {ex.Message}" });
            }
        }


        // POST: api/Face/recognize
        [HttpPost("recognize")]
        [ProducesResponseType(typeof(RecognitionResponse), StatusCodes.Status200OK)]
        [ProducesResponseType(StatusCodes.Status400BadRequest)]
        [ProducesResponseType(StatusCodes.Status500InternalServerError)]
        public async Task<IActionResult> RecognizeFace([FromForm] IFormFile file,
                                                     [FromQuery] bool extractLandmarks = false,
                                                     [FromQuery] bool checkLiveness = true) // Added checkLiveness parameter
        {
            extractLandmarks = true;
            if (file == null || file.Length == 0)
            {
                return BadRequest(new { message = "Image file is required." });
            }

            if (!_registeredEmbeddings.Any())
            {
                return BadRequest(new { message = "No faces registered yet. Cannot perform recognition." });
            }

            try
            {
                byte[] imageBytes;
                using (var memoryStream = new MemoryStream())
                {
                    await file.CopyToAsync(memoryStream);
                    imageBytes = memoryStream.ToArray();
                }

                using var image = Utils.DecodeImage(imageBytes);
                if (image == null)
                {
                    return BadRequest(new { message = "Could not decode query image." });
                }

                // Process query image
                var queryResults = _faceProcessingService.ProcessImage(image, extractLandmarks, checkLiveness); // Pass checkLiveness
                if (!queryResults.Any(r => r.Embedding != null)) // Check if any face yielded an embedding
                {
                    _logger.LogWarning("No faces with valid embeddings detected in the query image.");
                    // Return unknown but potentially with landmarks if requested and detected
                    var firstResult = queryResults.FirstOrDefault();
                    return Ok(new RecognitionResponse
                    {
                        Label = "unknown",
                        Similarity = -1f,
                        QueryLandmarks5pt = firstResult?.Landmarks5pt,
                        QueryLandmarks68pt = extractLandmarks ? firstResult?.Landmarks68pt : null,
                        LivenessScore = firstResult?.LivenessScore ?? -1f, // Add liveness info even if no embedding
                        IsLive = firstResult?.IsLive ?? false
                    });
                }

                // --- Find Best Match ---
                string overallBestMatchLabel = "unknown";
                float overallHighestSimilarity = -1.0f;
                GalleryEntry? overallBestMatchingEntry = null;
                FaceProcessingResult? bestQueryFaceResult = null;

                foreach (var queryFace in queryResults)
                {
                    // Skip faces without embeddings OR faces that failed the liveness check
                    if (queryFace.Embedding == null || !queryFace.IsLive) continue;

                    string currentFaceBestLabel = "unknown";
                    float currentFaceHighestSimilarity = -1.0f;
                    GalleryEntry? currentFaceBestEntry = null;

                    // Compare this query face against all registered entries
                    foreach (var kvp in _registeredEmbeddings)
                    {
                        var label = kvp.Key;
                        var entriesList = kvp.Value;

                        foreach (var registeredEntry in entriesList)
                        {
                            if (registeredEntry.Embedding == null) continue;

                            float similarity = Utils.CosineSimilarity(queryFace.Embedding, registeredEntry.Embedding);

                            if (similarity > currentFaceHighestSimilarity)
                            {
                                currentFaceHighestSimilarity = similarity;
                                currentFaceBestLabel = label;
                                currentFaceBestEntry = registeredEntry;
                            }
                        }
                    }

                    // Check if this query face's best match is better than the overall best found so far
                    if (currentFaceHighestSimilarity > overallHighestSimilarity)
                    {
                        overallHighestSimilarity = currentFaceHighestSimilarity;
                        overallBestMatchLabel = currentFaceBestLabel;
                        overallBestMatchingEntry = currentFaceBestEntry;
                        bestQueryFaceResult = queryFace; // Store the query face that gave this best match
                    }
                }

                _logger.LogInformation($"Recognition attempt: Best match '{overallBestMatchLabel}' with similarity {overallHighestSimilarity:F4}. Landmarks requested: {extractLandmarks}");

                // Apply threshold
                if (overallHighestSimilarity < _configService.SimilarityThreshold)
                {
                    overallBestMatchLabel = "unknown";
                    overallBestMatchingEntry = null; // No match above threshold
                }

                // Prepare response
                var response = new RecognitionResponse
                {
                    Label = overallBestMatchLabel,
                    Similarity = overallHighestSimilarity,
                    QueryEmbedding = bestQueryFaceResult?.Embedding, // Embedding of the best matching query face
                    MatchedEmbedding = overallBestMatchingEntry?.Embedding,
                    MatchedImageFilename = overallBestMatchingEntry?.ImageFilename,
                    QueryLandmarks5pt = bestQueryFaceResult?.Landmarks5pt,
                    QueryLandmarks68pt = extractLandmarks ? bestQueryFaceResult?.Landmarks68pt : null,
                    LivenessScore = bestQueryFaceResult?.LivenessScore ?? -1f, // Liveness info from the best matching query face
                    IsLive = bestQueryFaceResult?.IsLive ?? false
                };

                return Ok(response);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error during recognition.");
                return StatusCode(StatusCodes.Status500InternalServerError, new { message = $"An internal error occurred: {ex.Message}" });
            }
        }

        // GET: api/Face/registered
        [HttpGet("registered")]
        [ProducesResponseType(typeof(RegisteredLabelsResponse), StatusCodes.Status200OK)]
        public IActionResult GetRegisteredLabels()
        {
            var response = new RegisteredLabelsResponse
            {
                Labels = _registeredEmbeddings.Keys.ToList()
            };
            return Ok(response);
        }

        // GET: api/Face/gallery_data
        [HttpGet("gallery_data")]
        [ProducesResponseType(typeof(GalleryDataResponse), StatusCodes.Status200OK)]
        public IActionResult GetGalleryData()
        {
            // Return a copy to avoid external modification? For now, return direct reference.
            var response = new GalleryDataResponse
            {
                Data = _registeredEmbeddings
            };
            return Ok(response);
        }


        // DELETE: api/Face/delete_entry?label=...&amp;filename=...
        [HttpDelete("delete_entry")]
        [ProducesResponseType(StatusCodes.Status200OK)]
        [ProducesResponseType(StatusCodes.Status404NotFound)]
        [ProducesResponseType(StatusCodes.Status500InternalServerError)]
        public async Task<IActionResult> DeleteEntry([FromQuery] string label, [FromQuery] string filename)
        {
            if (string.IsNullOrWhiteSpace(label) || string.IsNullOrWhiteSpace(filename))
            {
                return BadRequest(new { message = "Label and filename are required." });
            }

            string imagePath = Path.Combine(_configService.GalleryDirectory, filename);
            bool entryRemoved = false;

            try
            {
                bool saveNeeded = false; // Flag to indicate if save is required
                lock (_loadLock) // Lock for modification
                {
                    if (!_registeredEmbeddings.TryGetValue(label, out var entriesList))
                    {
                        return NotFound(new { message = $"Label '{label}' not found." });
                    }

                    int entryIndex = entriesList.FindIndex(e => e.ImageFilename.Equals(filename, StringComparison.OrdinalIgnoreCase));

                    if (entryIndex == -1)
                    {
                        return NotFound(new { message = $"Entry with filename '{filename}' not found for label '{label}'." });
                    }

                    // Remove from in-memory list
                    entriesList.RemoveAt(entryIndex);
                    entryRemoved = true;
                    _logger.LogInformation($"Removed entry {filename} for label {label} from memory.");

                    // If list is empty, remove the label key
                    if (!entriesList.Any())
                    {
                        _registeredEmbeddings.Remove(label);
                        _logger.LogInformation($"Removed empty label '{label}' from memory.");
                    }
                    saveNeeded = true; // Mark that save is needed after lock release
                } // End lock

                // Save changes asynchronously (outside the lock)
                if (saveNeeded)
                {
                    try
                    {
                        await _persistenceService.SaveEmbeddingsAsync(_registeredEmbeddings);
                        _logger.LogInformation("Successfully saved updated embeddings after deletion.");
                    }
                    catch (Exception saveEx)
                    {
                        _logger.LogError(saveEx, "Error saving embeddings after deletion.");
                        // Decide if this should cause the request to fail
                    }
                }


                // Delete the image file (outside the lock)
                if (entryRemoved) // Only delete file if entry was successfully removed from memory
                {
                    Utils.CleanUpImage(imagePath); // Uses its own error handling
                }

                return Ok(new { message = $"Successfully deleted entry: label='{label}', filename='{filename}'" });
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, $"Error deleting entry: label='{label}', filename='{filename}'.");
                // Consider reloading embeddings from file if save failed to ensure consistency?
                return StatusCode(StatusCodes.Status500InternalServerError, new { message = $"An internal error occurred during deletion: {ex.Message}" });
            }
        }


        // Removed GetImage action - Static files are served by UseStaticFiles middleware in Program.cs
    }
}