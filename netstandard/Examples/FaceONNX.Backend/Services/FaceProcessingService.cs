using FaceONNX; // Core FaceONNX library
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using System;
using System.Collections.Generic;
using System.Drawing; // For PointF
using System.Linq;
using Microsoft.Extensions.Logging; // For logging

namespace FaceONNX.Backend.Services
{
    /// <summary>
    /// Service for processing faces using FaceONNX models (detection, landmarks, embedding).
    /// Loads and manages the ONNX models.
    /// </summary>
    public class FaceProcessingService : IDisposable
    {
        private readonly FaceDetector _faceDetector;
        private readonly FaceEmbedder _faceEmbedder;
        private readonly Face68LandmarksExtractor _faceLandmarksExtractor;
        private readonly IFaceLivenessDetector _faceLivenessDetector; // Added
        private readonly ILogger<FaceProcessingService> _logger;
        private const float DefaultLivenessThreshold = 0.5f; // Added - Threshold for liveness check
        private bool _disposed = false;

        public FaceProcessingService(ConfigurationService config,
                                     ILogger<FaceProcessingService> logger,
                                     IFaceLivenessDetector faceLivenessDetector) // Added detector injection
        {
            _logger = logger;
            _faceLivenessDetector = faceLivenessDetector; // Added assignment
            try
            {
                _logger.LogInformation("Loading FaceONNX models...");
                _faceDetector = new FaceDetector(); // Load detection model
                _faceEmbedder = new FaceEmbedder(); // Load embedding model
                _faceLandmarksExtractor = new Face68LandmarksExtractor(); // Load landmark model
                // Liveness detector is injected, assuming it's loaded via DI
                _logger.LogInformation("FaceONNX models loaded successfully.");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Fatal Error: Could not load FaceONNX models.");
                // Re-throw to prevent the application from starting in an invalid state
                throw new ApplicationException("Failed to initialize FaceProcessingService due to model loading errors.", ex);
            }
        }

        /// <summary>
        /// Processes an image to detect faces, extract embeddings, and optionally landmarks.
        /// </summary>
        /// <param name="image">The input image.</param>
        /// <param name="extractLandmarks">Whether to extract 68-point landmarks.</param>
        /// <returns>A list of results for each detected face.</returns>
        public List<FaceProcessingResult> ProcessImage(Image<Rgb24> image, bool extractLandmarks = false, bool performLivenessCheck = true) // Added performLivenessCheck parameter
        {
            var results = new List<FaceProcessingResult>();
            var imageArray = Utils.ImageToMultiDimArray(image); // Use helper method

            // 1. Detect Faces
            var detectedFaces = _faceDetector.Forward(imageArray);
            _logger.LogInformation($"Detected {detectedFaces.Length} faces.");

            foreach (var detectedFace in detectedFaces)
            {
                var result = new FaceProcessingResult { DetectionBox = detectedFace.Box };

                if (!detectedFace.Box.IsEmpty)
                {
                    try
                    {
                        // 2. Extract Landmarks (always needed for alignment)
                        var landmarks = _faceLandmarksExtractor.Forward(imageArray, detectedFace.Box);
                        var angle = landmarks.RotationAngle; // Get rotation angle for alignment

                        // Store 5-point landmarks if available (often part of detection)
                        result.Landmarks5pt = ConvertToFloatArray(detectedFace.Points.All, 0, 0); // Pass 0 for offsets

                        if (extractLandmarks)
                        {
                            // Add offset (Box.X, Box.Y) to make landmarks relative to the original image
                            result.Landmarks68pt = ConvertToFloatArray(landmarks.All, detectedFace.Box.X, detectedFace.Box.Y);
                        }

                        // 3. Align Face
                        var alignedFace = FaceProcessingExtensions.Align(imageArray, detectedFace.Box, angle);

                        // 4. Perform Liveness Check (Optional, on Aligned Face)
                        if (performLivenessCheck)
                        {
                            result.LivenessScore = _faceLivenessDetector.Forward(alignedFace); // Check on aligned face
                            result.IsLive = result.LivenessScore >= DefaultLivenessThreshold;
                            _logger.LogInformation($"Face at {detectedFace.Box}: Liveness Score = {result.LivenessScore}, IsLive = {result.IsLive}");
                        }
                        else
                        {
                            result.LivenessScore = -1f; // Default score if check is skipped
                            result.IsLive = true; // Assume live if check is skipped
                            _logger.LogInformation($"Face at {detectedFace.Box}: Liveness check skipped.");
                        }


                        // 5. Extract Embedding from Aligned Face (only if live or check skipped)
                        result.Embedding = _faceEmbedder.Forward(alignedFace);
                    }
                    catch (Exception ex)
                    {
                        _logger.LogError(ex, $"Error processing face at box {detectedFace.Box}. Skipping embedding/landmarks for this face.");
                        // Continue processing other faces, but this one won't have embedding/landmarks
                        result.Embedding = null;
                        result.Landmarks68pt = null;
                    }
                }
                else
                {
                    _logger.LogWarning("Detected face with empty bounding box. Skipping processing for this face.");
                }

                results.Add(result);
            }

            return results;
        }

        // Helper to convert FaceONNX Point[] to float[point_index][x=0, y=1]
        private float[][]? ConvertToFloatArray(System.Drawing.Point[]? points, int offsetX = 0, int offsetY = 0)
        {
            if (points == null) return null;
            // Create jagged array: float[number_of_points][2 for x,y], adding the offset
            return points.Select(p => new float[] { p.X + offsetX, p.Y + offsetY }).ToArray();
        }


        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    // Dispose managed resources
                    _faceDetector?.Dispose();
                    _faceEmbedder?.Dispose();
                    _faceLandmarksExtractor?.Dispose();
                    _faceLivenessDetector?.Dispose(); // Added
                    _logger.LogInformation("FaceProcessingService disposed.");
                }
                // Dispose unmanaged resources if any

                _disposed = true;
            }
        }

        ~FaceProcessingService()
        {
            Dispose(false);
        }
    }

    /// <summary>
    /// Helper class to hold the results of processing a single face.
    /// </summary>
    public class FaceProcessingResult
    {
        public System.Drawing.Rectangle DetectionBox { get; set; }
        public float[]? Embedding { get; set; }
        public float[][]? Landmarks5pt { get; set; } // Basic landmarks from detector
        public float[][]? Landmarks68pt { get; set; } // Detailed landmarks (if requested and live)
        public float LivenessScore { get; set; } // Added
        public bool IsLive { get; set; } // Added
    }
}