using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing; // For Mutate/Resize if needed later
using System;
using System.IO;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Http; // For IFormFile

namespace FaceONNX.Backend.Services
{
    /// <summary>
    /// Provides utility methods for image processing and file handling.
    /// </summary>
    public static class Utils
    {
        /// <summary>
        /// Converts an ImageSharp Image<Rgb24> to the float[][,] format required by FaceONNX.
        /// Normalizes pixel values to the range [0, 1].
        /// Assumes BGR channel order for FaceONNX models.
        /// </summary>
        /// <param name="image">The input image.</param>
        /// <returns>A float array representing the image [channels][height, width].</returns>
        public static float[][,] ImageToMultiDimArray(Image<Rgb24> image)
        {
            var array = new[]
            {
                new float[image.Height, image.Width], // B channel
                new float[image.Height, image.Width], // G channel
                new float[image.Height, image.Width]  // R channel
            };

            image.ProcessPixelRows(pixelAccessor =>
            {
                for (var y = 0; y < pixelAccessor.Height; y++)
                {
                    var row = pixelAccessor.GetRowSpan(y);
                    for (var x = 0; x < pixelAccessor.Width; x++)
                    {
                        // Normalize to [0, 1] and assign to BGR channels
                        array[2][y, x] = row[x].R / 255.0F; // R -> index 2
                        array[1][y, x] = row[x].G / 255.0F; // G -> index 1
                        array[0][y, x] = row[x].B / 255.0F; // B -> index 0
                    }
                }
            });

            return array;
        }

        /// <summary>
        /// Decodes an image from a byte array.
        /// </summary>
        /// <param name="imageBytes">The byte array containing image data.</param>
        /// <returns>An Image<Rgb24> object or null if decoding fails.</returns>
        public static Image<Rgb24>? DecodeImage(byte[] imageBytes)
        {
            try
            {
                // Load the image and ensure it's in Rgb24 format
                return Image.Load<Rgb24>(imageBytes);
            }
            catch (Exception ex) // Catch specific ImageSharp exceptions if needed
            {
                Console.Error.WriteLine($"Error decoding image: {ex.Message}");
                return null;
            }
        }

        /// <summary>
        /// Saves an uploaded file (IFormFile) to the specified directory with a unique name.
        /// Generates a unique filename based on label, original filename, and timestamp.
        /// </summary>
        /// <param name="file">The uploaded file.</param>
        /// <param name="label">The label associated with the file.</param>
        /// <param name="directory">The directory to save the file in.</param>
        /// <returns>A tuple containing the unique filename and the file content as byte array.</returns>
        public static async Task<(string uniqueFilename, byte[] fileBytes)> SaveUploadedImageAsync(IFormFile file, string label, string directory)
        {
            if (file == null || file.Length == 0)
            {
                throw new ArgumentException("File is empty or null.", nameof(file));
            }

            // Sanitize label and filename to prevent path traversal issues
            var safeLabel = SanitizeFilename(label);
            var safeOriginalFilename = SanitizeFilename(Path.GetFileNameWithoutExtension(file.FileName));
            var extension = Path.GetExtension(file.FileName).ToLowerInvariant(); // Keep original extension

            // Create a unique filename
            var timestamp = DateTime.UtcNow.ToString("yyyyMMddHHmmssfff");
            var uniqueFilename = $"{safeLabel}_{safeOriginalFilename}_{timestamp}{extension}";
            var filePath = Path.Combine(directory, uniqueFilename);

            // Ensure directory exists
            Directory.CreateDirectory(directory);

            byte[] fileBytes;
            using (var memoryStream = new MemoryStream())
            {
                await file.CopyToAsync(memoryStream);
                fileBytes = memoryStream.ToArray();
            }

            await File.WriteAllBytesAsync(filePath, fileBytes);

            return (uniqueFilename, fileBytes);
        }

        /// <summary>
        /// Deletes a file if it exists. Logs errors if deletion fails.
        /// </summary>
        /// <param name="filePath">The full path to the file.</param>
        public static void CleanUpImage(string filePath)
        {
            try
            {
                if (File.Exists(filePath))
                {
                    File.Delete(filePath);
                    Console.WriteLine($"Deleted file: {filePath}");
                }
            }
            catch (IOException ex)
            {
                Console.Error.WriteLine($"Error deleting file {filePath}: {ex.Message}");
                // Decide if you need to re-throw or just log
            }
        }

        /// <summary>
        /// Removes invalid characters from a filename or label.
        /// </summary>
        private static string SanitizeFilename(string name)
        {
            var invalidChars = Path.GetInvalidFileNameChars();
            var sanitized = new StringBuilder(name.Length);
            foreach (char c in name)
            {
                if (!invalidChars.Contains(c))
                {
                    sanitized.Append(c);
                }
                else
                {
                    sanitized.Append('_'); // Replace invalid chars with underscore
                }
            }
            // Replace potential directory separators as well
            sanitized.Replace(Path.DirectorySeparatorChar, '_');
            sanitized.Replace(Path.AltDirectorySeparatorChar, '_');

            return sanitized.ToString().Trim().TrimStart('.'); // Basic trimming
        }

        /// <summary>
        /// Calculates the cosine similarity between two vectors (embeddings).
        /// </summary>
        public static float CosineSimilarity(float[] vector1, float[] vector2)
        {
            if (vector1 == null || vector2 == null || vector1.Length != vector2.Length || vector1.Length == 0)
            {
                // Handle invalid input, e.g., return 0 or throw exception
                return 0.0f;
            }

            float dotProduct = 0.0f;
            float magnitude1 = 0.0f;
            float magnitude2 = 0.0f;

            for (int i = 0; i < vector1.Length; i++)
            {
                dotProduct += vector1[i] * vector2[i];
                magnitude1 += vector1[i] * vector1[i];
                magnitude2 += vector2[i] * vector2[i];
            }

            magnitude1 = (float)Math.Sqrt(magnitude1);
            magnitude2 = (float)Math.Sqrt(magnitude2);

            if (magnitude1 == 0.0f || magnitude2 == 0.0f)
            {
                // Handle zero magnitude vectors
                return 0.0f;
            }

            return dotProduct / (magnitude1 * magnitude2);
        }
    }
}