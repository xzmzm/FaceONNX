using System.Text.Json.Serialization;

namespace FaceONNX.Backend.Models
{
    /// <summary>
    /// Represents a single registered face entry containing the embedding and image filename.
    /// </summary>
    public class GalleryEntry
    {
        // Initialize non-nullable properties and add explicit JSON names
        [JsonPropertyName("embedding")]
        public float[] Embedding { get; set; } = Array.Empty<float>();

        // Make filename nullable to better represent potential absence (though expected after registration)
        [JsonPropertyName("image_filename")]
        public string? ImageFilename { get; set; } // Removed default initializer "= string.Empty;"
    }
}