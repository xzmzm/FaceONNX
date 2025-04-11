using System.Collections.Generic;
using System.Text.Json.Serialization; // Add this using
namespace FaceONNX.Backend.Models
{
    /// <summary>
    /// Represents the response containing the complete gallery data.
    /// The key is the label, and the value is a list of GalleryEntry objects for that label.
    /// </summary>
    public class GalleryDataResponse
    {
        // Initialize non-nullable property and explicitly set JSON name
        [JsonPropertyName("data")] // Force the name to 'data'
        public Dictionary<string, List<GalleryEntry>> Data { get; set; } = new Dictionary<string, List<GalleryEntry>>();
    }
}