using System.Collections.Generic;

namespace FaceONNX.Backend.Models
{
    /// <summary>
    /// Represents the response containing the complete gallery data.
    /// The key is the label, and the value is a list of GalleryEntry objects for that label.
    /// </summary>
    public class GalleryDataResponse
    {
        // Initialize non-nullable property
        public Dictionary<string, List<GalleryEntry>> Data { get; set; } = new Dictionary<string, List<GalleryEntry>>();
    }
}