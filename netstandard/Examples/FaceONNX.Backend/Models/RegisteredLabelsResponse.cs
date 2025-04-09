using System.Collections.Generic;

namespace FaceONNX.Backend.Models
{
    /// <summary>
    /// Represents the response containing a list of registered labels.
    /// </summary>
    public class RegisteredLabelsResponse
    {
        // Initialize non-nullable property
        public List<string> Labels { get; set; } = new List<string>();
    }
}