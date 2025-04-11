using System.Text.Json.Serialization;

namespace FaceONNX.Backend.Models
{
    /// <summary>
    /// Represents a generic response containing a message string.
    /// Used for simple success, error, or informational responses.
    /// </summary>
    public class MessageResponse
    {
        [JsonPropertyName("message")]
        public string Message { get; set; } = string.Empty;
    }
}