using System.Text.Json.Serialization;

namespace FaceONNX.Backend.Models
{
    /// <summary>
    /// Represents the response from a successful face registration.
    /// </summary>
    public class RegisterResponse
    {
        [JsonPropertyName("message")]
        public string Message { get; set; } = string.Empty;

        [JsonPropertyName("filename")]
        public string? Filename { get; set; } // Filename can be null if something went wrong before saving
    }
}