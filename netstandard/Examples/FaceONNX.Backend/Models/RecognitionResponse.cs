using System.Drawing; // Keep for PointF if used elsewhere, or remove if not needed
using System.Text.Json.Serialization;

namespace FaceONNX.Backend.Models
{
    /// <summary>
    /// Represents the response from the face recognition endpoint.
    /// </summary>
    public class RecognitionResponse
    {
        [JsonPropertyName("label")]
        public string Label { get; set; } = "unknown";

        [JsonPropertyName("similarity")]
        public float Similarity { get; set; }

        [JsonPropertyName("query_embedding")]
        public float[]? QueryEmbedding { get; set; }

        [JsonPropertyName("matched_embedding")]
        public float[]? MatchedEmbedding { get; set; }

        [JsonPropertyName("matched_image_filename")]
        public string? MatchedImageFilename { get; set; }

        // Change landmark types to array of arrays (float[point_index][x=0, y=1])
        [JsonPropertyName("query_landmarks_5pt")]
        public float[][]? QueryLandmarks5pt { get; set; }

        [JsonPropertyName("query_landmarks_68pt")]
        public float[][]? QueryLandmarks68pt { get; set; }

        [JsonPropertyName("liveness_score")]
        public float LivenessScore { get; set; } // Added

        [JsonPropertyName("is_live")]
        public bool IsLive { get; set; } // Added
    }
}