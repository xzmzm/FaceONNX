using System.Collections.Generic;
using System.Text.Json.Serialization;
using FaceONNX.Backend.Models;

namespace FaceONNX.Backend.Services;

/// <summary>
/// Source generator context for JSON serialization in PersistenceService.
/// Improves performance and AOT compatibility.
/// </summary>
// Configure source generator options to match previous behavior
[JsonSourceGenerationOptions(WriteIndented = true, PropertyNameCaseInsensitive = true)]
// Add JsonSerializable attributes for all root types used in serialization/deserialization
[JsonSerializable(typeof(Dictionary<string, List<GalleryEntry>>))] // Used by PersistenceService
[JsonSerializable(typeof(GalleryDataResponse))] // Response for GET /api/gallery_data
[JsonSerializable(typeof(RegisteredLabelsResponse))] // Response for GET /api/registered
[JsonSerializable(typeof(RecognitionResponse))] // Response for POST /api/recognize
[JsonSerializable(typeof(RegisterResponse))] // Response for POST /api/register
[JsonSerializable(typeof(MessageResponse))] // Generic message response (used by Delete, NotFound, BadRequest etc.)
// Removed RegisterFaceRequest as it's no longer used/exists
// While the generator often infers nested types, explicitly adding them can sometimes help.
// [JsonSerializable(typeof(List<GalleryEntry>))] // Usually inferred
// [JsonSerializable(typeof(GalleryEntry))]      // Usually inferred
// [JsonSerializable(typeof(float[]))]         // Usually inferred
internal partial class PersistenceJsonContext : JsonSerializerContext
{
}