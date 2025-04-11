using FaceONNX; // Add FaceONNX namespace for detectors/embedders
using FaceONNX.Backend.Services;
using Microsoft.Extensions.FileProviders; // For static files
using System.IO; // For Path
using System.Text.Json; // For JsonNamingPolicy
using FaceONNX.Backend.Models; // Add Models namespace
// using FaceONNX.Backend.Services; // Removed duplicate using
using Microsoft.AspNetCore.Mvc; // For [FromQuery], [FromForm] attributes if needed, and IResult types
using SixLabors.ImageSharp; // For Image.Load
using System.Collections.Concurrent; // For thread-safe dictionary if needed, though lock is used here

var builder = WebApplication.CreateBuilder(args);

// --- Configure Kestrel to listen on port 8000 ---
builder.WebHost.ConfigureKestrel(options =>
{
    options.ListenLocalhost(8000);
});

// --- Configure Services ---

// 1. Add CORS
var AllowSpecificOrigins = "_allowSpecificOrigins";
builder.Services.AddCors(options =>
{
    options.AddPolicy(name: AllowSpecificOrigins,
                      policy =>
                      {
                          policy.WithOrigins("http://localhost:5173", // Vite default dev server
                                             "http://127.0.0.1:5173",
                                             "https://facereg.elsoft.org")
                                .AllowAnyHeader()
                                .AllowAnyMethod();
                          // Add production frontend URL here if needed
                          // .WithOrigins("https://your-frontend-domain.com")
                      });
});

// 2. Register custom services (Singleton for services holding state/resources)
builder.Services.AddSingleton<ConfigurationService>();
builder.Services.AddSingleton<PersistenceService>();
builder.Services.AddSingleton<FaceProcessingService>(); // Loads models, keep as singleton
builder.Services.AddSingleton<IFaceLivenessDetector, FaceLivenessDetector>(); // Add Liveness Detector

// Configure global JSON options to use the source generator context for Minimal APIs
builder.Services.Configure<Microsoft.AspNetCore.Http.Json.JsonOptions>(options =>
{
    options.SerializerOptions.TypeInfoResolver = PersistenceJsonContext.Default;
    // Apply snake_case policy globally if needed (was previously in AddControllers)
    options.SerializerOptions.PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower;
});

// 3. Add Controllers service (REMOVED for Minimal API migration)
// builder.Services.AddControllers()
//     .AddJsonOptions(options =>
//     {
//         // Use snake_case for JSON property names to match Python backend
//         options.JsonSerializerOptions.PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower;
//         // Configure MVC to use the source-generated JSON context for AOT compatibility
//         options.JsonSerializerOptions.TypeInfoResolver = PersistenceJsonContext.Default;
//     });

// 4. Add Swagger/OpenAPI (REMOVED - Minimal APIs use AddEndpointsApiExplorer directly if needed, but keeping commented for now)
// builder.Services.AddEndpointsApiExplorer(); // Re-enable if Swagger/OpenAPI is needed with Minimal APIs

// --- Build the App ---
var app = builder.Build();

// --- Minimal API Data Cache & Loading ---
// In-memory cache (static for simplicity in Program.cs)
var registeredEmbeddings = new Dictionary<string, List<GalleryEntry>>();
var galleryLoadLock = new object(); // Lock for modifications

// Load initial data at startup
var initialPersistenceService = app.Services.GetRequiredService<PersistenceService>();
var initialLogger = app.Services.GetRequiredService<ILogger<Program>>(); // Logger for Program

initialLogger.LogInformation("Loading initial gallery data at startup...");
try
{
    // Load synchronously at startup
    registeredEmbeddings = await initialPersistenceService.LoadEmbeddingsAsync();
    initialLogger.LogInformation($"Loaded {registeredEmbeddings.Sum(kv => kv.Value.Count)} embeddings for {registeredEmbeddings.Count} labels at startup.");
}
catch (Exception ex)
{
    initialLogger.LogError(ex, "Failed to load initial gallery data at startup. Application might not function correctly.");
    // Decide whether to throw or continue with empty data
    // throw; // Uncomment to prevent startup on load failure
}

// --- Configure the HTTP request pipeline ---

// 1. HTTPS Redirection (Keep standard)
app.UseHttpsRedirection();

// 2. Static Files for Gallery Images
// Get the gallery path from the ConfigurationService
var configService = app.Services.GetRequiredService<ConfigurationService>();
var webAppPath = Path.GetFullPath(configService.WebAppDirectory); // Ensure absolute path
var galleryPath = Path.GetFullPath(configService.GalleryDirectory); // Ensure absolute path
if (!Directory.Exists(galleryPath))
{
    Directory.CreateDirectory(galleryPath); // Ensure it exists
    Console.WriteLine($"Created gallery directory: {galleryPath}");
}
Console.WriteLine($"Serving static files from: {galleryPath}");
app.UseStaticFiles(new StaticFileOptions
{
    FileProvider = new PhysicalFileProvider(galleryPath),
    RequestPath = "/api/images" // Serve gallery images from /api/images path
});

app.UseCors(AllowSpecificOrigins);

Console.WriteLine($"Serving frontend static files from: {webAppPath}");

// Serve index.html for root requests
var defaultFilesOptions = new DefaultFilesOptions
{
    FileProvider = new PhysicalFileProvider(webAppPath),
    RequestPath = "" // Serve from root
};
defaultFilesOptions.DefaultFileNames.Clear(); // Use only index.html
defaultFilesOptions.DefaultFileNames.Add("index.html");
app.UseDefaultFiles(defaultFilesOptions);

// Serve other static files (js, css, images) from webapp
app.UseStaticFiles(new StaticFileOptions
{
    FileProvider = new PhysicalFileProvider(webAppPath),
    RequestPath = "" // Serve from root
});

// app.UseAuthorization();

app.UseRouting(); // Explicitly add routing middleware before mapping endpoints

// app.MapControllers(); // REMOVED for Minimal API migration

// --- Minimal API Endpoints ---

// GET /api/registered
app.MapGet("/api/registered", (ILogger<Program> logger) =>
{
    logger.LogInformation("GET /api/registered called");
    // Access the static cache (consider thread safety if writes were frequent, but reads are okay)
    var response = new RegisteredLabelsResponse
    {
        Labels = registeredEmbeddings.Keys.ToList()
    };
    return Results.Ok(response);
})
.Produces<RegisteredLabelsResponse>() // Add OpenAPI metadata if needed later
.WithTags("FaceAPI"); // Group endpoints in Swagger UI if used

// GET /api/gallery_data
app.MapGet("/api/gallery_data", (
    ILogger<Program> logger,
    // Inject the globally configured JsonOptions
    Microsoft.Extensions.Options.IOptions<Microsoft.AspNetCore.Http.Json.JsonOptions> jsonOptionsAccessor
    ) =>
{
    logger.LogInformation("GET /api/gallery_data called");
    // Return a copy or direct reference based on needs. Direct reference used here.
    var response = new GalleryDataResponse
    {
        Data = registeredEmbeddings
    };
    // Return using Results.Ok - global options and [JsonPropertyName] attribute should handle serialization
    return Results.Ok(response);
})
.Produces<GalleryDataResponse>()
.WithTags("FaceAPI");

// POST /api/register
// Note: Minimal APIs handle form binding differently. We access HttpRequest.Form directly.
app.MapPost("/api/register", async (
    HttpRequest httpRequest,
    PersistenceService persistenceService,
    FaceProcessingService faceProcessingService,
    ConfigurationService configService,
    ILogger<Program> logger) =>
{
    logger.LogInformation("POST /api/register called");

    // --- Extract data from form ---
    if (!httpRequest.HasFormContentType)
    {
        return Results.BadRequest(new MessageResponse { Message = "Request must be form data." });
    }

    var form = await httpRequest.ReadFormAsync();
    string? label = form["label"]; // Get label from form
    IFormFile? file = form.Files["file"]; // Get file from form

    if (string.IsNullOrWhiteSpace(label))
    {
        return Results.BadRequest(new MessageResponse { Message = "Label cannot be empty." });
    }
    if (file == null || file.Length == 0)
    {
        return Results.BadRequest(new MessageResponse { Message = "Image file is required." });
    }
    // --- End Extract data ---


    string? savedFilename = null;
    string? imageSavePath = null;

    try
    {
        // 1. Save uploaded image (using request.File and request.Label)
        var (uniqueFilename, imageBytes) = await Utils.SaveUploadedImageAsync(file, label, configService.GalleryDirectory);
        savedFilename = uniqueFilename;
        imageSavePath = Path.Combine(configService.GalleryDirectory, savedFilename);
        logger.LogInformation($"Saved uploaded image as: {savedFilename}");

        // 2. Decode image
        using var image = Utils.DecodeImage(imageBytes);
        if (image == null)
        {
            Utils.CleanUpImage(imageSavePath); // Clean up saved file
            return Results.BadRequest(new MessageResponse { Message = "Could not decode uploaded image." });
        }

        // 3. Process image for embedding
        var processingResults = faceProcessingService.ProcessImage(image, extractLandmarks: false);

        // Find the first valid embedding
        float[]? embedding = processingResults.FirstOrDefault(r => r.Embedding != null && r.Embedding.Length > 0)?.Embedding;

        if (embedding == null)
        {
            Utils.CleanUpImage(imageSavePath); // Clean up saved file
            logger.LogWarning($"No valid embedding found for image {savedFilename}.");
            return Results.BadRequest(new MessageResponse { Message = "Could not detect face or extract embedding from the image." });
        }
        logger.LogInformation($"Using embedding from face for registration.");


        // 4. Add to registered embeddings (thread-safe update)
        var newEntry = new GalleryEntry { Embedding = embedding, ImageFilename = savedFilename };

        bool saveNeeded = false;
        lock (galleryLoadLock) // Use the lock for modifying the shared dictionary
        {
            if (registeredEmbeddings.TryGetValue(label, out var entriesList))
            {
                entriesList.Add(newEntry);
            }
            else
            {
                registeredEmbeddings[label] = new List<GalleryEntry> { newEntry };
            }
            logger.LogInformation($"Registered '{label}'. Total embeddings for label: {registeredEmbeddings[label].Count}");
            saveNeeded = true;
        }

        // 5. Save updated embeddings to file (outside the lock)
        if (saveNeeded)
        {
            try
            {
                await persistenceService.SaveEmbeddingsAsync(registeredEmbeddings);
                logger.LogInformation("Successfully saved updated embeddings.");
            }
            catch (Exception saveEx)
            {
                logger.LogError(saveEx, "Error saving embeddings after registration.");
                // Decide on behavior - maybe return a specific status?
            }
        }

        // Use Results.Json and return the specific RegisterResponse type
        var registerResponse = new RegisterResponse { Message = $"Face for '{label}' registered successfully.", Filename = savedFilename };
        return Results.Json(registerResponse, PersistenceJsonContext.Default.RegisterResponse);
    }
    catch (Exception ex)
    {
        logger.LogError(ex, "Error during registration.");
        if (imageSavePath != null) Utils.CleanUpImage(imageSavePath);
        return Results.Problem($"An internal error occurred: {ex.Message}"); // Use Results.Problem for 500 errors
    }
})
.Accepts<IFormFile>("multipart/form-data") // Hint for OpenAPI if used
.Produces(StatusCodes.Status200OK)
.Produces(StatusCodes.Status400BadRequest)
.Produces(StatusCodes.Status500InternalServerError)
.WithTags("FaceAPI");


// POST /api/recognize
app.MapPost("/api/recognize", async (
    HttpRequest httpRequest,
    // Make query parameters nullable to allow defaults
    [FromQuery] bool? extractLandmarks, // Read from query (nullable)
    [FromQuery] bool? checkLiveness,     // Read from query (nullable)
    PersistenceService persistenceService, // Inject services
    FaceProcessingService faceProcessingService,
    ConfigurationService configService,
    ILogger<Program> logger) =>
{
    logger.LogInformation("POST /api/recognize called");

    // Assign default values if query parameters are null
    bool actualExtractLandmarks = extractLandmarks ?? false; // Default to false as per original controller
    bool actualCheckLiveness = checkLiveness ?? true;      // Default to true as per original controller

    // Note: Original controller logic forced extractLandmarks = true later. Replicating that here.
    actualExtractLandmarks = true;

    if (!httpRequest.HasFormContentType)
    {
        return Results.BadRequest(new MessageResponse { Message = "Request must be form data." });
    }
    var form = await httpRequest.ReadFormAsync();
    IFormFile? file = form.Files["file"];

    if (file == null || file.Length == 0)
    {
        return Results.BadRequest(new MessageResponse { Message = "Image file is required." });
    }

    // Check if gallery is empty (access static cache)
    if (!registeredEmbeddings.Any())
    {
        return Results.BadRequest(new MessageResponse { Message = "No faces registered yet. Cannot perform recognition." });
    }

    try
    {
        byte[] imageBytes;
        using (var memoryStream = new MemoryStream())
        {
            await file.CopyToAsync(memoryStream);
            imageBytes = memoryStream.ToArray();
        }

        using var image = Utils.DecodeImage(imageBytes);
        if (image == null)
        {
            return Results.BadRequest(new MessageResponse { Message = "Could not decode query image." });
        }

        // Process query image using actual values (with defaults applied)
        var queryResults = faceProcessingService.ProcessImage(image, actualExtractLandmarks, actualCheckLiveness);
        if (!queryResults.Any(r => r.Embedding != null))
        {
            logger.LogWarning("No faces with valid embeddings detected in the query image.");
            var firstResult = queryResults.FirstOrDefault();
            var unknownResponse = new RecognitionResponse
            {
                Label = "unknown",
                Similarity = -1f,
                QueryLandmarks5pt = firstResult?.Landmarks5pt,
                QueryLandmarks68pt = actualExtractLandmarks ? firstResult?.Landmarks68pt : null,
                LivenessScore = firstResult?.LivenessScore ?? -1f,
                IsLive = firstResult?.IsLive ?? false
            };
            return Results.Json(unknownResponse, PersistenceJsonContext.Default.RecognitionResponse);
        }

        // --- Find Best Match (using static cache) ---
        string overallBestMatchLabel = "unknown";
        float overallHighestSimilarity = -1.0f;
        GalleryEntry? overallBestMatchingEntry = null;
        FaceProcessingResult? bestQueryFaceResult = null;

        foreach (var queryFace in queryResults)
        {
            if (queryFace.Embedding == null) continue;

            string currentFaceBestLabel = "unknown";
            float currentFaceHighestSimilarity = -1.0f;
            GalleryEntry? currentFaceBestEntry = null;

            // Access static cache
            foreach (var kvp in registeredEmbeddings)
            {
                var label = kvp.Key;
                var entriesList = kvp.Value;
                foreach (var registeredEntry in entriesList)
                {
                    if (registeredEntry.Embedding == null) continue;
                    float similarity = Utils.CosineSimilarity(queryFace.Embedding, registeredEntry.Embedding);
                    if (similarity > currentFaceHighestSimilarity)
                    {
                        currentFaceHighestSimilarity = similarity;
                        currentFaceBestLabel = label;
                        currentFaceBestEntry = registeredEntry;
                    }
                }
            }

            if (currentFaceHighestSimilarity > overallHighestSimilarity)
            {
                overallHighestSimilarity = currentFaceHighestSimilarity;
                overallBestMatchLabel = currentFaceBestLabel;
                overallBestMatchingEntry = currentFaceBestEntry;
                bestQueryFaceResult = queryFace;
            }
        }

        logger.LogInformation($"Recognition attempt: Best match '{overallBestMatchLabel}' with similarity {overallHighestSimilarity:F4}. Landmarks requested: {actualExtractLandmarks}");

        // Apply threshold
        if (overallHighestSimilarity < configService.SimilarityThreshold)
        {
            overallBestMatchLabel = "unknown";
            overallBestMatchingEntry = null;
        }

        // Prepare response
        var response = new RecognitionResponse
        {
            Label = overallBestMatchLabel,
            Similarity = overallHighestSimilarity,
            QueryEmbedding = bestQueryFaceResult?.Embedding,
            MatchedEmbedding = overallBestMatchingEntry?.Embedding,
            MatchedImageFilename = overallBestMatchingEntry?.ImageFilename,
            QueryLandmarks5pt = bestQueryFaceResult?.Landmarks5pt,
            QueryLandmarks68pt = actualExtractLandmarks ? bestQueryFaceResult?.Landmarks68pt : null,
            LivenessScore = bestQueryFaceResult?.LivenessScore ?? -1f,
            IsLive = bestQueryFaceResult?.IsLive ?? false
        };

        return Results.Json(response, PersistenceJsonContext.Default.RecognitionResponse);
    }
    catch (Exception ex)
    {
        logger.LogError(ex, "Error during recognition.");
        return Results.Problem($"An internal error occurred: {ex.Message}");
    }
})
.Accepts<IFormFile>("multipart/form-data")
.Produces<RecognitionResponse>()
.Produces(StatusCodes.Status400BadRequest)
.Produces(StatusCodes.Status500InternalServerError)
.WithTags("FaceAPI");


// DELETE /api/delete_entry
app.MapDelete("/api/delete_entry", async (
    [FromQuery] string label,
    [FromQuery] string filename,
    PersistenceService persistenceService,
    ConfigurationService configService,
    ILogger<Program> logger) =>
{
    logger.LogInformation($"DELETE /api/delete_entry called for Label: {label}, Filename: {filename}");

    if (string.IsNullOrWhiteSpace(label) || string.IsNullOrWhiteSpace(filename))
    {
        return Results.BadRequest(new MessageResponse { Message = "Label and filename are required." });
    }

    string imagePath = Path.Combine(configService.GalleryDirectory, filename);
    bool entryRemoved = false;
    bool saveNeeded = false;

    try
    {
        lock (galleryLoadLock) // Lock for modification
        {
            if (!registeredEmbeddings.TryGetValue(label, out var entriesList))
            {
                return Results.NotFound(new MessageResponse { Message = $"Label '{label}' not found." });
            }

            int entryIndex = entriesList.FindIndex(e => e.ImageFilename != null && e.ImageFilename.Equals(filename, StringComparison.OrdinalIgnoreCase));

            if (entryIndex == -1)
            {
                return Results.NotFound(new MessageResponse { Message = $"Entry with filename '{filename}' not found for label '{label}'." });
            }

            // Remove from in-memory list
            entriesList.RemoveAt(entryIndex);
            entryRemoved = true;
            logger.LogInformation($"Removed entry {filename} for label {label} from memory.");

            // If list is empty, remove the label key
            if (!entriesList.Any())
            {
                registeredEmbeddings.Remove(label);
                logger.LogInformation($"Removed empty label '{label}' from memory.");
            }
            saveNeeded = true;
        } // End lock

        // Save changes asynchronously (outside the lock)
        if (saveNeeded)
        {
            try
            {
                await persistenceService.SaveEmbeddingsAsync(registeredEmbeddings);
                logger.LogInformation("Successfully saved updated embeddings after deletion.");
            }
            catch (Exception saveEx)
            {
                logger.LogError(saveEx, "Error saving embeddings after deletion.");
                // Decide if this should cause the request to fail - maybe return Problem?
                return Results.Problem("Failed to save changes after deletion.", statusCode: StatusCodes.Status500InternalServerError);
            }
        }

        // Delete the image file (outside the lock)
        if (entryRemoved)
        {
            Utils.CleanUpImage(imagePath); // Uses its own error handling
        }

        return Results.Ok(new MessageResponse { Message = $"Successfully deleted entry: label='{label}', filename='{filename}'" });
    }
    catch (Exception ex)
    {
        logger.LogError(ex, $"Error deleting entry: label='{label}', filename='{filename}'.");
        return Results.Problem($"An internal error occurred during deletion: {ex.Message}");
    }
})
.Produces(StatusCodes.Status200OK)
.Produces(StatusCodes.Status404NotFound)
.Produces(StatusCodes.Status400BadRequest)
.Produces(StatusCodes.Status500InternalServerError)
.WithTags("FaceAPI");


// --- Fallback for SPA routing ---
// Serve index.html from the webapp directory for any requests that don't match an API route or a static file.
// This is crucial for single-page applications using client-side routing.
app.MapFallbackToFile("index.html", new StaticFileOptions { FileProvider = new PhysicalFileProvider(webAppPath) });


// --- Run the App ---
app.Run();
