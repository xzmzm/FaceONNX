using FaceONNX.Backend.Services;
using Microsoft.Extensions.FileProviders; // For static files
using System.IO; // For Path
using System.Text.Json; // For JsonNamingPolicy

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

// 3. Add Controllers service
builder.Services.AddControllers()
    .AddJsonOptions(options =>
    {
        // Use snake_case for JSON property names to match Python backend
        options.JsonSerializerOptions.PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower;
    });

// 4. Add Swagger/OpenAPI (Keep for development)
builder.Services.AddEndpointsApiExplorer();

// --- Build the App ---
var app = builder.Build();

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

app.MapControllers(); // Maps attribute-routed controllers (like FaceController)

// --- Fallback for SPA routing ---
// Serve index.html from the webapp directory for any requests that don't match an API route or a static file.
// This is crucial for single-page applications using client-side routing.
app.MapFallbackToFile("index.html", new StaticFileOptions { FileProvider = new PhysicalFileProvider(webAppPath) });


// --- Run the App ---
app.Run();
