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
    RequestPath = "/images" // Serve gallery images from /images path
});

app.UseCors(AllowSpecificOrigins);

// app.UseAuthorization();

app.MapControllers(); // Maps attribute-routed controllers (like FaceController)

// --- Run the App ---
app.Run();
