using CarrierDeduperAPI.Service;

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddCors(options =>
{
    options.AddPolicy("AllowBlazorClient", policy =>
    {
        policy.WithOrigins("http://localhost:5153") // Blazor client URL
              .AllowAnyMethod()
              .AllowAnyHeader();
    });
});
// Register services BEFORE Build()
builder.Services.AddSingleton(new EmbeddingService("Model/model.onnx", "Tokenizer/vocab.txt"));

builder.Services.AddControllers();
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();

var app = builder.Build();

// Configure the HTTP request pipeline
if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

app.UseAuthorization();
app.UseCors("AllowBlazorClient");
app.MapControllers();

app.Run();

