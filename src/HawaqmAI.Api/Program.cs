using System.Text;
using Azure.Identity;
using HawaqmAI.Api.Configuration;
using Microsoft.Extensions.Options;
using HawaqmAI.Api.Data;
using HawaqmAI.Api.Middleware;
using HawaqmAI.Api.Services;
using Microsoft.AspNetCore.Authentication.JwtBearer;
using Microsoft.EntityFrameworkCore;
using Microsoft.IdentityModel.Tokens;
using Microsoft.OpenApi.Models;
using Serilog;
using Serilog.Events;

// Bootstrap logger for startup errors
Log.Logger = new LoggerConfiguration()
    .MinimumLevel.Override("Microsoft", LogEventLevel.Information)
    .Enrich.FromLogContext()
    .WriteTo.Console()
    .CreateBootstrapLogger();

try
{
    Log.Information("Starting HAWAQM AI Chat Service");

    var builder = WebApplication.CreateBuilder(args);

    // ── Serilog ─────────────────────────────────────────────────────────────
    builder.Host.UseSerilog((context, services, configuration) => configuration
        .ReadFrom.Configuration(context.Configuration)
        .ReadFrom.Services(services)
        .Enrich.FromLogContext()
        .WriteTo.Console()
        .WriteTo.File(
            path: "logs/hawaqm-ai-.log",
            rollingInterval: RollingInterval.Day,
            retainedFileCountLimit: 30,
            outputTemplate: "[{Timestamp:yyyy-MM-dd HH:mm:ss.fff zzz} {Level:u3}] {Message:lj} {Properties:j}{NewLine}{Exception}"));

    // ── Configuration options ────────────────────────────────────────────────
    builder.Services.Configure<AzureAIOptions>(builder.Configuration.GetSection("AzureAI"));
    builder.Services.Configure<DatabaseOptions>(builder.Configuration.GetSection("Database"));
    builder.Services.Configure<ChatOptions>(builder.Configuration.GetSection("Chat"));
    builder.Services.Configure<AqmsApiOptions>(builder.Configuration.GetSection("AqmsApi"));

    // ── Entity Framework — Chat History (read-write) ────────────────────────
    var chatConnStr = builder.Configuration["Database:ChatHistoryConnection"];
    if (string.IsNullOrWhiteSpace(chatConnStr))
        throw new InvalidOperationException("Database:ChatHistoryConnection must be configured.");
    builder.Services.AddDbContext<ChatHistoryDbContext>(options =>
        options.UseSqlServer(chatConnStr, sql =>
        {
            sql.CommandTimeout(30);
            sql.EnableRetryOnFailure(maxRetryCount: 3, maxRetryDelay: TimeSpan.FromSeconds(5), errorNumbersToAdd: null);
        }));

    // ── Dapper / ADO.NET — Air Quality Data (read-only) ────────────────────
    // Connection string is injected directly into SqlExecutorService via IOptions

    // ── In-memory cache (replaces Redis for v1) ──────────────────────────────
    builder.Services.AddMemoryCache();

    // ── Application services ─────────────────────────────────────────────────
    builder.Services.AddHttpContextAccessor();
    builder.Services.AddSingleton<IQueryRouterService, QueryRouterService>();
    builder.Services.AddSingleton<IAzureAIService, AzureAIService>();
    builder.Services.AddScoped<IRbacEngine, RbacEngine>();
    builder.Services.AddScoped<ISqlExecutorService, SqlExecutorService>();
    builder.Services.AddScoped<ISessionService, SessionService>();
    builder.Services.AddScoped<IResponseFormatterService, ResponseFormatterService>();
    builder.Services.AddScoped<IUserSiteService, UserSiteService>();
    builder.Services.AddScoped<IStationResolverService, StationResolverService>();
    builder.Services.AddScoped<IExternalApiService, ExternalApiService>();

    // Background cleanup service
    builder.Services.AddHostedService<SessionCleanupService>();

    // ── JWT Authentication ───────────────────────────────────────────────────
    var jwtSection = builder.Configuration.GetSection("Jwt");
    var jwtSecret = jwtSection["Secret"];
    var jwtIssuer = jwtSection["Issuer"];
    var jwtAudience = jwtSection["Audience"];

    if (string.IsNullOrWhiteSpace(jwtSecret) && builder.Environment.IsProduction())
        throw new InvalidOperationException("Jwt:Secret must be configured in production.");

    if (!string.IsNullOrWhiteSpace(jwtSecret))
    {
        builder.Services.AddAuthentication(options =>
        {
            options.DefaultAuthenticateScheme = JwtBearerDefaults.AuthenticationScheme;
            options.DefaultChallengeScheme = JwtBearerDefaults.AuthenticationScheme;
        })
        .AddJwtBearer(options =>
        {
            // Disable ASP.NET's automatic claim type remapping so claim names
            // stay as they are in the token (e.g. "UserId", "Role", "UserName")
            options.MapInboundClaims = false;
            options.TokenValidationParameters = new TokenValidationParameters
            {
                ValidateIssuerSigningKey = true,
                IssuerSigningKey = new SymmetricSecurityKey(Encoding.UTF8.GetBytes(jwtSecret)),
                ValidateIssuer = !string.IsNullOrWhiteSpace(jwtIssuer),
                ValidIssuer = jwtIssuer,
                ValidateAudience = !string.IsNullOrWhiteSpace(jwtAudience),
                ValidAudience = jwtAudience,
                ValidateLifetime = true,
                ClockSkew = TimeSpan.FromMinutes(5),
                NameClaimType = "UserName",
                RoleClaimType = "Role"
            };
            options.Events = new JwtBearerEvents
            {
                OnMessageReceived = ctx =>
                {
                    // Read JWT from the same HttpOnly cookie the main AQMS API sets
                    if (ctx.Request.Cookies.TryGetValue("Token", out var cookieToken))
                        ctx.Token = cookieToken;
                    else if (ctx.Request.Cookies.TryGetValue("token", out var cookieTokenLower))
                        ctx.Token = cookieTokenLower;
                    return Task.CompletedTask;
                },
                OnAuthenticationFailed = ctx =>
                {
                    Log.Warning("JWT authentication failed: {Error}", ctx.Exception.Message);
                    return Task.CompletedTask;
                },
                OnTokenValidated = ctx =>
                {
                    Log.Information("JWT token validated for: {User}", ctx.Principal?.Identity?.Name ?? "unknown");
                    return Task.CompletedTask;
                },
                OnChallenge = ctx =>
                {
                    Log.Warning("JWT challenge: {Error} | {ErrorDescription}", ctx.Error, ctx.ErrorDescription);
                    return Task.CompletedTask;
                }
            };
        });
    }
    else
    {
        Log.Warning("JWT Secret not configured — authentication is DISABLED. Set Jwt:Secret for production.");
        builder.Services.AddAuthentication(); // no-op auth for dev
    }

    builder.Services.AddAuthorization();

    // ── CORS ─────────────────────────────────────────────────────────────────
    var allowedOrigins = builder.Configuration.GetSection("Cors:AllowedOrigins").Get<string[]>()
        ?? ["http://localhost:3000"];
    // Always ensure localhost:3000 is included for dev
    if (!allowedOrigins.Contains("http://localhost:3000"))
        allowedOrigins = [.. allowedOrigins, "http://localhost:3000"];
    builder.Services.AddCors(options =>
    {
        options.AddPolicy("HawaqmCors", policy =>
            policy.WithOrigins(allowedOrigins)
                  .AllowAnyHeader()
                  .AllowAnyMethod()
                  .AllowCredentials());
    });

    // ── Controllers + Swagger ─────────────────────────────────────────────────
    builder.Services.AddControllers()
        .AddJsonOptions(options =>
        {
            options.JsonSerializerOptions.PropertyNamingPolicy = System.Text.Json.JsonNamingPolicy.CamelCase;
            options.JsonSerializerOptions.DefaultIgnoreCondition = System.Text.Json.Serialization.JsonIgnoreCondition.WhenWritingNull;
        });

    builder.Services.AddEndpointsApiExplorer();
    builder.Services.AddSwaggerGen(c =>
    {
        c.SwaggerDoc("v1", new OpenApiInfo
        {
            Title = "HAWAQM AI Chat Service",
            Version = "v1",
            Description = "AI-powered natural language interface for air quality monitoring data"
        });
        c.AddSecurityDefinition("Bearer", new OpenApiSecurityScheme
        {
            Description = "JWT Authorization header using the Bearer scheme. Enter 'Bearer {token}'",
            Name = "Authorization",
            In = ParameterLocation.Header,
            Type = SecuritySchemeType.ApiKey,
            Scheme = "Bearer"
        });
        c.AddSecurityRequirement(new OpenApiSecurityRequirement
        {
            {
                new OpenApiSecurityScheme { Reference = new OpenApiReference { Type = ReferenceType.SecurityScheme, Id = "Bearer" } },
                Array.Empty<string>()
            }
        });
    });

    // ── HttpClient for Azure AI ──────────────────────────────────────────────
    builder.Services.AddHttpClient("AzureAI", client =>
    {
        client.Timeout = TimeSpan.FromSeconds(90);
    });

    // ── HttpClient for AQMS Web API ──────────────────────────────────────────
    builder.Services.AddHttpClient("AqmsApi", (sp, client) =>
    {
        var opts = sp.GetRequiredService<IOptions<AqmsApiOptions>>().Value;
        if (!string.IsNullOrWhiteSpace(opts.BaseUrl))
            client.BaseAddress = new Uri(opts.BaseUrl);
        client.Timeout = TimeSpan.FromSeconds(opts.TimeoutSeconds);
    });

    var app = builder.Build();

    // ── Auto-migrate chat tables on startup ──────────────────────────────────
    using (var scope = app.Services.CreateScope())
    {
        try
        {
            var db = scope.ServiceProvider.GetRequiredService<ChatHistoryDbContext>();
            // Apply pending migrations without attempting to CREATE the database
            // (the DB already exists — shared with the main AQMS database)
            var pending = await db.Database.GetPendingMigrationsAsync();
            if (pending.Any())
            {
                await db.Database.MigrateAsync();
                Log.Information("Chat history database migrations applied");
            }
            else
            {
                Log.Information("Chat history database is up to date");
            }
        }
        catch (Exception ex)
        {
            Log.Warning(ex, "Could not apply database migrations — chat history may be unavailable");
        }
    }

    // ── Middleware pipeline ───────────────────────────────────────────────────
    app.UseSerilogRequestLogging(options =>
    {
        options.MessageTemplate = "HTTP {RequestMethod} {RequestPath} responded {StatusCode} in {Elapsed:0.0000}ms";
    });

    if (app.Environment.IsDevelopment())
    {
        app.UseSwagger();
        app.UseSwaggerUI(c =>
        {
            c.SwaggerEndpoint("/swagger/v1/swagger.json", "HAWAQM AI v1");
            c.RoutePrefix = "swagger";
        });
        app.UseDeveloperExceptionPage();
    }

    app.UseCors("HawaqmCors");

    app.UseAuthentication();
    app.UseAuthorization();

    app.UseMiddleware<RequestLoggingMiddleware>();

    // Global exception handler
    app.UseExceptionHandler(errApp =>
    {
        errApp.Run(async ctx =>
        {
            ctx.Response.StatusCode = 500;
            ctx.Response.ContentType = "application/json";
            var errorResponse = new
            {
                success = false,
                error = "An unexpected error occurred.",
                requestId = ctx.TraceIdentifier
            };
            await ctx.Response.WriteAsJsonAsync(errorResponse);
        });
    });

    app.MapControllers();

    Log.Information("HAWAQM AI Chat Service started on {Urls}", string.Join(", ", app.Urls));
    await app.RunAsync();
}
catch (Exception ex)
{
    Log.Fatal(ex, "HAWAQM AI Chat Service terminated unexpectedly");
}
finally
{
    await Log.CloseAndFlushAsync();
}
