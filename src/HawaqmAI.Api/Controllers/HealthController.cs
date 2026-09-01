using HawaqmAI.Api.Configuration;
using HawaqmAI.Api.Data;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Data.SqlClient;
using Microsoft.Extensions.Options;
using Serilog;

namespace HawaqmAI.Api.Controllers;

/// <summary>
/// Health check endpoint for IIS monitoring and load balancer probes.
/// GET /api/health
/// </summary>
[ApiController]
[Route("api/[controller]")]
public sealed class HealthController : ControllerBase
{
    private static readonly Serilog.ILogger _log = Log.ForContext<HealthController>();
    private readonly ChatHistoryDbContext _chatDb;
    private readonly DatabaseOptions _dbOptions;

    public HealthController(ChatHistoryDbContext chatDb, IOptions<DatabaseOptions> dbOptions)
    {
        _chatDb = chatDb;
        _dbOptions = dbOptions.Value;
    }

    /// <summary>
    /// Returns service health status including database connectivity.
    /// </summary>
    [HttpGet]
    [ProducesResponseType(typeof(HealthResponse), 200)]
    [ProducesResponseType(typeof(HealthResponse), 503)]
    public async Task<IActionResult> GetHealth(CancellationToken ct)
    {
        var checks = new Dictionary<string, string>();
        var healthy = true;

        // Check chat history DB
        try
        {
            var canConnect = await _chatDb.Database.CanConnectAsync(ct);
            checks["chat_db"] = canConnect ? "ok" : "error";
            if (!canConnect) healthy = false;
        }
        catch (Exception ex)
        {
            _log.Warning(ex, "Chat DB health check failed");
            checks["chat_db"] = "error";
            healthy = false;
        }

        // Check air quality DB
        try
        {
            await using var conn = new SqlConnection(_dbOptions.AirQualityConnection);
            await conn.OpenAsync(ct);
            checks["airquality_db"] = "ok";
        }
        catch (Exception ex)
        {
            _log.Warning(ex, "Air quality DB health check failed");
            checks["airquality_db"] = "error";
            healthy = false;
        }

        var response = new HealthResponse
        {
            Status = healthy ? "healthy" : "degraded",
            Version = "1.0.0",
            Timestamp = DateTime.UtcNow,
            Checks = checks
        };

        return healthy ? Ok(response) : StatusCode(503, response);
    }
}

/// <summary>Health check response payload.</summary>
public sealed class HealthResponse
{
    public string Status { get; init; } = string.Empty;
    public string Version { get; init; } = string.Empty;
    public DateTime Timestamp { get; init; }
    public Dictionary<string, string> Checks { get; init; } = [];
}
