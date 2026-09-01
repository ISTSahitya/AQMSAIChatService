using System.Text.Json;
using HawaqmAI.Api.Configuration;
using HawaqmAI.Api.Models;
using Microsoft.Extensions.Options;
using Serilog;

namespace HawaqmAI.Api.Services;

/// <summary>
/// Builds the final ChatResponse from pipeline outputs.
/// Handles text, table, and chart response types.
/// Injects AQI category labels and mandatory metadata.
/// </summary>
public sealed class ResponseFormatterService : IResponseFormatterService
{
    private static readonly Serilog.ILogger _log = Log.ForContext<ResponseFormatterService>();
    private readonly ChatOptions _chatOptions;

    // MOCCAE AQI levels — order matters (checked lowest to highest)
    private static readonly (int Max, string Label)[] AqiLevels =
    [
        (50,  "Good"),
        (100, "Moderate"),
        (150, "Unhealthy for Sensitive Groups"),
        (200, "Unhealthy"),
        (300, "Very Unhealthy"),
        (500, "Hazardous")
    ];

    public ResponseFormatterService(IOptions<ChatOptions> chatOptions)
    {
        _chatOptions = chatOptions.Value;
    }

    /// <inheritdoc/>
    public ChatResponse Format(
        Guid sessionId,
        Guid messageId,
        ApprovedQuery template,
        string llmSummary,
        List<Dictionary<string, object?>> rows,
        int rowCount,
        int executionTimeMs,
        ResolvedScope scope,
        string? modelUsed,
        int tokenCount)
    {
        var responseType = template.ResponseType;

        // Enrich AQI numeric values with category labels
        if (rows.Count > 0)
            rows = EnrichWithAqiLabels(rows);

        // For device_last_reading: strip metadata columns from table display
        // (DeviceName and LastMeasured are the same on every row — shown once in the summary)
        if (template.Id == "device_last_reading" && rows.Count > 0)
        {
            rows = rows.Select(r =>
            {
                var d = new Dictionary<string, object?>(r);
                d.Remove("DeviceName");
                d.Remove("LastMeasured");
                return d;
            }).ToList();
        }

        // For region_aqi_geographical: strip AQICategory — already in summary text
        if (template.Id == "region_aqi_geographical" && rows.Count > 0)
        {
            rows = rows.Select(r =>
            {
                var d = new Dictionary<string, object?>(r);
                d.Remove("AQICategory");
                return d;
            }).ToList();
        }

        // For site_aqi_all and site_aqi_single: AQI Category is already set by AggregateByStation —
        // remove the duplicate AQI_category column that EnrichWithAqiLabels adds for the "AQI" column
        if (template.Id is "site_aqi_all" or "site_aqi_single" && rows.Count > 0)
        {
            rows = rows.Select(r =>
            {
                var d = new Dictionary<string, object?>(r);
                d.Remove("AQI_category");
                d.Remove("AQI Category_category");
                return d;
            }).ToList();
        }

        // Build column list from first row keys
        var columns = rows.Count > 0
            ? rows[0].Keys.ToList()
            : [];

        // For chart type, serialize data as JSON string for ChartData field too
        string? chartDataJson = null;
        if (responseType == "chart" && rows.Count > 0)
        {
            chartDataJson = JsonSerializer.Serialize(rows,
                new JsonSerializerOptions { PropertyNamingPolicy = JsonNamingPolicy.CamelCase });
        }

        var metadata = new ResponseMetadata
        {
            DataSource = template.TableUsed,
            DateRange = scope.DateRangeLabel,
            RowCount = rowCount,
            ExecutionTimeMs = executionTimeMs,
            ModelUsed = modelUsed,
            TokenCount = tokenCount,
            QueryTemplateId = template.Id,
            Disclaimer = _chatOptions.Disclaimer
        };

        _log.Debug("Formatted response: type={Type}, rows={Rows}, template={Template}",
            responseType, rowCount, template.Id);

        // Text-only templates — suppress table data so only the summary is shown
        var suppressTable = responseType == "text";

        return new ChatResponse
        {
            Success = true,
            Message = llmSummary,
            Sql = null, // Only expose SQL in debug/admin mode — omit for regulators
            ResponseType = responseType,
            ChartType = template.ChartType,
            Data = !suppressTable && rows.Count > 0 ? rows : null,
            Columns = !suppressTable && columns.Count > 0 ? columns : null,
            Metadata = metadata,
            SessionId = sessionId,
            MessageId = messageId
        };
    }

    /// <inheritdoc/>
    public ChatResponse FormatError(
        Guid sessionId,
        string errorMessage,
        string errorCode,
        string? clarificationPrompt = null)
    {
        return new ChatResponse
        {
            Success = false,
            Message = errorMessage,
            ResponseType = "text",
            SessionId = sessionId,
            MessageId = Guid.NewGuid(),
            Error = errorMessage,
            ErrorCode = errorCode,
            ClarificationPrompt = clarificationPrompt
        };
    }

    /// <inheritdoc/>
    public string ClassifyAqi(double aqiValue)
    {
        foreach (var (max, label) in AqiLevels)
        {
            if (aqiValue <= max) return label;
        }
        return "Hazardous";
    }

    // ── Private helpers ──────────────────────────────────────────────────────

    private List<Dictionary<string, object?>> EnrichWithAqiLabels(List<Dictionary<string, object?>> rows)
    {
        var aqiKeys = rows[0].Keys
            .Where(k => k.Contains("aqi", StringComparison.OrdinalIgnoreCase))
            .ToList();

        if (aqiKeys.Count == 0) return rows;

        return rows.Select(row =>
        {
            var enriched = new Dictionary<string, object?>(row);
            foreach (var key in aqiKeys)
            {
                if (row.TryGetValue(key, out var val) && val is not null)
                {
                    if (double.TryParse(val.ToString(), out var aqiNum))
                    {
                        enriched[$"{key}_category"] = ClassifyAqi(aqiNum);
                    }
                }
            }
            return enriched;
        }).ToList();
    }
}
