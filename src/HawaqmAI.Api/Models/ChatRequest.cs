using System.ComponentModel.DataAnnotations;

namespace HawaqmAI.Api.Models;

/// <summary>Incoming chat request from the frontend.</summary>
public sealed class ChatRequest
{
    /// <summary>User's natural language question.</summary>
    [Required]
    [MaxLength(2000)]
    public string Message { get; set; } = string.Empty;

    /// <summary>
    /// Existing session ID to continue a conversation.
    /// Null or omitted starts a new session.
    /// </summary>
    public Guid? SessionId { get; set; }

    /// <summary>Scope bar filters (site, period, pollutants, sensors).</summary>
    public ScopeRequest? Scope { get; set; }
}

/// <summary>Scope bar filter state sent with each chat request.</summary>
public sealed class ScopeRequest
{
    /// <summary>Site ID selected in the scope bar.</summary>
    public int? SiteId { get; set; }

    /// <summary>Region selected in the scope bar — one of "Abu Dhabi", "Al Ain", "Al Dhafra". Null means all regions.</summary>
    public string? Region { get; set; }

    /// <summary>Date range selected in the scope bar.</summary>
    public DateRangeRequest? Period { get; set; }

    /// <summary>Pollutant columns the user wants to query (e.g. ["pm25","co"]).</summary>
    public List<string> Pollutants { get; set; } = [];

    /// <summary>Specific sensor IDs to filter to (empty = all sensors at site).</summary>
    public List<string> Sensors { get; set; } = [];
}

/// <summary>Date range from/to pair.</summary>
public sealed class DateRangeRequest
{
    [Required]
    public DateOnly From { get; set; }

    [Required]
    public DateOnly To { get; set; }
}
