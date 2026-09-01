namespace HawaqmAI.Api.Models;

/// <summary>
/// Resolved scope context injected into SQL queries by the RBAC engine.
/// Built from the user's scope bar selection + their JWT permitted sites.
/// </summary>
public sealed class ResolvedScope
{
    /// <summary>Site IDs that RBAC has approved for this query (intersection of user selection + JWT grants).</summary>
    public List<int> SiteIds { get; set; } = [];

    /// <summary>Start of date range (inclusive).</summary>
    public DateOnly? StartDate { get; set; }

    /// <summary>End of date range (inclusive).</summary>
    public DateOnly? EndDate { get; set; }

    /// <summary>Validated pollutant columns (e.g. ["pm25", "co"]).</summary>
    public List<string> Pollutants { get; set; } = [];

    /// <summary>Sensor IDs to restrict to (empty = all).</summary>
    public List<string> Sensors { get; set; } = [];

    /// <summary>Region filter from scope bar — "Abu Dhabi", "Al Ain", or "Al Dhafra". Null means all regions.</summary>
    public string? Region { get; set; }

    /// <summary>Human-readable date range string for response metadata.</summary>
    public string DateRangeLabel
    {
        get
        {
            if (StartDate is null && EndDate is null) return string.Empty;
            var fmt = "MMM d, yyyy";
            return StartDate == EndDate
                ? StartDate!.Value.ToString(fmt)
                : $"{StartDate?.ToString(fmt)} – {EndDate?.ToString(fmt)}";
        }
    }
}
