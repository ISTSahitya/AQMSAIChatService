using HawaqmAI.Api.Models;
using Serilog;

namespace HawaqmAI.Api.Validators;

/// <summary>
/// Validates and resolves scope bar input against the user's JWT-granted permissions.
/// Produces a <see cref="ResolvedScope"/> that can be safely injected into SQL.
/// </summary>
public static class ScopeValidator
{
    private static readonly Serilog.ILogger _log = Log.ForContext(typeof(ScopeValidator));

    private static readonly HashSet<string> AllowedRegions =
        new(StringComparer.OrdinalIgnoreCase) { "Abudhabi", "AL Ain", "AlDhafra" };

    private static readonly HashSet<string> AllowedPollutants =
        new(StringComparer.OrdinalIgnoreCase)
        {
            "pm25", "pm10", "co", "co2", "no2", "o3", "so2", "ch2o", "tvoc", "aqi", "temperature", "humidity"
        };

    /// <summary>
    /// Resolves and validates the user's scope request against their permitted sites.
    /// Returns a validated <see cref="ResolvedScope"/> ready for SQL injection.
    /// </summary>
    public static (ResolvedScope Scope, string? Error) Resolve(
        ScopeRequest? request,
        UserContext user)
    {
        var scope = new ResolvedScope();

        // ── Site IDs ─────────────────────────────────────────────────────────
        if (request?.SiteId.HasValue == true)
        {
            var requestedId = request.SiteId.Value;
            if (!user.HasAllSitesAccess && !user.PermittedSiteIds.Contains(requestedId))
            {
                _log.Warning(
                    "User {UserId} requested site {SiteId} but is not permitted",
                    user.UserId, requestedId);
                return (scope, $"You do not have access to site {requestedId}.");
            }
            scope.SiteIds = [requestedId];
        }
        else
        {
            // No site specified — use all permitted sites
            scope.SiteIds = user.HasAllSitesAccess ? [] : user.PermittedSiteIds;
        }

        // ── Date range ───────────────────────────────────────────────────────
        if (request?.Period is not null)
        {
            if (request.Period.From > request.Period.To)
                return (scope, "Scope date range is invalid: 'from' is after 'to'.");

            var span = request.Period.To.ToDateTime(TimeOnly.MinValue) -
                       request.Period.From.ToDateTime(TimeOnly.MinValue);

            if (span.TotalDays > 365)
                return (scope, "Date range cannot exceed 365 days.");

            scope.StartDate = request.Period.From;
            scope.EndDate = request.Period.To;
        }

        // ── Pollutants ───────────────────────────────────────────────────────
        if (request?.Pollutants is { Count: > 0 })
        {
            var invalid = request.Pollutants
                .Where(p => !AllowedPollutants.Contains(p))
                .ToList();

            if (invalid.Count > 0)
                return (scope, $"Unknown pollutant(s): {string.Join(", ", invalid)}");

            scope.Pollutants = request.Pollutants
                .Select(p => p.ToLowerInvariant())
                .Distinct()
                .ToList();
        }

        // ── Region ───────────────────────────────────────────────────────────
        if (!string.IsNullOrWhiteSpace(request?.Region))
        {
            if (!AllowedRegions.Contains(request.Region))
                return (scope, $"Unknown region '{request.Region}'. Valid regions: Abu Dhabi, Al Ain, Al Dhafra.");
            scope.Region = request.Region;
        }

        // ── Sensors ──────────────────────────────────────────────────────────
        if (request?.Sensors is { Count: > 0 })
        {
            // Sensor IDs are alphanumeric only (prevent injection)
            var invalidSensors = request.Sensors
                .Where(s => !s.All(c => char.IsLetterOrDigit(c) || c == '-' || c == '_'))
                .ToList();

            if (invalidSensors.Count > 0)
                return (scope, $"Invalid sensor ID format: {string.Join(", ", invalidSensors)}");

            scope.Sensors = request.Sensors;
        }

        return (scope, null);
    }
}
