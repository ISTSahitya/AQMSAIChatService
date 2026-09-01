namespace HawaqmAI.Api.Services;

/// <summary>
/// Resolves a station name (as typed by a user) to its numeric DB ID
/// by doing a fuzzy LIKE lookup against DMN_Stations.StationName.
/// Results are cached for 10 minutes.
/// </summary>
public interface IStationResolverService
{
    /// <summary>
    /// Returns the station ID for the best name match, or null if not found.
    /// </summary>
    Task<int?> ResolveByNameAsync(string stationName, CancellationToken ct = default);
}
