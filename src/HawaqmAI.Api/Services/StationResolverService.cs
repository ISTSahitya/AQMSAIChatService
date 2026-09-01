using Dapper;
using HawaqmAI.Api.Configuration;
using Microsoft.Data.SqlClient;
using Microsoft.Extensions.Caching.Memory;
using Microsoft.Extensions.Options;
using Serilog;

namespace HawaqmAI.Api.Services;

/// <summary>
/// Resolves a user-typed station name to a numeric DB ID via fuzzy LIKE lookup.
/// Caches all station names for 10 minutes to avoid repeated DB hits.
/// </summary>
public sealed class StationResolverService : IStationResolverService
{
    private static readonly Serilog.ILogger _log = Log.ForContext<StationResolverService>();
    private readonly DatabaseOptions _dbOptions;
    private readonly IMemoryCache _cache;
    private const string CacheKey = "all_station_names";
    private static readonly TimeSpan CacheDuration = TimeSpan.FromMinutes(10);

    public StationResolverService(IOptions<DatabaseOptions> dbOptions, IMemoryCache cache)
    {
        _dbOptions = dbOptions.Value;
        _cache = cache;
    }

    /// <inheritdoc/>
    public async Task<int?> ResolveByNameAsync(string stationName, CancellationToken ct = default)
    {
        if (string.IsNullOrWhiteSpace(stationName)) return null;

        var stations = await GetAllStationsAsync(ct);
        var query = stationName.Trim().ToLowerInvariant();

        // 1. Exact match (case-insensitive)
        var exact = stations.FirstOrDefault(s => s.Name.Equals(query, StringComparison.OrdinalIgnoreCase));
        if (exact != default)
        {
            _log.Debug("StationResolver: exact match '{Name}' → ID={Id}", stationName, exact.Id);
            return exact.Id;
        }

        // 2. Contains match — find station whose name contains the query or vice versa
        var contains = stations
            .Where(s => s.Name.Contains(query, StringComparison.OrdinalIgnoreCase)
                     || query.Contains(s.Name, StringComparison.OrdinalIgnoreCase))
            .OrderBy(s => Math.Abs(s.Name.Length - query.Length)) // prefer closest length
            .FirstOrDefault();

        if (contains != default)
        {
            _log.Debug("StationResolver: contains match '{Query}' → '{Name}' ID={Id}", stationName, contains.Name, contains.Id);
            return contains.Id;
        }

        // 3. Word overlap — score by how many words from the query appear in the station name
        var queryWords = query.Split([' ', '-', '_'], StringSplitOptions.RemoveEmptyEntries)
                              .Where(w => w.Length > 2)
                              .ToHashSet();

        var bestOverlap = stations
            .Select(s => new
            {
                s.Id,
                s.Name,
                Score = queryWords.Count(w => s.Name.Contains(w, StringComparison.OrdinalIgnoreCase))
            })
            .Where(x => x.Score > 0)
            .OrderByDescending(x => x.Score)
            .FirstOrDefault();

        if (bestOverlap != null)
        {
            _log.Debug("StationResolver: word-overlap match '{Query}' → '{Name}' ID={Id} (score={Score})",
                stationName, bestOverlap.Name, bestOverlap.Id, bestOverlap.Score);
            return bestOverlap.Id;
        }

        _log.Warning("StationResolver: no match found for '{Name}'", stationName);
        return null;
    }

    private async Task<List<(int Id, string Name)>> GetAllStationsAsync(CancellationToken ct)
    {
        if (_cache.TryGetValue(CacheKey, out List<(int, string)>? cached) && cached is not null)
            return cached;

        await using var conn = new SqlConnection(_dbOptions.AirQualityConnection);
        await conn.OpenAsync(ct);

        var rows = await conn.QueryAsync<(int Id, string Name)>(
            new CommandDefinition(
                "SELECT ID, StationName FROM DMN_Stations",
                cancellationToken: ct));

        var list = rows.AsList();
        _cache.Set(CacheKey, list, CacheDuration);
        _log.Debug("StationResolver: cached {Count} station names", list.Count);
        return list;
    }
}
