using Dapper;
using HawaqmAI.Api.Configuration;
using Microsoft.Data.SqlClient;
using Microsoft.Extensions.Caching.Memory;
using Microsoft.Extensions.Options;
using Serilog;

namespace HawaqmAI.Api.Services;

/// <summary>
/// Queries the AQMS DB to resolve which stations a user is permitted to access.
/// Pipeline: Users.GroupID → RoleStations.StationID (same DB as air quality data).
///
/// Results are cached in-memory for 5 minutes per userId to avoid a DB hit
/// on every chat message while still picking up permission changes reasonably quickly.
/// </summary>
public sealed class UserSiteService : IUserSiteService
{
    private static readonly Serilog.ILogger _log = Log.ForContext<UserSiteService>();
    private readonly DatabaseOptions _dbOptions;
    private readonly IMemoryCache _cache;

    // Cache permitted site IDs per user for 5 minutes
    private static readonly TimeSpan CacheDuration = TimeSpan.FromMinutes(5);

    public UserSiteService(IOptions<DatabaseOptions> dbOptions, IMemoryCache cache)
    {
        _dbOptions = dbOptions.Value;
        _cache = cache;
    }

    /// <inheritdoc/>
    public async Task<List<int>> GetPermittedSiteIdsAsync(int userId, CancellationToken ct = default)
    {
        var cacheKey = $"user_sites:{userId}";

        if (_cache.TryGetValue(cacheKey, out List<int>? cached) && cached is not null)
            return cached;

        try
        {
            await using var conn = new SqlConnection(_dbOptions.AirQualityConnection);
            await conn.OpenAsync(ct);

            // Step 1: get the user's GroupID
            var groupId = await conn.ExecuteScalarAsync<int?>(
                new CommandDefinition(
                    "SELECT TOP 1 GroupID FROM Users WHERE ID = @userId AND Status = 1",
                    new { userId },
                    cancellationToken: ct));

            if (groupId is null)
            {
                _log.Warning("UserSiteService: no active user found for UserId={UserId}", userId);
                var empty = new List<int>();
                _cache.Set(cacheKey, empty, CacheDuration);
                return empty;
            }

            // Step 2: get all StationIDs assigned to that group
            var siteIds = (await conn.QueryAsync<int>(
                new CommandDefinition(
                    "SELECT StationID FROM RoleStations WHERE UserGroupID = @groupId",
                    new { groupId },
                    cancellationToken: ct))).AsList();

            _log.Debug(
                "UserSiteService: UserId={UserId} GroupID={GroupId} has {Count} permitted sites: [{Sites}]",
                userId, groupId, siteIds.Count, string.Join(",", siteIds));

            _cache.Set(cacheKey, siteIds, CacheDuration);
            return siteIds;
        }
        catch (Exception ex)
        {
            _log.Error(ex, "UserSiteService: failed to resolve sites for UserId={UserId}", userId);
            // Return empty on failure — RBAC will treat as all-sites-access.
            // Log is sufficient; don't break the request.
            return new List<int>();
        }
    }
}
