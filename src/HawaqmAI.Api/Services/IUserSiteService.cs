namespace HawaqmAI.Api.Services;

/// <summary>
/// Resolves the station IDs a user is permitted to access,
/// by querying the AQMS DB via UserId → GroupID → RoleStations.
/// </summary>
public interface IUserSiteService
{
    /// <summary>
    /// Returns the list of StationIDs the user is allowed to see.
    /// Empty list means the user has no group-based site restriction (admin).
    /// Results are cached per userId for the lifetime of the request pipeline.
    /// </summary>
    Task<List<int>> GetPermittedSiteIdsAsync(int userId, CancellationToken ct = default);
}
