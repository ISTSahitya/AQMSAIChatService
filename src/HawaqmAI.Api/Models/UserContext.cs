namespace HawaqmAI.Api.Models;

/// <summary>
/// Authenticated user identity and permissions extracted from the JWT token.
/// Built by JwtAuthMiddleware and passed through the request pipeline.
/// </summary>
public sealed class UserContext
{
    /// <summary>Unique user identifier (sub claim from JWT).</summary>
    public string UserId { get; set; } = string.Empty;

    /// <summary>Display name from JWT name claim.</summary>
    public string DisplayName { get; set; } = string.Empty;

    /// <summary>Email from JWT email claim.</summary>
    public string Email { get; set; } = string.Empty;

    /// <summary>
    /// Role from JWT (e.g. "admin", "regulator", "viewer").
    /// Controls which approved query templates are visible.
    /// </summary>
    public string Role { get; set; } = "viewer";

    /// <summary>
    /// Site IDs this user is permitted to query.
    /// Empty list means access to ALL sites (admin).
    /// </summary>
    public List<int> PermittedSiteIds { get; set; } = [];

    /// <summary>
    /// Query categories this role can access (e.g. ["air_quality"]).
    /// Device health templates are restricted to device_admin role.
    /// </summary>
    public List<string> PermittedCategories { get; set; } = ["air_quality"];

    /// <summary>Whether the user has access to all sites (admin shortcut).</summary>
    public bool HasAllSitesAccess => PermittedSiteIds.Count == 0;

    /// <summary>Client IP address for audit logging.</summary>
    public string IpAddress { get; set; } = string.Empty;
}
