using System.Diagnostics;
using HawaqmAI.Api.Models;
using HawaqmAI.Api.Services;
using Serilog;

namespace HawaqmAI.Api.Middleware;

/// <summary>
/// Audit logging middleware — logs every request with user ID, IP, path, method,
/// status code, and elapsed time. Required by NFR-02.
/// Runs AFTER authentication so user identity is available.
/// Also resolves permitted site IDs from the DB via IUserSiteService.
/// </summary>
public sealed class RequestLoggingMiddleware
{
    private readonly RequestDelegate _next;
    private static readonly Serilog.ILogger _log = Log.ForContext<RequestLoggingMiddleware>();

    public RequestLoggingMiddleware(RequestDelegate next)
    {
        _next = next;
    }

    public async Task InvokeAsync(HttpContext context, IUserSiteService userSiteService)
    {
        var sw = Stopwatch.StartNew();

        // Supports both DOH token claims (UserId) and standard JWT claims (sub)
        var userIdStr = context.User?.FindFirst("UserId")?.Value
                     ?? context.User?.FindFirst("sub")?.Value
                     ?? context.User?.FindFirst("http://schemas.xmlsoap.org/ws/2005/05/identity/claims/nameidentifier")?.Value
                     ?? "anonymous";

        var ip = context.Connection.RemoteIpAddress?.ToString() ?? "unknown";
        var method = context.Request.Method;
        var path = context.Request.Path.Value ?? "/";

        // Build and store UserContext for downstream controllers
        if (context.User?.Identity?.IsAuthenticated == true)
        {
            var userCtx = BuildUserContext(context, userIdStr, ip);

            // Resolve permitted sites from DB using the numeric UserId claim
            if (int.TryParse(userIdStr, out var numericUserId) && numericUserId > 0)
            {
                userCtx.PermittedSiteIds = await userSiteService.GetPermittedSiteIdsAsync(
                    numericUserId, context.RequestAborted);
            }

            context.Items["UserContext"] = userCtx;
        }

        try
        {
            await _next(context);
        }
        finally
        {
            sw.Stop();
            var statusCode = context.Response.StatusCode;

            _log.Information(
                "AUDIT | {Method} {Path} | User={UserId} | IP={Ip} | Status={Status} | Elapsed={ElapsedMs}ms",
                method, path, userIdStr, ip, statusCode, sw.ElapsedMilliseconds);
        }
    }

    private static UserContext BuildUserContext(HttpContext context, string userId, string ip)
    {
        var claims = context.User.Claims.ToList();

        var role = claims.FirstOrDefault(c => c.Type == "Role")?.Value
                ?? claims.FirstOrDefault(c => c.Type == "role")?.Value
                ?? claims.FirstOrDefault(c => c.Type == "http://schemas.microsoft.com/ws/2008/06/identity/claims/role")?.Value
                ?? "viewer";

        var displayName = claims.FirstOrDefault(c => c.Type == "UserName")?.Value
                       ?? claims.FirstOrDefault(c => c.Type == "name")?.Value
                       ?? claims.FirstOrDefault(c => c.Type == "preferred_username")?.Value
                       ?? userId;

        var email = claims.FirstOrDefault(c => c.Type == "Email")?.Value
                 ?? claims.FirstOrDefault(c => c.Type == "email")?.Value
                 ?? claims.FirstOrDefault(c => c.Type == "upn")?.Value
                 ?? string.Empty;

        // Category permissions based on role
        var permittedCategories = role switch
        {
            "admin" => new List<string> { "air_quality", "device_health", "admin" },
            "device_admin" => new List<string> { "air_quality", "device_health" },
            _ => new List<string> { "air_quality" }
        };

        // PermittedSiteIds will be populated after this by IUserSiteService
        return new UserContext
        {
            UserId = userId,
            DisplayName = displayName,
            Email = email,
            Role = role,
            PermittedSiteIds = new List<int>(),
            PermittedCategories = permittedCategories,
            IpAddress = ip
        };
    }
}
