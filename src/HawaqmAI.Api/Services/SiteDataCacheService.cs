using System.Net.Http.Headers;
using System.Text.Json;
using HawaqmAI.Api.Configuration;
using Microsoft.Extensions.Options;
using Serilog;

namespace HawaqmAI.Api.Services;

/// <summary>
/// Background singleton that caches the GetAllSiteData API response.
///
/// Refresh strategy:
///   1. On each real user request ExternalApiService calls UpdateCache() with the live JSON —
///      so the cache is seeded from the first successful user call (no ServiceToken needed).
///   2. If AqmsApi:ServiceToken is configured the background loop also refreshes every 5 min
///      independently of user traffic, keeping the cache warm overnight or during idle periods.
///   3. If no ServiceToken the background loop is a no-op — the cache stays warm from user traffic.
/// </summary>
public sealed class SiteDataCacheService : BackgroundService, ISiteDataCacheService
{
    private static readonly Serilog.ILogger _log = Log.ForContext<SiteDataCacheService>();

    private readonly IHttpClientFactory _httpClientFactory;
    private readonly AqmsApiOptions _options;

    // Cache state
    private volatile string? _rawJson;
    private volatile string? _lastError;
    private int _refreshCount;
    private DateTime? _lastRefreshedAt;
    private DateTime? _nextRefreshAt;
    private readonly object _metaLock = new();

    private int _siteCount;
    private int _deviceCount;

    public SiteDataCacheService(
        IHttpClientFactory httpClientFactory,
        IOptions<AqmsApiOptions> options)
    {
        _httpClientFactory = httpClientFactory;
        _options = options.Value;
    }

    /// <inheritdoc/>
    public string? GetRawJson() => _rawJson;

    /// <inheritdoc/>
    public void UpdateCache(string json)
    {
        var (siteCount, deviceCount) = CountSitesAndDevices(json);
        _rawJson = json;
        lock (_metaLock)
        {
            _lastRefreshedAt = DateTime.UtcNow;
            _lastError       = null;
            _siteCount       = siteCount;
            _deviceCount     = deviceCount;
            _refreshCount    = Interlocked.Increment(ref _refreshCount);
        }
        _log.Debug("SiteDataCache: updated from user request — {Sites} sites, {Devices} devices", siteCount, deviceCount);
    }

    /// <inheritdoc/>
    public SiteDataCacheInfo GetCacheInfo()
    {
        lock (_metaLock)
        {
            return new SiteDataCacheInfo
            {
                LastRefreshedAt = _lastRefreshedAt,
                NextRefreshAt   = _nextRefreshAt,
                SiteCount       = _siteCount,
                DeviceCount     = _deviceCount,
                IsPopulated     = _rawJson != null,
                LastError       = _lastError,
                RefreshCount    = _refreshCount
            };
        }
    }

    /// <inheritdoc/>
    public async Task<bool> ForceRefreshAsync(CancellationToken ct = default)
    {
        _log.Information("SiteDataCache: manual force-refresh requested");
        return await RefreshAsync(ct);
    }

    // ── BackgroundService ────────────────────────────────────────────────────

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        if (string.IsNullOrWhiteSpace(_options.ServiceToken))
        {
            _log.Information("SiteDataCache: no ServiceToken configured — background refresh disabled. Cache will be seeded from user requests.");
            return;
        }

        _log.Information("SiteDataCache: background refresh started (interval={Min} min)", _options.CacheRefreshIntervalMinutes);

        // Initial populate
        await RefreshAsync(stoppingToken);

        while (!stoppingToken.IsCancellationRequested)
        {
            var delay = TimeSpan.FromMinutes(_options.CacheRefreshIntervalMinutes);
            lock (_metaLock) { _nextRefreshAt = DateTime.UtcNow.Add(delay); }

            try { await Task.Delay(delay, stoppingToken); }
            catch (OperationCanceledException) { break; }

            await RefreshAsync(stoppingToken);
        }

        _log.Information("SiteDataCache: background refresh stopped");
    }

    // ── Private helpers ──────────────────────────────────────────────────────

    private async Task<bool> RefreshAsync(CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(_options.BaseUrl))
        {
            _log.Warning("SiteDataCache: AqmsApi:BaseUrl is not configured — skipping refresh");
            lock (_metaLock) { _lastError = "AqmsApi:BaseUrl not configured"; }
            return false;
        }

        var url = _options.BaseUrl.TrimEnd('/') + "/api/AirQuality/GetAllSiteData";
        _log.Debug("SiteDataCache: refreshing from {Url}", url);

        try
        {
            var http = _httpClientFactory.CreateClient("AqmsApi");
            using var req = new HttpRequestMessage(HttpMethod.Get, url);
            req.Headers.Authorization = new AuthenticationHeaderValue("Bearer", _options.ServiceToken);

            using var resp = await http.SendAsync(req, ct);
            if (!resp.IsSuccessStatusCode)
            {
                var body = await resp.Content.ReadAsStringAsync(ct);
                _log.Warning("SiteDataCache: refresh failed — HTTP {Status}: {Body}",
                    (int)resp.StatusCode, body[..Math.Min(300, body.Length)]);
                lock (_metaLock) { _lastError = $"HTTP {(int)resp.StatusCode}: {resp.ReasonPhrase}"; }
                return false;
            }

            var json = await resp.Content.ReadAsStringAsync(ct);
            UpdateCache(json);
            _log.Information("SiteDataCache: background refresh complete — {Sites} sites, {Devices} devices",
                _siteCount, _deviceCount);
            return true;
        }
        catch (OperationCanceledException)
        {
            _log.Information("SiteDataCache: refresh cancelled (shutdown)");
            return false;
        }
        catch (Exception ex)
        {
            _log.Error(ex, "SiteDataCache: refresh threw an exception");
            lock (_metaLock) { _lastError = ex.Message; }
            return false;
        }
    }

    private static (int sites, int devices) CountSitesAndDevices(string json)
    {
        try
        {
            using var doc = JsonDocument.Parse(json);
            if (doc.RootElement.ValueKind != JsonValueKind.Array) return (0, 0);
            var deviceCount = doc.RootElement.GetArrayLength();
            var siteCount = doc.RootElement.EnumerateArray()
                .Select(el => el.TryGetProperty("StationId", out var sid) ? sid.GetInt32() : 0)
                .Distinct().Count(id => id != 0);
            return (siteCount, deviceCount);
        }
        catch { return (0, 0); }
    }
}
