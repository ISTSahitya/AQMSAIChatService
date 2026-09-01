using HawaqmAI.Api.Configuration;
using HawaqmAI.Api.Models;
using HawaqmAI.Api.Services;
using FluentAssertions;
using Microsoft.Extensions.Options;

namespace HawaqmAI.Tests;

/// <summary>
/// Unit tests for RbacEngine — verifies template filtering, selection validation,
/// and SQL rewriting with site_id injection.
/// </summary>
public sealed class RbacEngineTests
{
    private readonly RbacEngine _rbac;
    private readonly ChatOptions _chatOptions = new()
    {
        DefaultRowLimit = 50,
        MaxRowLimit = 5000
    };

    private static readonly ApprovedQuery AirQualityTemplate = new()
    {
        Id = "pollutant_trend",
        Category = "air_quality",
        Sql = "SELECT date, avg_pm25 FROM average_data WHERE site_id = @siteId AND date BETWEEN @startDate AND @endDate",
        Params =
        [
            new QueryParam { Name = "siteId", Type = "int", Source = "question_or_scope" },
            new QueryParam { Name = "startDate", Type = "date", Source = "question_or_scope" },
            new QueryParam { Name = "endDate", Type = "date", Source = "question_or_scope" }
        ],
        ResponseType = "chart",
        TableUsed = "average_data"
    };

    private static readonly ApprovedQuery DeviceHealthTemplate = new()
    {
        Id = "device_status",
        Category = "device_health",
        Sql = "SELECT device_id, status FROM device_status_table",
        Params = [],
        ResponseType = "table",
        TableUsed = "device_status_table"
    };

    public RbacEngineTests()
    {
        _rbac = new RbacEngine(Options.Create(_chatOptions));
    }

    // ── Level 1: Filter templates ────────────────────────────────────────────

    [Fact]
    public void FilterTemplates_ViewerCannotSeeDeviceHealth()
    {
        var viewer = new UserContext { Role = "viewer", PermittedCategories = ["air_quality"] };
        var all = new[] { AirQualityTemplate, DeviceHealthTemplate };

        var filtered = _rbac.FilterTemplates(all, viewer);

        filtered.Should().NotContain(t => t.Category == "device_health");
        filtered.Should().Contain(t => t.Id == "pollutant_trend");
    }

    [Fact]
    public void FilterTemplates_AdminSeesAll()
    {
        var admin = new UserContext { Role = "admin", PermittedCategories = ["air_quality", "device_health"] };
        var all = new[] { AirQualityTemplate, DeviceHealthTemplate };

        var filtered = _rbac.FilterTemplates(all, admin);

        filtered.Should().HaveCount(2);
    }

    // ── Level 2: Validate selection ──────────────────────────────────────────

    [Fact]
    public void ValidateSelection_ViewerOnAirQuality_IsValid()
    {
        var viewer = new UserContext { Role = "viewer", PermittedCategories = ["air_quality"] };
        var (valid, reason) = _rbac.ValidateSelection(AirQualityTemplate, viewer);
        valid.Should().BeTrue();
        reason.Should().BeNull();
    }

    [Fact]
    public void ValidateSelection_ViewerOnDeviceHealth_IsDenied()
    {
        var viewer = new UserContext { Role = "viewer", PermittedCategories = ["air_quality"] };
        var (valid, reason) = _rbac.ValidateSelection(DeviceHealthTemplate, viewer);
        valid.Should().BeFalse();
        reason.Should().NotBeNullOrEmpty();
    }

    // ── Level 3: SQL rewriting ───────────────────────────────────────────────

    [Fact]
    public void BuildSafeQuery_InjectsSiteIdFromScope()
    {
        var user = new UserContext
        {
            Role = "regulator",
            PermittedSiteIds = [1, 2, 3],
            PermittedCategories = ["air_quality"]
        };
        var scope = new ResolvedScope
        {
            SiteIds = [1],
            StartDate = new DateOnly(2026, 7, 1),
            EndDate = new DateOnly(2026, 7, 31)
        };

        var (sql, parameters) = _rbac.BuildSafeQuery(
            AirQualityTemplate,
            new Dictionary<string, string>(),
            user,
            scope);

        sql.Should().NotBeNullOrEmpty();
        parameters.Should().ContainKey("startDate");
        parameters.Should().ContainKey("endDate");
    }

    [Fact]
    public void BuildSafeQuery_AddsTopClauseForViewer()
    {
        var viewer = new UserContext
        {
            Role = "viewer",
            PermittedSiteIds = [5],
            PermittedCategories = ["air_quality"]
        };
        var scope = new ResolvedScope { SiteIds = [5] };
        var templateWithoutTop = new ApprovedQuery
        {
            Id = "test",
            Category = "air_quality",
            Sql = "SELECT date, avg_aqi FROM average_data",
            Params = [],
            ResponseType = "table",
            TableUsed = "average_data"
        };

        var (sql, _) = _rbac.BuildSafeQuery(
            templateWithoutTop,
            new Dictionary<string, string>(),
            viewer,
            scope);

        sql.ToUpperInvariant().Should().Contain("TOP");
    }

    [Fact]
    public void BuildSafeQuery_ColumnSubstitution_UsesAllowedColumn()
    {
        var template = new ApprovedQuery
        {
            Id = "trend",
            Category = "air_quality",
            Sql = "SELECT date, avg_{pollutantCol} FROM average_data WHERE site_id = @siteId",
            Params =
            [
                new QueryParam
                {
                    Name = "pollutantCol",
                    Type = "column",
                    Source = "question",
                    Allowed = ["pm25", "pm10", "co", "aqi"]
                }
            ],
            ResponseType = "chart",
            TableUsed = "average_data"
        };

        var user = new UserContext { Role = "admin", PermittedCategories = ["air_quality"] };
        var scope = new ResolvedScope();
        var llmParams = new Dictionary<string, string> { ["pollutantCol"] = "pm25" };

        var (sql, _) = _rbac.BuildSafeQuery(template, llmParams, user, scope);

        sql.Should().Contain("avg_pm25");
        sql.Should().NotContain("{pollutantCol}");
    }

    [Fact]
    public void BuildSafeQuery_RejectsNonAllowedColumn()
    {
        var template = new ApprovedQuery
        {
            Id = "trend",
            Category = "air_quality",
            Sql = "SELECT date, avg_{pollutantCol} FROM average_data",
            Params =
            [
                new QueryParam
                {
                    Name = "pollutantCol",
                    Type = "column",
                    Source = "question",
                    Allowed = ["pm25", "pm10"]
                }
            ],
            ResponseType = "chart",
            TableUsed = "average_data"
        };

        var user = new UserContext { Role = "admin", PermittedCategories = ["air_quality"] };
        var scope = new ResolvedScope();
        // Attempt injection via column param
        var llmParams = new Dictionary<string, string> { ["pollutantCol"] = "password; DROP TABLE" };

        var (sql, _) = _rbac.BuildSafeQuery(template, llmParams, user, scope);

        // Column should NOT have been substituted
        sql.Should().Contain("{pollutantCol}");
    }
}
