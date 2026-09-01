using HawaqmAI.Api.Configuration;
using HawaqmAI.Api.Models;
using HawaqmAI.Api.Services;
using FluentAssertions;
using Microsoft.Extensions.Caching.Memory;
using Microsoft.Extensions.Options;
using System.Reflection;

namespace HawaqmAI.Tests;

/// <summary>
/// Unit tests for QueryRouterService — verifies template matching logic.
/// </summary>
public sealed class QueryRouterTests
{
    private readonly UserContext _adminUser = new()
    {
        UserId = "test-admin",
        Role = "admin",
        PermittedCategories = ["air_quality", "device_health"]
    };

    private readonly UserContext _viewerUser = new()
    {
        UserId = "test-viewer",
        Role = "viewer",
        PermittedCategories = ["air_quality"]
    };

    private readonly QueryRouterService _router;

    public QueryRouterTests()
    {
        var cache = new MemoryCache(Options.Create(new MemoryCacheOptions()));
        var opts = Options.Create(new ChatOptions
        {
            HighConfidenceThreshold = 0.85,
            LowConfidenceThreshold = 0.50
        });
        _router = new QueryRouterService(cache, opts);
    }

    [Theory]
    [InlineData("What is the current AQI at site 5?")]
    [InlineData("What is the air quality right now?")]
    [InlineData("Show me the latest AQI for site 12")]
    public async Task CurrentAqiQuestion_MatchesCurrentAqiTemplate(string question)
    {
        var result = await _router.RouteAsync(question, _adminUser);
        result.Should().NotBeNull();
        result.NeedsClarification.Should().BeFalse();
    }

    [Theory]
    [InlineData("compare site 1 vs site 2 for PM2.5")]
    [InlineData("what is the difference between site 3 and site 7")]
    public async Task CompareQuestion_DoesNotReturnClarification(string question)
    {
        var result = await _router.RouteAsync(question, _adminUser);
        result.NeedsClarification.Should().BeFalse();
    }

    [Fact]
    public async Task CompletelyUnrelatedQuestion_ReturnsClarification()
    {
        var result = await _router.RouteAsync("What is the meaning of life?", _adminUser);
        // Score will be very low — expect clarification
        result.NeedsClarification.Should().BeTrue();
    }

    [Fact]
    public async Task ViewerUser_CanAccessAirQualityTemplates()
    {
        var result = await _router.RouteAsync("show me PM2.5 trend", _viewerUser);
        result.NeedsClarification.Should().BeFalse();
    }

    [Fact]
    public void GetPermittedTemplates_FiltersByCategory()
    {
        var viewerTemplates = _router.GetPermittedTemplates(_viewerUser);
        viewerTemplates.Should().AllSatisfy(t =>
            t.Category.Should().Be("air_quality"));
    }

    [Fact]
    public void GetPermittedTemplates_AdminGetsAll()
    {
        var adminTemplates = _router.GetPermittedTemplates(_adminUser);
        adminTemplates.Should().NotBeEmpty();
    }
}
