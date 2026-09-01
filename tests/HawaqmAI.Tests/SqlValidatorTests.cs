using HawaqmAI.Api.Validators;
using FluentAssertions;

namespace HawaqmAI.Tests;

/// <summary>
/// Unit tests for SqlValidator — ensures only safe SELECT queries are permitted.
/// </summary>
public sealed class SqlValidatorTests
{
    // ── Valid queries ────────────────────────────────────────────────────────

    [Fact]
    public void ValidSelect_ReturnsValid()
    {
        var sql = "SELECT TOP 50 site_id, avg_aqi FROM average_data WHERE site_id = @siteId";
        var (valid, error) = SqlValidator.Validate(sql);
        valid.Should().BeTrue();
        error.Should().BeNull();
    }

    [Fact]
    public void ValidSelectWithJoin_ReturnsValid()
    {
        var sql = "SELECT a.site_id, a.avg_pm25 FROM average_data a WHERE a.date BETWEEN @start AND @end";
        var (valid, _) = SqlValidator.Validate(sql);
        valid.Should().BeTrue();
    }

    [Fact]
    public void ValidSelectFromHistoryData_ReturnsValid()
    {
        var sql = "SELECT device_id, pm25, timestamp FROM history_data WHERE timestamp > @date";
        var (valid, _) = SqlValidator.Validate(sql);
        valid.Should().BeTrue();
    }

    // ── Forbidden DML ────────────────────────────────────────────────────────

    [Theory]
    [InlineData("INSERT INTO live_data VALUES (1, 2, 3)")]
    [InlineData("UPDATE average_data SET avg_aqi = 0")]
    [InlineData("DELETE FROM history_data")]
    [InlineData("DROP TABLE live_data")]
    [InlineData("ALTER TABLE average_data ADD new_col INT")]
    [InlineData("TRUNCATE TABLE live_data")]
    [InlineData("EXEC sp_executesql N'DROP TABLE live_data'")]
    public void ForbiddenKeyword_ReturnsInvalid(string sql)
    {
        var (valid, error) = SqlValidator.Validate(sql);
        valid.Should().BeFalse();
        error.Should().NotBeNullOrEmpty();
    }

    // ── Non-SELECT start ─────────────────────────────────────────────────────

    [Fact]
    public void NonSelectStatement_ReturnsInvalid()
    {
        var sql = "MERGE INTO average_data USING source ON 1=1 WHEN MATCHED THEN UPDATE SET avg_aqi = 0";
        var (valid, error) = SqlValidator.Validate(sql);
        valid.Should().BeFalse();
        error.Should().Contain("SELECT");
    }

    // ── Table allowlist ──────────────────────────────────────────────────────

    [Fact]
    public void QueryFromNonAllowedTable_ReturnsInvalid()
    {
        var sql = "SELECT * FROM sys.tables";
        var (valid, error) = SqlValidator.Validate(sql);
        valid.Should().BeFalse();
        error.Should().ContainAny("sys.tables", "not permitted");
    }

    [Fact]
    public void QueryFromAllowedTableWithNonAllowedJoin_ReturnsInvalid()
    {
        var sql = "SELECT l.pm25, u.password FROM live_data l JOIN users u ON 1=1";
        var (valid, error) = SqlValidator.Validate(sql);
        valid.Should().BeFalse();
    }

    // ── Empty input ──────────────────────────────────────────────────────────

    [Fact]
    public void EmptySql_ReturnsInvalid()
    {
        var (valid, error) = SqlValidator.Validate(string.Empty);
        valid.Should().BeFalse();
        error.Should().Contain("empty");
    }

    [Fact]
    public void WhitespaceSql_ReturnsInvalid()
    {
        var (valid, error) = SqlValidator.Validate("   ");
        valid.Should().BeFalse();
    }

    // ── Comment-hidden injection ─────────────────────────────────────────────

    [Fact]
    public void CommentHiddenDrop_ReturnsInvalid()
    {
        var sql = "SELECT * FROM live_data; /* DROP TABLE live_data */";
        var (valid, error) = SqlValidator.Validate(sql);
        valid.Should().BeFalse();
    }

    // ── EnsureTopClause ──────────────────────────────────────────────────────

    [Fact]
    public void EnsureTopClause_AddsTopWhenMissing()
    {
        var sql = "SELECT site_id FROM average_data WHERE site_id = 1";
        var result = SqlValidator.EnsureTopClause(sql, 50);
        result.Should().Contain("TOP 50");
    }

    [Fact]
    public void EnsureTopClause_DoesNotAddWhenTopPresent()
    {
        var sql = "SELECT TOP 100 site_id FROM average_data";
        var result = SqlValidator.EnsureTopClause(sql, 50);
        // Should not add another TOP
        result.Should().NotContain("TOP 50 SELECT TOP 100");
        result.Should().Contain("TOP 100");
    }

    // ── Column allowlist ─────────────────────────────────────────────────────

    [Theory]
    [InlineData("pm25", true)]
    [InlineData("PM25", true)]   // case-insensitive
    [InlineData("aqi", true)]
    [InlineData("password", false)]
    [InlineData("user_id", false)]
    [InlineData("'; DROP TABLE--", false)]
    public void IsAllowedColumn_ValidatesCorrectly(string column, bool expected)
    {
        var allowlist = new[] { "pm25", "pm10", "co", "aqi", "no2" };
        SqlValidator.IsAllowedColumn(column, allowlist).Should().Be(expected);
    }
}
