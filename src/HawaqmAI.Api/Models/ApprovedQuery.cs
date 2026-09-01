namespace HawaqmAI.Api.Models;

/// <summary>
/// An approved query template loaded from Knowledge/approved-queries.json.
/// The LLM selects from these — it never writes its own SQL.
/// </summary>
public sealed class ApprovedQuery
{
    /// <summary>Unique template identifier (e.g. "current_aqi").</summary>
    public string Id { get; set; } = string.Empty;

    /// <summary>Natural language patterns used for keyword matching.</summary>
    public List<string> Patterns { get; set; } = [];

    /// <summary>Category for RBAC filtering (e.g. "air_quality", "device_health").</summary>
    public string Category { get; set; } = string.Empty;

    /// <summary>
    /// Parameterised SQL template. Parameters use @paramName convention.
    /// Column substitutions use {columnName} notation.
    /// </summary>
    public string Sql { get; set; } = string.Empty;

    /// <summary>Parameter definitions for filling the SQL template.</summary>
    public List<QueryParam> Params { get; set; } = [];

    /// <summary>Response rendering type: "text", "table", or "chart".</summary>
    public string ResponseType { get; set; } = "text";

    /// <summary>Chart type when ResponseType == "chart".</summary>
    public string? ChartType { get; set; }

    /// <summary>Primary table used (for data source metadata).</summary>
    public string TableUsed { get; set; } = string.Empty;

    /// <summary>Optional human-readable description shown in debug/Swagger.</summary>
    public string? Description { get; set; }

    /// <summary>Minimum role required to use this template.</summary>
    public string RequiredRole { get; set; } = "viewer";

    /// <summary>
    /// When true, RBAC will NOT inject a site-ID filter on this query.
    /// Use for administrative listing queries (list stations, sites by region/sector)
    /// where the region/sector filter is the relevant scope constraint.
    /// </summary>
    public bool SkipSiteFilter { get; set; } = false;

    /// <summary>
    /// When true, this template is answered directly from faq.json without SQL or API calls.
    /// </summary>
    public bool IsFaq { get; set; } = false;

    /// <summary>
    /// When set, this template calls an external API instead of executing SQL.
    /// The SQL field is ignored when ApiCall is present.
    /// </summary>
    public ApiCallDefinition? ApiCall { get; set; }
}

/// <summary>Defines an external API call for a template instead of SQL execution.</summary>
public sealed class ApiCallDefinition
{
    /// <summary>HTTP method: "GET" or "POST".</summary>
    public string Method { get; set; } = "GET";

    /// <summary>
    /// API path relative to AqmsApi:BaseUrl.
    /// Supports {paramName} placeholders filled from LLM parameters or scope.
    /// Example: "/api/AirQuality/GetSiteData?siteId={siteId}"
    /// </summary>
    public string Path { get; set; } = string.Empty;

    /// <summary>
    /// How to flatten the API response into rows for the formatter.
    /// "devices" = one row per device (GetSiteData response shape).
    /// "sites"   = one row per site (GetAllSiteData response shape).
    /// </summary>
    public string ResponseShape { get; set; } = "devices";
}

/// <summary>A single parameter definition within an approved query template.</summary>
public sealed class QueryParam
{
    /// <summary>Parameter name matching the @paramName in the SQL.</summary>
    public string Name { get; set; } = string.Empty;

    /// <summary>Parameter type: "int", "date", "string", "column".</summary>
    public string Type { get; set; } = string.Empty;

    /// <summary>Where the value comes from: "question", "scope", "question_or_scope".</summary>
    public string Source { get; set; } = "question";

    /// <summary>
    /// Allowlist of permitted values (used for "column" type to prevent injection).
    /// </summary>
    public List<string>? Allowed { get; set; }

    /// <summary>Whether the parameter is optional.</summary>
    public bool Optional { get; set; } = false;
}

/// <summary>Root wrapper for approved-queries.json deserialization.</summary>
public sealed class ApprovedQueriesFile
{
    public List<ApprovedQuery> Queries { get; set; } = [];
}
