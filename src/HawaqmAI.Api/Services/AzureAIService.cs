using System.Security.Cryptography.X509Certificates;
using System.Text;
using System.Text.Json;
using Azure.AI.OpenAI;
using Azure.Identity;
using HawaqmAI.Api.Configuration;
using HawaqmAI.Api.Models;
using Microsoft.Extensions.Options;
using OpenAI.Chat;
using OaiChatMessage = OpenAI.Chat.ChatMessage;
using Serilog;

namespace HawaqmAI.Api.Services;

/// <summary>
/// Communicates with Azure AI Foundry (OpenAI-compatible endpoint).
/// Authenticates via certificate credential (production) or API key (dev).
/// </summary>
public sealed class AzureAIService : IAzureAIService
{
    private static readonly Serilog.ILogger _log = Log.ForContext<AzureAIService>();
    private readonly AzureAIOptions _options;
    private readonly ChatClient _chatClient;

    public AzureAIService(IOptions<AzureAIOptions> options)
    {
        _options = options.Value;
        _chatClient = BuildChatClient();
    }

    /// <inheritdoc/>
    public async Task<LlmSelectionResult> SelectTemplateAsync(
        string question,
        IReadOnlyList<ApprovedQuery> candidates,
        IReadOnlyList<ConversationTurn> history,
        ResolvedScope scope,
        string? previousTemplateId = null,
        CancellationToken ct = default)
    {
        var systemPrompt = BuildSelectionSystemPrompt(candidates, previousTemplateId);
        var userPrompt = BuildSelectionUserPrompt(question, scope);
        // Don't pass history when candidates include device_last_reading — history causes
        // the LLM to pick a device name from a previous turn instead of the current question
        var hasDeviceTemplate = candidates.Any(c => c.Id == "device_last_reading");
        var messages = BuildMessages(systemPrompt, history, userPrompt, includeHistory: !hasDeviceTemplate);

        var response = await CallWithRetryAsync(messages, ct);
        return ParseSelectionResponse(response, candidates);
    }

    /// <inheritdoc/>
    public async Task<LlmParameterResult> FillParametersAsync(
        string question,
        ApprovedQuery template,
        IReadOnlyList<ConversationTurn> history,
        ResolvedScope scope,
        CancellationToken ct = default)
    {
        var systemPrompt = BuildParameterSystemPrompt(template);
        var userPrompt = BuildParameterUserPrompt(question, scope);
        // Don't include history when the template extracts an entity name (deviceName/stationName)
        // directly stated in the current question — history causes the LLM to pick the wrong entity
        var hasEntityParam = template.Params.Any(p =>
            p.Name.Equals("deviceName", StringComparison.OrdinalIgnoreCase) ||
            p.Name.Equals("stationName", StringComparison.OrdinalIgnoreCase));
        var messages = BuildMessages(systemPrompt, history, userPrompt, includeHistory: !hasEntityParam);

        var response = await CallWithRetryAsync(messages, ct);
        return ParseParameterResponse(response);
    }

    // ── Format templates keyed by template ID ───────────────────────────────
    // These match the exact Q&A format patterns specified by users in response-formats.json.
    // Placeholder syntax: {ColumnName} is replaced by the LLM from the actual data values.
    private static readonly Dictionary<string, string> _formatTemplates = new(StringComparer.OrdinalIgnoreCase)
    {
        // Data Success Rate from Compliancestatas API
        ["data_success_rate"] = """
            Data columns: "DataSuccessRate" (%), "ActiveDevices", "InactiveDevices", "TotalDevices", "AQI", "AQICategory", "CriticalSiteCount".
            The minimum target for Data Success Rate is 75%.

            If user asked specifically about Data Success Rate / DSR:
            FORMAT: "The current Data Success Rate for {year} is {DataSuccessRate}%, which is {above/below} the 75% minimum target. {ActiveDevices} out of {TotalDevices} devices are currently active."
            Example: "The current Data Success Rate for 2026 is 87.75%, which is above the 75% minimum target. 32 out of 38 devices are currently active."

            If user asked about active device count:
            FORMAT: "There are currently {ActiveDevices} active devices out of {TotalDevices} total devices ({InactiveDevices} inactive)."

            If user asked about critical sites / priority hotspots:
            FORMAT: "There are {CriticalSiteCount} priority hotspot site(s) with critical AQI levels for {year}."

            If user asked a general question covering all metrics:
            Output all available values in plain text. Include DSR %, active devices, AQI, and critical site count.

            NEVER output placeholder text — always use actual values from the data.
            If DataSuccessRate is null: "Data Success Rate information is not available at this time."
            """,

        // AQI category filter — columns: SiteName, RegionName, AQI, AQICategory, LastUpdated
        ["site_aqi_by_category"] = """
            Data columns: "SiteName", "RegionName", "AQI", "AQICategory", "LastUpdated".
            AQI categories: 0–50 Good, 51–100 Moderate, 101–150 Unhealthy for Sensitive Groups, 151–200 Unhealthy, 201–300 Very Unhealthy, 301+ Hazardous.

            If a SPECIFIC category was requested (e.g. "critical", "unhealthy", "good"):
            Line 1: "There are {N} site(s) with {AQICategory} air quality right now."
            Then list each site: "- {SiteName} ({RegionName}): AQI {AQI} — {AQICategory}"
            If no sites match: "No sites currently have {aqiCategory} air quality."

            For "critical" or "hotspot" (AQI > 100 — any of: Unhealthy for Sensitive Groups, Unhealthy, Very Unhealthy, Hazardous):
            Line 1: "There are {N} air quality hotspot(s) right now (AQI above 100):"
            Then list each site: "- {SiteName} ({RegionName}): AQI {AQI} — {AQICategory}"
            If empty: "No sites are currently above AQI 100. All monitored sites have acceptable air quality."

            If NO category was specified (all sites grouped):
            Line 1: "Current AQI status across your {N} accessible site(s):"
            Group sites by AQICategory. For each category present, show:
            "{AQICategory} ({count} site(s)): {SiteName1} ({AQI1}), {SiteName2} ({AQI2}), ..."
            Order groups from worst to best: Hazardous → Very Unhealthy → Unhealthy → Unhealthy for Sensitive Groups → Moderate → Good.

            NEVER output placeholder text — always use actual values from the data.
            """,

        // All sites AQI summary — columns: Site Name, AQI, AQI Category
        ["site_aqi_all"] = """
            Write one short intro sentence only — the table below will show the full data.
            FORMAT: "Here is the current AQI for your {N} accessible site(s). The table shows each site with its AQI value and category."
            If no data: "No AQI data is currently available for your sites."
            Do NOT list individual sites — the table handles that.
            """,

        // Site single — ONE specific value (AQI, one pollutant, or a named pair like temp+humidity)
        // responseType = "text" — returns a single sentence, no table.
        ["site_aqi_single"] = """
            Data has one row per parameter with columns: "Site Name", "Parameter", "Value", "Unit", "Last Updated".
            ALWAYS use ACTUAL values from the data. NEVER output placeholder text like {{VALUE}} or {{UNIT}}.
            If "Last Updated" is "N/A" or empty, omit the timestamp. If Unit is empty or null, omit it.
            Output ONE sentence of plain text only. No bullets. No markdown. No table.

            User asked for AQI or air quality category → find the row where Parameter = "AQI Index":
            FORMAT: "The current AQI at {Site Name} is {Value}, classified as {AQI category}, last updated at {Last Updated}."
            Example: "The current AQI at Al Saad Indian School is 72, classified as Moderate, last updated at 2026-09-15 18:45:00."

            User asked for ONE specific pollutant (CO2, PM2.5, PM10, CO, NO2, O3, TVOC) → find that parameter's row:
            FORMAT: "The latest {Parameter} reading at {Site Name} is {Value} {Unit}, recorded at {Last Updated}."
            Example: "The latest CO2 reading at Al Bateen School is 487 PPM, recorded at 2026-09-01 18:45:00."

            User asked for TWO or THREE specific parameters (e.g. temperature and humidity, NO2 and O3 and CO, PM10 and PM2.5):
            CRITICAL: For EACH parameter, find the row where the "Parameter" column EXACTLY matches the parameter name.
            PM2.5 value must come from the row where Parameter = "PM2.5". PM10 value must come from the row where Parameter = "PM10". Never swap or guess values.
            FORMAT: "At {Site Name} (as of {Last Updated}): {Parameter1} is {Value1} {Unit1}, {Parameter2} is {Value2} {Unit2}, and {Parameter3} is {Value3} {Unit3}."
            Example: "At Al Bateen School (as of 2026-09-15 18:45:00): Temperature is 28.4 °C, and Humidity is 61.2 %."
            Example: "At Al Saad Indian School (as of 2026-09-15 17:25:00): PM10 is 36.96 µg/m³, and PM2.5 is 42.82 µg/m³."

            AQI categories: 0–50 Good, 51–100 Moderate, 101–150 Unhealthy for Sensitive Groups, 151–200 Unhealthy, 201–300 Very Unhealthy, 301+ Hazardous.
            """,

        // Site all readings — ALL parameters returned as a table
        // responseType = "table" — summary sentence + full table displayed by frontend.
        ["site_readings_all"] = """
            Data has one row per parameter with columns: "Parameter", "Value", "Unit", "Last Updated".
            ALWAYS use ACTUAL values from the data. NEVER output placeholder text.
            Write ONE short intro sentence only — the table below will show all parameters.
            If "Last Updated" is "N/A" or empty, omit the timestamp. Output plain text only.

            FORMAT: "Here are the latest readings at {Site Name} as of {Last Updated}. The table below shows all monitored parameters."
            Example: "Here are the latest readings at Al Saad Indian School as of 2026-09-15 18:45:00. The table below shows all monitored parameters."
            Note: the table rows do NOT include a Site Name column — the site is named in the intro sentence above.

            If no data rows: "No current readings are available for that site. The device may be offline."
            """,

        // School / sector pollutant queries — data columns: Site Name, Region, Parameter, Value, Unit, Last Updated
        ["schools_latest_pollutant"] = """
            Data columns: "Site Name", "Region", "Parameter", "Value", "Unit", "Last Updated".
            FORMAT: "There are {Count} school site(s) with {Parameter} data."
            Then one line per row: "{Site Name} in {Region}: {Parameter} = {Value} {Unit} (last updated: {Last Updated})."
            If Parameter is "AQI Index", omit the Unit (do NOT show hPa or any unit for AQI).
            If no rows: "There are 0 school site(s) with {Parameter} data."
            Keep under 150 words. List every site from the data.
            """,

        ["sector_latest_pollutant"] = """
            Data columns: "Site Name", "Region", "Parameter", "Value", "Unit", "Last Updated".
            FORMAT: "There are {Count} {SectorName} site(s) with {Parameter} data."
            Then one line per row: "{Site Name} in {Region}: {Parameter} = {Value} {Unit} (last updated: {Last Updated})."
            If Parameter is "AQI Index", omit the Unit (do NOT show hPa or any unit for AQI).
            If no rows: "There are 0 {SectorName} site(s) with {Parameter} data in this region."
            Keep under 150 words. List every site from the data.
            """,

        // Devices filtered by region / sector (no specific site)
        ["devices_by_filter"] = """
            Data columns: "DeviceName", "StationName", "RegionName", "SectorName", "SubSectorName", "Status" (Active/Inactive), "LastActive", "AQI".
            Active = last reading within 15 minutes. Inactive = last reading older than 15 minutes.

            Line 1: state what was filtered and total counts.
            "There are {ActiveCount} active and {InactiveCount} inactive device(s) in {filter description}."

            Group by StationName. For each site:
            "{StationName} ({RegionName} — {SectorName}{SubSector}):"
            "  - {DeviceName} | {Status} | Last Active: {LastActive} | AQI: {AQI}"
            Only show SubSectorName if not null/empty: " / {SubSectorName}".
            Only show AQI if not null.

            If user asked only about active devices: show only Active rows.
            If user asked only about inactive devices: show only Inactive rows.
            If no rows: "No devices found matching the specified filter."
            NEVER output placeholder text — always use actual values from the data.
            """,

        // Device queries
        ["device_status"] = """
            Data columns: "DeviceName", "Status" (Active/Inactive), "LastMeasured".

            Status meanings:
            - Active   = last reading was within the past 15 minutes (device is online and sending data)
            - Inactive = last reading was more than 15 minutes ago (device is offline or has stopped sending)

            If user asked about active/inactive status:
            Active:   "Device {DeviceName} is currently Active. Its last reading was received at {LastMeasured}."
            Inactive: "Device {DeviceName} is currently Inactive. Its last reading was received at {LastMeasured}."

            If user asked when the device last sent data:
            FORMAT: "Device {DeviceName} last sent data at {LastMeasured}."

            If no rows returned: "No device found matching that name. Please check the device name and try again."
            NEVER output placeholder text — always use actual values from the data.
            """,

        ["device_last_reading"] = """
            FORMAT RULES for device readings:
            The data rows each represent one parameter reading. Each row has: DeviceName, ParameterName, ParameterValue, UnitName, LastMeasured.

            ONE row (user asked for a specific parameter like AQI, PM2.5, CO, etc.):
            "{DeviceName} — {ParameterName}: {ParameterValue} {UnitName} (Last measured: {LastMeasured})"
            If ParameterName is "AQI Index", append AQI category: e.g. "AQI Index: 58 — Moderate"

            MULTIPLE rows (user asked for all readings / reading / readings):
            Line 1: "{DeviceName} — Last measured: {LastMeasured}"  (use DeviceName and LastMeasured from the first row)
            Then output EVERY row as one line: "{ParameterName}: {ParameterValue} {UnitName}" — skip UnitName if empty or null.
            You MUST list ALL parameters from ALL rows. Do not stop after the first line.

            AQI categories: 0–50 Good, 51–100 Moderate, 101–150 Unhealthy for Sensitive Groups, 151–200 Unhealthy, 201–300 Very Unhealthy, 301+ Hazardous.
            No JSON, no bullets, no extra commentary.
            """,

        ["count_sites_total"] = """
            Data columns: "TotalSiteCount".
            FORMAT: "HAWAQM currently has {TotalSiteCount} registered sites."
            Example: "HAWAQM currently has 8 registered sites."
            NEVER output placeholder text — always use the actual count from the data.
            """,

        ["list_devices"] = """
            Data columns: "TotalDeviceCount".
            FORMAT: "HAWAQM currently has {TotalDeviceCount} registered devices."
            Example: "HAWAQM currently has 42 registered devices."
            NEVER output placeholder text — always use the actual count from the data.
            """,

        ["active_device_count"] = """
            Data columns: "ActiveDeviceCount", "TotalDeviceCount".
            FORMAT: "{ActiveDeviceCount} devices are currently active out of {TotalDeviceCount} registered devices."
            Example: "3 devices are currently active out of 10 registered devices."
            NEVER output placeholder text — always use actual values from the data.
            """,

        ["offline_devices"] = """
            Data columns: "DeviceName", "StationName", "LastActive".
            If no rows: "All devices are currently online."
            If rows exist:
            Line 1: "There are {N} offline device(s). The offline devices are:"
            Then one line per device: "- {DeviceName} at {StationName} (last active: {LastActive})"
            List EVERY device from the data. Do not skip any.
            NEVER output placeholder text — always use actual values from the data.
            """,

        ["online_devices"] = """
            Data columns: "DeviceName", "StationName", "LastActive".
            If no rows: "No devices are currently online."
            If rows exist:
            Line 1: "There are {N} online device(s):"
            Then one line per device: "- {DeviceName} at {StationName}"
            List EVERY device from the data. Do not skip any.
            NEVER output placeholder text — always use actual values from the data.
            """,

        ["devices_with_readings"] = """
            Data columns: "DeviceName", "StationName", "RegionName", "SectorName", "SubSectorName", "Status" (Active/Inactive), "LastActive", "AQI".
            Active = last reading within 15 minutes. Inactive = last reading older than 15 minutes.

            Line 1: state what was filtered and total count.
            FORMAT: "There are {N} device(s) under {filter description}."

            Then list every device, one per line:
            "- {DeviceName} | {StationName} | {RegionName} | {SectorName}{SubSector} | Status: {Status} | Last Active: {LastActive} | AQI: {AQI}"
            Only include SubSectorName if it is not null/empty: " ({SubSectorName})".
            Only include AQI if it is not null.

            If results span multiple sites, group by StationName:
            "{StationName} ({RegionName} — {SectorName}):"
            "  - {DeviceName} | Status: {Status} | Last Active: {LastActive} | AQI: {AQI}"

            If no rows: "No devices found matching the specified filter. Please check the site, region, or sector name."
            NEVER output placeholder text — always use actual values from the data.
            """,

        ["list_devices_at_station"] = """
            Data columns: "DeviceName", "StationName".
            FORMAT: "{StationName} has {N} registered device(s): {DeviceList}."
            Replace {StationName} with the actual station name from the first row.
            Replace {N} with the total count of rows.
            Replace {DeviceList} with a comma-separated list of every DeviceName from ALL rows.
            Example: "Al Bateen School has 3 registered device(s): BA 0010, BA 0011, SEI100M 0114."
            If no rows: "No devices found for that site. Please check the site name and try again."
            NEVER output placeholder text — always use actual values from the data.
            """,

        // Site count queries
        ["list_stations"] = """
            Data columns: "StationName", "RegionName".
            FORMAT: "You have access to {N} site(s): {StationName1}, {StationName2}, ..."
            If sites span multiple regions, group by region: "You have access to {N} site(s) — {RegionName1}: {names}, {RegionName2}: {names}."
            List every site from the data by name. Never just give a count. No table, plain text only.
            NEVER output placeholder text — always use actual values from the data.
            """,

        ["count_sites_in_region"] = """
            If the data is empty: "You do not have access to any sites in {RegionName}. Contact your administrator if you need access."
            If data has rows: "You have access to {N} site(s) in {RegionName}: {StationName1}, {StationName2}, ..."
            Never output blank placeholders. Always use the actual region name from the question.
            """,

        ["sites_count_by_region"] = """
            HAWAQM has 3 regions in total: Abu Dhabi, Al Ain, and Al Dhafra.
            The data rows show only the regions the user has access to.
            FORMAT when user has access to all 3 regions: "HAWAQM has 3 regions — Abu Dhabi ({TotalSites1} sites, {ActiveSites1} active), Al Ain ({TotalSites2} sites, {ActiveSites2} active), and Al Dhafra ({TotalSites3} sites, {ActiveSites3} active)."
            FORMAT when user has access to fewer than 3 regions: "HAWAQM has 3 regions in total (Abu Dhabi, Al Ain, and Al Dhafra). Based on your access permissions, you can view data for: {RegionName} ({TotalSites} sites, {ActiveSites} active). Contact your administrator to request access to additional regions."
            Always list only the regions present in the data rows as the user's accessible regions.
            """,

        ["sites_count_by_sector"] = """
            Data columns: "SectorName", "TotalSites", "ActiveSites".
            FORMAT: "The current sector distribution is {SectorName1}: {TotalSites1}, {SectorName2}: {TotalSites2}, {SectorName3}: {TotalSites3}, and {SectorName4}: {TotalSites4}."
            Use the exact SectorName and TotalSites values from each row. List all sectors from the data in one sentence.
            Example: "The current sector distribution is Public & Govt-School: 2, Residential: 1, and Commercial: 4."
            NEVER output placeholder text — always use actual values from the data.
            """,

        ["sites_by_sector"] = """
            Data columns: "StationName", "OrganizationName", "SectorName", "SubSectorName", "RegionName", "Status".
            If no rows: "There are no {SectorName} sites found for your query. Contact your administrator if you need access."
            If rows exist and ALL are in one region: "There are {N} {SectorName} site(s) in {RegionName}: {StationName1}, {StationName2}, ..."
            If rows span multiple regions, group by region:
              "There are {N} {SectorName} site(s):
               {RegionName1}: {StationName1}, {StationName2}
               {RegionName2}: {StationName3}, {StationName4}"
            Always use the actual SectorName and RegionName values from the data. Never output blank placeholders.
            If SubSectorName is present and the user asked about a sub-sector, mention it in the intro sentence.
            """,

        ["sites_by_region"] = """
            Data columns: "StationName", "OrganizationName", "SectorName", "SubSectorName", "RegionName", "Status".
            If no rows: "There are no sites found in that region. Contact your administrator if you need access."
            If rows exist: "There are {N} site(s) in {RegionName}: {StationName1}, {StationName2}, ..."
            Always use the actual RegionName from the data rows. Never output blank placeholders.
            If sites span different sectors, you may add the sector in brackets after each name: "{StationName} ({SectorName})".
            """,

        // Regional AQI
        ["aqi_by_region"] = """
            FORMAT (single region): "The current overall AQI for the {RegionName} region is {AQI}, classified as {AQICategory} as of {LastUpdated}."
            FORMAT (all regions): "The current regional AQI values are Abudhabi: {AbuDhabiAQI} ({AbuDhabiLevel}), AL Ain: {AlAinAQI} ({AlAinLevel}), and AlDhafra: {AlDhafraAQI} ({AlDhafraLevel})."
            AQI categories: 0–50 Good, 51–100 Moderate, 101–150 Unhealthy for Sensitive Groups, 151–200 Unhealthy, 201–300 Very Unhealthy, 301+ Hazardous.
            Keep under 50 words.
            """,

        ["region_aqi_geographical"] = """
            Data columns: "RegionName", "AQI", "AQICategory", "ActiveStationsCount".
            These values are the same as shown on the HAWAQM Executive Dashboard map.
            AQI categories: 0–50 Good, 51–100 Moderate, 101–150 Unhealthy for Sensitive Groups, 151–200 Unhealthy, 201–300 Very Unhealthy, 301+ Hazardous.

            If one row (single region asked):
            FORMAT: "The current AQI for {RegionName} is {AQI} ({AQICategory})."
            Only append " based on {ActiveStationsCount} active site(s)" if ActiveStationsCount > 0.
            Example: "The current AQI for Abu Dhabi is 64 (Moderate)."

            If multiple rows (all regions — user asked for region-wise or all regions):
            Line 1: "Current AQI by region:"
            Then one line per region: "- {RegionName}: {AQI} ({AQICategory})"
            List ALL rows from the data. Do not skip any region.
            Example:
            "Current AQI by region:
            - Abu Dhabi: 64 (Moderate)
            - Al Ain: 64 (Moderate)
            - Al Dhafra: 64 (Moderate)"

            If no rows or AQI is null for a region, skip that region.
            If all rows are empty: "No AQI data is available for the requested region(s)."
            NEVER output placeholder text — always use actual values from the data.
            NEVER say "as of {timestamp}" — this data has no timestamp field.
            """,

        // AQI rankings
        ["lowest_aqi_today"] = """
            Data columns: "StationName", "RegionName", "MinAQI", "LastUpdated".
            FORMAT: "The lowest site AQI recorded today is {MinAQI} at {StationName}, {RegionName}, at {LastUpdated}. The AQI category is {AQICategory}."
            Example: "The lowest site AQI recorded today is 32 at Default Site, AL dhafra, at 2026-09-01 19:40:00. The AQI category is Good."
            AQI categories: 0–50 Good, 51–100 Moderate, 101–150 Unhealthy for Sensitive Groups, 151–200 Unhealthy, 201–300 Very Unhealthy, 301+ Hazardous.
            NEVER output placeholder text — always use actual values from the data.
            """,

        ["highest_aqi_today"] = """
            Data columns: "StationName", "RegionName", "MaxAQI", "LastUpdated".
            FORMAT: "The highest site AQI recorded today is {MaxAQI} at {StationName}, {RegionName}, at {LastUpdated}. The AQI category is {AQICategory}."
            Example: "The highest site AQI recorded today is 96 at Al Saad Indian School, Al Ain, at 2026-09-01 19:40:00. The AQI category is Moderate."
            AQI categories: 0–50 Good, 51–100 Moderate, 101–150 Unhealthy for Sensitive Groups, 151–200 Unhealthy, 201–300 Very Unhealthy, 301+ Hazardous.
            NEVER output placeholder text — always use actual values from the data.
            """,

        ["site_aqi_ranking"] = """
            Data columns: "StationName", "RegionName", "AQI", "LastUpdated".
            Line 1: "Current AQI values by site (lowest to highest):"
            Then one line per site: "- {StationName} ({RegionName}): AQI {AQI} — {AQICategory}"
            List EVERY site from the data. Do not skip any.
            AQI categories: 0–50 Good, 51–100 Moderate, 101–150 Unhealthy for Sensitive Groups, 151–200 Unhealthy, 201–300 Very Unhealthy, 301+ Hazardous.
            NEVER output placeholder text — always use actual values from the data.
            """,
    };

    /// <inheritdoc/>
    public async Task<string> SummarizeResultsAsync(
        string question,
        ApprovedQuery template,
        List<Dictionary<string, object?>> results,
        ResolvedScope scope,
        CancellationToken ct = default)
    {
        // Look up exact format template for this query type
        _formatTemplates.TryGetValue(template.Id, out var formatRule);

        var specificPrompt = formatRule != null
            ? $"""
              You are HAWAQM AI, an air quality analysis assistant for a UAE government monitoring platform.
              AQI categories: 0–50 Good, 51–100 Moderate, 101–150 Unhealthy for Sensitive Groups, 151–200 Unhealthy, 201–300 Very Unhealthy, 301+ Hazardous.

              Use the EXACT format rule below. Fill in placeholders with actual values from the data.
              Do NOT add extra sentences. Do NOT make recommendations. Output plain text only.

              {formatRule}
              """
            : null;

        var systemPrompt = specificPrompt ?? """
              You are HAWAQM AI, an air quality analysis assistant for a government monitoring platform.
              Summarize the query results based on what the user asked. Follow these FORMAT RULES strictly:

              COUNT / HOW MANY questions (e.g. "how many sites", "how many devices"):
              → One sentence: "There are {N} {thing}." Then list them if ≤ 10 items.

              LIST questions (e.g. "list sites", "show stations", "which devices"):
              → One intro sentence, then a comma-separated list of names. No bullets.

              SINGLE VALUE questions (e.g. "what is the AQI at X", "CO2 at site Y"):
              → One sentence: "{Site/Device} recorded {parameter}: {value} {unit} at {timestamp}."
              → If AQI: append the category "(Good / Moderate / Unhealthy for Sensitive Groups / Unhealthy / Very Unhealthy / Hazardous)".

              RANKING / TOP questions (e.g. "highest AQI", "worst site", "most polluted"):
              → One sentence stating the top result with value and location.

              TREND / AVERAGE questions (e.g. "average PM2.5", "trend over week"):
              → Two sentences max: average/peak value, then one observation about the trend.

              COMPARISON questions (e.g. "compare site A vs B"):
              → One sentence per site with its value, then one sentence on which is higher/lower.

              GENERAL / ALL DATA questions:
              → Two to three sentences covering the key values. Mention AQI category if present.

              RULES FOR ALL FORMATS:
              - Output plain text only — no markdown, no bullet points, no bold, no headers.
              - Never repeat column names from the data table.
              - Never say "the data shows" or "the query returned" — just state the facts.
              - Do NOT make recommendations. Only describe what the data shows.
              - Keep under 80 words.
              AQI categories: 0–50 Good, 51–100 Moderate, 101–150 Unhealthy for Sensitive Groups, 151–200 Unhealthy, 201–300 Very Unhealthy, 301+ Hazardous.
              """;

        var dataPreview = results.Count > 20
            ? results.Take(20).ToList()
            : results;

        var formatInstruction = formatRule != null
            ? "Apply the format rule from the system prompt exactly. Fill EVERY placeholder with the ACTUAL value from the data rows above. CRITICAL: match each parameter name EXACTLY to its row — the Value for PM2.5 must come from the row where Parameter = 'PM2.5', the Value for PM10 must come from the row where Parameter = 'PM10', etc. NEVER swap, guess, or reorder values. NEVER output placeholder text like {{VALUE}}, {{UNIT}}, or {{LAST_UPDATED}} — always replace them with real numbers and strings. Output plain text only."
            : "Apply the correct FORMAT RULE from the system prompt based on the user's question type. Fill all placeholders with actual data values. Match each parameter name EXACTLY to its data row — never swap values between parameters. Output plain text only.";

        var userPrompt = $"""
            User question: "{question}"
            Template used: {template.Id}
            Date range: {scope.DateRangeLabel}
            Total rows returned: {results.Count}
            Data (first {dataPreview.Count} rows):
            {JsonSerializer.Serialize(dataPreview, new JsonSerializerOptions { WriteIndented = false })}

            {formatInstruction}
            """;

        var messages = new List<OaiChatMessage>
        {
            OaiChatMessage.CreateSystemMessage(systemPrompt),
            OaiChatMessage.CreateUserMessage(userPrompt)
        };

        var (content, _, _) = await CallWithRetryAsync(messages, ct);
        _log.Information("SummarizeResultsAsync: template={Id} rows={Rows} summary_len={Len} summary='{Preview}'",
            template.Id, results.Count, content.Length, content[..Math.Min(200, content.Length)]);
        return content;
    }

    /// <inheritdoc/>
    public async Task<string> AnswerFaqAsync(
        string question,
        string faqContext,
        CancellationToken ct = default)
    {
        var systemPrompt = $"""
            You are HAWAQM AI, an air quality monitoring assistant for a UAE government platform.
            Answer the user's question using ONLY the information provided below. Do not invent facts.
            Be concise — one to three sentences maximum. Output plain text only.

            HAWAQM Knowledge Base:
            {faqContext}
            """;

        var messages = new List<OaiChatMessage>
        {
            OaiChatMessage.CreateSystemMessage(systemPrompt),
            OaiChatMessage.CreateUserMessage(question)
        };

        var (content, _, _) = await CallWithRetryAsync(messages, ct);
        return content;
    }

    // ── Private: client building ────────────────────────────────────────────

    private ChatClient BuildChatClient()
    {
        if (string.IsNullOrWhiteSpace(_options.Endpoint))
            throw new InvalidOperationException("AzureAI:Endpoint must be configured.");

        AzureOpenAIClient client;

        if (_options.AuthMethod == "certificate" && !string.IsNullOrWhiteSpace(_options.CertificateThumbprint))
        {
            _log.Information("Azure AI: using certificate credential (thumbprint: {T})", _options.CertificateThumbprint[..8] + "...");
            using var store = new X509Store(StoreName.My, StoreLocation.LocalMachine);
            store.Open(OpenFlags.ReadOnly);
            var certs = store.Certificates.Find(X509FindType.FindByThumbprint, _options.CertificateThumbprint, false);

            if (certs.Count == 0)
                throw new InvalidOperationException($"Certificate with thumbprint {_options.CertificateThumbprint} not found in LocalMachine\\My store.");

            var credential = new ClientCertificateCredential(_options.TenantId, _options.ClientId, certs[0]);
            client = new AzureOpenAIClient(new Uri(_options.Endpoint), credential);
        }
        else if (!string.IsNullOrWhiteSpace(_options.ApiKey))
        {
            _log.Information("Azure AI: using API key auth (development mode)");
            client = new AzureOpenAIClient(new Uri(_options.Endpoint), new System.ClientModel.ApiKeyCredential(_options.ApiKey));
        }
        else
        {
            _log.Information("Azure AI: using DefaultAzureCredential");
            client = new AzureOpenAIClient(new Uri(_options.Endpoint), new DefaultAzureCredential());
        }

        return client.GetChatClient(_options.DeploymentName);
    }

    // ── Private: LLM call ───────────────────────────────────────────────────

    private async Task<(string Content, int Tokens, string Model)> CallWithRetryAsync(
        List<OaiChatMessage> messages,
        CancellationToken ct,
        int attempt = 0)
    {
        try
        {
            var completionOptions = new ChatCompletionOptions();

            if (!_options.DeploymentName.StartsWith("gpt-5", StringComparison.OrdinalIgnoreCase))
            {
                completionOptions.MaxOutputTokenCount = _options.MaxTokens;
                completionOptions.Temperature = (float)_options.Temperature;
            }

            var completion = await _chatClient.CompleteChatAsync(messages, completionOptions, ct);

            var content = completion.Value.Content.FirstOrDefault()?.Text ?? string.Empty;
            var tokens = completion.Value.Usage?.TotalTokenCount ?? 0;
            var model = _options.DeploymentName;

            _log.Debug("Azure AI response: {Tokens} tokens, {Length} chars", tokens, content.Length);
            return (content, tokens, model);
        }
        catch (Exception ex) when (attempt < _options.MaxTokens && attempt < 2)
        {
            _log.Warning(ex, "Azure AI call failed (attempt {Attempt}), retrying...", attempt + 1);
            await Task.Delay(TimeSpan.FromSeconds(Math.Pow(2, attempt)), ct);
            return await CallWithRetryAsync(messages, ct, attempt + 1);
        }
    }

    // ── Private: prompt building ─────────────────────────────────────────────

    private static string BuildSelectionSystemPrompt(IReadOnlyList<ApprovedQuery> candidates, string? previousTemplateId = null)
    {
        var sb = new StringBuilder();
        sb.AppendLine("""
            You are HAWAQM AI, a query selector for an air quality monitoring system.
            Your ONLY job is to select the best approved query template for the user's question and fill its parameters.
            You MUST NOT write your own SQL. You MUST select from the provided templates.

            CRITICAL — template selection rules:

            PRIORITY RULE — site name beats region:
            If the user mentions a SPECIFIC site or station name (any name that is NOT just "Abu Dhabi", "Al Ain", or "Al Dhafra" alone), ALWAYS treat it as a site query regardless of whether a region is also mentioned.
            Examples where site name takes priority:
            - "AQI at Al Saad Indian School in Al Ain" → site query (stationName="Al Saad Indian School"), NOT a region query
            - "CO2 at Abu Dhabi Residential" → site query (stationName="Abu Dhabi Residential")
            - "readings at Al Ain Commercial Institutional" → site query
            - "AQI in Abu Dhabi" (no site name, only region) → region query

            1. If the user mentions a SPECIFIC site or station name AND asks for ONE value (AQI, AQI category, or a specific pollutant like CO2, PM2.5, NO2, temperature, humidity) → select "site_aqi_single". Returns a text sentence.
               When extracting stationName: use ONLY the site name itself — strip any trailing region qualifier like "in Al Ain", "in Abu Dhabi", "in Al Dhafra". Example: "Al Saad Indian School in Al Ain" → stationName = "Al Saad Indian School".
            2. If the user mentions a SPECIFIC site or station name AND asks for ALL readings, ALL parameters, ALL current data, or the last 24 hours of readings → select "site_readings_all". Returns a table.
               Same stationName rule: strip region qualifier from the extracted name.
            3. Only select "schools_latest_pollutant" when the user says "schools" generically (e.g. "all schools", "schools in Al Ain") with NO specific school or site name.
            4. If the user asks an AQI question that mentions ONLY a region name (Abu Dhabi, Abudhabi, Al Ain, Alain, Al Dhafra, Aldhafra) with NO specific site name, OR asks for "all regions", "by region", "region wise" → select "region_aqi_geographical". Extract regionName if one specific region is mentioned; omit it if all regions are requested. NEVER select region_aqi_geographical when a specific site name is also present — rule 1 takes priority.
            4b. If the user asks a generic AQI question with NO specific site name AND NO region (e.g. "show me the AQI", "what is the AQI", "show the AQI", "current AQI", "display AQI", "AQI at my sites", "AQI of my sites", "AQI for my sites", "current AQI in my sites", "what is the current AQI") → select "site_aqi_all". This returns AQI for ALL the user's accessible sites in a table. NEVER ask the user to clarify which site — just return site_aqi_all.
            5. If the user mentions BOTH a sector (Commercial, Residential, Public & Govt-School) AND a region (Abu Dhabi, Al Ain, Al Dhafra) → ALWAYS select "sites_by_sector". It supports both filters. NEVER select "sites_by_region" when a sector is also mentioned.
            6. Extract deviceName ONLY from the CURRENT question. Never use a device name from conversation history.
            7. Use the device name EXACTLY as written in the question (e.g. "BA0010", "BA0012", "SEI100M0010", "SEI100M0114").
            8. If the user asks about AQI hotspots, critical sites, or sites with a specific AQI quality level (good, moderate, unhealthy, hazardous, critical, healthy, safe, unsafe, poor, bad air quality) → select "site_aqi_by_category". If a specific level is mentioned, extract aqiCategory: map "critical"/"hotspot"→"critical", "good"/"healthy"/"safe"→"Good", "moderate"→"Moderate", "unhealthy for sensitive"→"Unhealthy for Sensitive Groups", "unhealthy"→"Unhealthy", "very unhealthy"→"Very Unhealthy", "hazardous"→"Hazardous". If no category is specified, omit aqiCategory to return all sites grouped.
            9. If the user asks for the current Data Success Rate, DSR value, data availability %, how many devices are active, active device count, or critical site count → select "data_success_rate". Extract year if mentioned (e.g. "2025", "2026"); omit year to default to the current year. NEVER select faq_answer for questions asking for the CURRENT or ACTUAL value of the DSR or active device count — those must come from the live API.
            10. If the user asks to LIST or SHOW devices under a SPECIFIC SITE or SCHOOL name (e.g. "devices under Al Naeem School", "devices at Al Saad Indian School and Al Bateen School") → select "devices_with_readings". Extract stationName as the site name or a keyword from it (e.g. "Al Naeem" or "Naeem"). If the user names MULTIPLE sites, join them with a space so the LIKE filter catches any partial match (e.g. stationName="Naeem Saad" won't work — instead extract the FIRST site name only; the user will see all devices grouped by site). NEVER select sites_by_sector or list_stations for device listing questions.
            If the user asks about devices in a REGION or SECTOR only (no specific site name) → select "devices_by_filter". Extract regionName and/or sectorName. This also handles "how many active devices in Abu Dhabi" — extract statusFilter if active/inactive is mentioned.

            Return ONLY valid JSON in this exact format:
            {
              "selectedQueryId": "<template id>",
              "parameters": {
                "<paramName>": "<value>"
              },
              "confidence": 0.95,
              "summary": "One sentence explaining what data will be retrieved"
            }
            """);

        if (!string.IsNullOrWhiteSpace(previousTemplateId))
        {
            sb.AppendLine($"""

            IMPORTANT CONTEXT: The user is asking a follow-up question referring to the previous result.
            The previous query used template "{previousTemplateId}".
            If the user is asking to sort, reorder, filter, or modify the previous result, prefer reusing "{previousTemplateId}" with updated or same parameters.
            Only switch to a different template if the user is clearly asking for completely different data.
            """);
        }

        sb.AppendLine("\nAvailable templates:");
        foreach (var t in candidates)
        {
            sb.AppendLine($"- id: \"{t.Id}\" | {t.Description} | params: {string.Join(", ", t.Params.Select(p => p.Name))}");
        }

        return sb.ToString();
    }

    private static string BuildParameterSystemPrompt(ApprovedQuery template)
    {
        var sb = new StringBuilder();
        sb.AppendLine($"""
            You are HAWAQM AI. The query template "{template.Id}" has been selected.
            Extract parameter values from the user's question.

            Template: {template.Description}
            Parameters needed:
            """);

        foreach (var p in template.Params)
        {
            var allowedStr = p.Allowed is { Count: > 0 } ? $" (allowed: {string.Join(", ", p.Allowed)})" : "";
            sb.AppendLine($"- {p.Name} (type: {p.Type}, source: {p.Source}){allowedStr}");
        }

        sb.AppendLine("""

            Return ONLY valid JSON:
            {
              "parameters": { "<paramName>": "<value>" },
              "confidence": 0.95
            }

            For dates: use ISO format YYYY-MM-DD.
            For columns: use lowercase snake_case (e.g. pm25, not PM2.5).
            For year: extract a 4-digit year if the user mentions one (e.g. "2025", "last year"). If no year is mentioned, omit it — the system will default to the current year automatically.
            For regionName: extract the region the user mentions (words like "in", "under", "for", "within", "at" before a region name all indicate the region). Map to: "Abu Dhabi" (also: abudhabi, abu-dhabi, abudabi), "Al Ain" (also: alain, al-ain), "Al Dhafra" (also: aldhafra, al-dhafra, dhafra). If the user asks about all regions or does not mention a specific region, omit regionName.
            For sectorName: there are exactly 3 sectors. Map ALL user variations to one of these three canonical values:
              → "Commercial": "commercial", "commercial sector", "commercial sites"
              → "Public & Govt-School": ANY phrase containing "school", "govt", "gov", "government", "public & govt", "public & gov", "public and gov", "public and govt", "public and government", "govt-school", "gov-school", "government school", "govt school", "gov school", "educational", "private and gov-school", "private and govt school"
              → "Residential": "residential", "residential sector", "residential sites"
            IMPORTANT: "public & Gov School", "Public and Government School", "Public & Government school", "gov school", "govt school" ALL map to "Public & Govt-School".
            If the user says something that doesn't match any of these three, omit sectorName.
            For parameterName: map common user words to the allowed parameter name — "co2", "carbon dioxide" → "CO2"; "pm2.5", "fine particles" → "PM2.5"; "pm10", "coarse particles" → "PM10"; "co", "carbon monoxide" → "CO"; "aqi", "air quality index" → "AQI Index"; "no2", "nitrogen dioxide" → "NO2"; "o3", "ozone" → "O3"; "tvoc", "voc" → "TVOC"; "ch2o", "formaldehyde" → "CH2O".
            For stationName: extract the site or station name the user mentions. CRITICAL — strip any region qualifier appended after the site name. Examples:
              - "Al Saad Indian School in Al Ain" → stationName = "Al Saad Indian School"
              - "Abu Dhabi Residential site" → stationName = "Abu Dhabi Residential"
              - "Al Ain Commercial Institutional" → stationName = "Al Ain Commercial Institutional" (Al Ain is part of the site name here, not a region qualifier)
              - "readings at Al Bateen School in Abu Dhabi" → stationName = "Al Bateen School"
            If the user says ONLY a region name ("Abu Dhabi", "Al Ain", "Al Dhafra") with no site name, do NOT extract stationName — that is a region query, not a site query.
            Never extract or guess a numeric ID — always use the name.
            For deviceName: extract the device name EXACTLY as stated in the CURRENT question only. Device names in this system follow patterns like "BA0010", "BA0012", "SEI1000003", "SEI100M0010", "SEI100M0114" (letters/prefix directly followed by 4-digit number, no space). Extract the full device name as-is. Never guess or invent a device name — only extract what is explicitly stated in the current question.
            For parameterName (device readings only): if the user asks for a specific parameter, map it — "aqi", "air quality index" → "AQI Index"; "pm2.5", "fine particles" → "PM2.5"; "pm10" → "PM10"; "co2", "carbon dioxide" → "CO2"; "co", "carbon monoxide" → "CO"; "no2" → "NO2"; "o3", "ozone" → "O3"; "temperature", "temp" → "Temperature"; "humidity" → "Humidity"; "voc", "tvoc" → "VOC"; "noise" → "Noise"; "ch2o", "formaldehyde" → "CH2O". If the user asks for "reading", "readings", "all readings", "all parameters", or "all data" — omit parameterName entirely.
            For statusFilter (devices_by_filter only): extract "Active" if user says "active", "online", "working", "sending"; extract "Inactive" if user says "inactive", "offline", "not sending", "down"; omit if user asks for all devices with no status preference.
            For aqiCategory (site_aqi_by_category only): map the user's words to one of: "Good" (good, healthy, safe, clean), "Moderate" (moderate), "Unhealthy for Sensitive Groups" (sensitive, sensitive groups, unhealthy for sensitive), "Unhealthy" (unhealthy, poor, bad), "Very Unhealthy" (very unhealthy, very poor), "Hazardous" (hazardous, dangerous, severe), "critical" (critical, hotspot, hotspots, worst, most polluted, above 100). If user says "all" or doesn't specify a category, omit aqiCategory.
            For minValue: extract a numeric lower bound when user says "more than X", "greater than X", "above X", "over X", "exceeding X". Use the number only (e.g. "more than 50" → minValue: "50").
            For maxValue: extract a numeric upper bound when user says "less than X", "below X", "under X", "at most X". Use the number only.
            IMPORTANT — pronoun resolution: if the user says "this site", "this station", "this location", "it", or similar pronouns without naming a site, look at the conversation history above to find the most recently mentioned site or station name, and use that as stationName.
            For deviceName pronoun resolution ONLY: if the user says "this device", "it", or similar pronouns WITHOUT explicitly naming a device in the current question, look at history for the most recently mentioned device name. But if the current question contains an explicit device name, ALWAYS use that — never the history device.
            If a parameter can't be extracted even from history, omit it from the JSON (don't guess).
            """);

        return sb.ToString();
    }

    private static string BuildSelectionUserPrompt(string question, ResolvedScope scope)
    {
        // Do NOT list permitted site IDs here — the LLM should extract siteId ONLY
        // when the user explicitly states a number in their question (e.g. "site 17").
        // Showing permitted IDs causes the LLM to guess IDs from the list instead of
        // extracting station names, leading to wrong site matches.
        var dateStr = scope.StartDate.HasValue
            ? $"Date range from scope: {scope.StartDate:yyyy-MM-dd} to {scope.EndDate:yyyy-MM-dd}"
            : "No date range in scope";

        var regionStr = !string.IsNullOrWhiteSpace(scope.Region)
            ? $"Region filter from scope bar: {scope.Region}"
            : "No region filter (all regions)";

        return $"Question: {question}\n{dateStr}\n{regionStr}";
    }

    private static string BuildParameterUserPrompt(string question, ResolvedScope scope)
        => BuildSelectionUserPrompt(question, scope);

    private static List<OaiChatMessage> BuildMessages(
        string systemPrompt,
        IReadOnlyList<ConversationTurn> history,
        string userPrompt,
        bool includeHistory = true)
    {
        var messages = new List<OaiChatMessage>
        {
            OaiChatMessage.CreateSystemMessage(systemPrompt)
        };

        if (includeHistory)
        {
            foreach (var turn in history.TakeLast(15))
            {
                messages.Add(turn.Role == "user"
                    ? OaiChatMessage.CreateUserMessage(turn.Content)
                    : OaiChatMessage.CreateAssistantMessage(turn.Content));
            }
        }

        messages.Add(OaiChatMessage.CreateUserMessage(userPrompt));
        return messages;
    }

    // ── Private: response parsing ────────────────────────────────────────────

    private static LlmSelectionResult ParseSelectionResponse(
        (string Content, int Tokens, string Model) response,
        IReadOnlyList<ApprovedQuery> candidates)
    {
        try
        {
            var json = ExtractJson(response.Content);
            using var doc = JsonDocument.Parse(json);
            var root = doc.RootElement;

            var id = root.GetProperty("selectedQueryId").GetString() ?? string.Empty;
            var confidence = root.TryGetProperty("confidence", out var confEl) ? confEl.GetDouble() : 0.8;
            var summary = root.TryGetProperty("summary", out var sumEl) ? sumEl.GetString() ?? string.Empty : string.Empty;
            var parameters = ParseParameters(root);

            // Validate the selected ID is actually in the candidate list
            if (!candidates.Any(c => c.Id == id))
            {
                _log.Warning("LLM selected unknown query id '{Id}', falling back to first candidate", id);
                id = candidates.FirstOrDefault()?.Id ?? string.Empty;
            }

            return new LlmSelectionResult
            {
                SelectedQueryId = id,
                Parameters = parameters,
                Confidence = confidence,
                Summary = summary,
                TokensUsed = response.Tokens,
                ModelUsed = response.Model
            };
        }
        catch (Exception ex)
        {
            _log.Error(ex, "Failed to parse LLM selection response: {Content}", response.Content[..Math.Min(500, response.Content.Length)]);
            return new LlmSelectionResult { SelectedQueryId = string.Empty, TokensUsed = response.Tokens };
        }
    }

    private static LlmParameterResult ParseParameterResponse(
        (string Content, int Tokens, string Model) response)
    {
        try
        {
            var json = ExtractJson(response.Content);
            using var doc = JsonDocument.Parse(json);
            var root = doc.RootElement;
            var confidence = root.TryGetProperty("confidence", out var confEl) ? confEl.GetDouble() : 0.8;
            var parameters = ParseParameters(root);

            return new LlmParameterResult
            {
                Parameters = parameters,
                Confidence = confidence,
                TokensUsed = response.Tokens,
                ModelUsed = response.Model
            };
        }
        catch (Exception ex)
        {
            _log.Error(ex, "Failed to parse LLM parameter response: {Content}", response.Content[..Math.Min(500, response.Content.Length)]);
            return new LlmParameterResult { TokensUsed = response.Tokens };
        }
    }

    private static Dictionary<string, string> ParseParameters(JsonElement root)
    {
        var result = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        if (root.TryGetProperty("parameters", out var paramsEl))
        {
            foreach (var prop in paramsEl.EnumerateObject())
            {
                var value = prop.Value.ValueKind == JsonValueKind.String
                    ? prop.Value.GetString() ?? string.Empty
                    : prop.Value.ToString();
                result[prop.Name] = value;
            }
        }
        return result;
    }

    private static string ExtractJson(string content)
    {
        // LLM sometimes wraps JSON in markdown code blocks
        var start = content.IndexOf('{');
        var end = content.LastIndexOf('}');
        if (start >= 0 && end > start)
            return content[start..(end + 1)];
        return content;
    }
}
