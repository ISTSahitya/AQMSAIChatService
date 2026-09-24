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
            The table below will show the full list — write ONE intro sentence only.

            If the user asked about "schools" specifically, refer to them as "school site(s)".
            If the user filtered by region, mention the region name in the intro.

            If user asked for AQI RANGE (more than X / less than X / between X and Y):
            FORMAT: "There are {N} site(s) with AQI {range description}{region suffix}. The table below shows the full list."
            Examples:
              "There are 4 site(s) with AQI above 30 in Al Ain. The table below shows the full list."
              "There are 2 site(s) with AQI below 100 in Abu Dhabi. The table below shows the full list."
              "There are 3 site(s) with AQI between 50 and 150. The table below shows the full list."
            If no rows: "No sites currently match the requested AQI range{region suffix}."

            If ONE specific category was requested (e.g. "Moderate", "Unhealthy"):
            FORMAT: "There are {N} site(s) currently with {AQICategory} air quality{region suffix}. The table below shows the full list."
            If no rows: "No sites currently have {AQICategory} air quality{region suffix}."

            If MULTIPLE categories were requested (e.g. "Moderate and Very Unhealthy"):
            Count sites per category from the data.
            FORMAT: "There are {N1} site(s) with {Cat1} and {N2} site(s) with {Cat2} air quality{region suffix}. The table below shows the full list."

            For "critical" or "hotspot" (AQI > 100):
            FORMAT: "There are {N} air quality hotspot(s){region suffix} (AQI above 100). The table below shows the full list."
            If empty: "No sites{region suffix} are currently above AQI 100."

            If NO filter at all:
            FORMAT: "Current AQI status across {N} accessible site(s). The table below shows all sites."

            {region suffix} = " in {RegionName}" if a region was filtered, otherwise omit.
            NEVER list individual sites in text — the table handles that.
            NEVER output placeholder text — always use actual N count from the data.
            """,

        // Pie chart: AQI category distribution — columns: Category, Count
        ["schools_aqi_pie"] = """
            Data columns: "Category" (AQI category name), "Count" (number of sites in that category).
            The pie chart below will visualise the distribution — write ONE short summary sentence only.

            If filtered to schools (sectorName = "Public & Govt-School"):
            FORMAT: "Here is the current AQI category distribution across {TotalCount} school site(s){region suffix}. The chart below shows the breakdown by category."
            Example: "Here is the current AQI category distribution across 42 school site(s). The chart below shows the breakdown by category."

            If no sector filter (all sites):
            FORMAT: "Here is the current AQI category distribution across {TotalCount} site(s){region suffix}. The chart below shows the breakdown by category."

            {TotalCount} = sum of all Count values in the data.
            {region suffix} = " in {RegionName}" if regionName was provided, otherwise omit.
            NEVER list individual sites or categories in text — the chart handles that.
            NEVER output placeholder text — always use actual totals from the data.
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
            Output plain text only. No bullets. No markdown. No table.

            SAFETY / HEALTH / ATTENDANCE QUESTION — if the user asks whether the site is safe, good for health, safe for children/elderly/disabled/vulnerable/sensitive groups, or whether people should attend, go, visit, or use the space:
            Find the AQI Index row and 1–2 key pollutant readings (e.g. PM2.5, CO2) from the data.
            FORMAT (use actual values — never placeholder text):
            "I can share the measurements: AQI {AQI_VALUE} ({AQI_LEVEL}) as of {Last Updated}, with PM2.5 at {PM2.5_VALUE} µg/m³ and CO2 at {CO2_VALUE} PPM. Whether the building is safe to attend is a decision for the responsible health authority — the data alone cannot decide that. I can prepare the full data picture to support the decision."
            If PM2.5 or CO2 is not available, omit that pollutant and use whatever key pollutant IS available.
            If no AQI data at all: "No current readings are available for {Site Name}. The device may be offline. Whether the building is safe to attend is a decision for the responsible health authority."
            Keywords that trigger this format: safe, safety, attend, attendance, suitable, healthy, unhealthy for, good for children, good for elderly, good for disabled, can children, should children, can people, should people, is it ok, is it fine, is it good.

            User asked for AQI or air quality category (NOT a safety question) → find the row where Parameter = "AQI Index":
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

            Pollutant groups (for context when user asks about "physical" or "chemical"):
            Physical: PM2.5, PM10, Temperature, Humidity, Noise
            Chemical: CO, CO₂, NO₂, SO₂, O₃, CH₂O, VOC/TVOC

            If NO rows returned (empty data): "No readings are currently available for that device. The device may be offline or has not sent any data yet."

            ONE row (user asked for a specific parameter like AQI, PM2.5, CO, etc.):
            "{DeviceName} — {ParameterName}: {ParameterValue} {UnitName} (Last measured: {LastMeasured})"
            If ParameterName is "AQI Index", append AQI category: e.g. "AQI Index: 58 — Moderate"
            If ParameterValue is null or missing: "{DeviceName} — {ParameterName}: No data available."

            MULTIPLE rows (user asked for all readings, physical readings, chemical readings, etc.):
            If user asked for "physical readings": intro = "{DeviceName} — Physical parameters (live):"
            If user asked for "chemical readings": intro = "{DeviceName} — Chemical parameters (live):"
            Otherwise: Line 1: "{DeviceName} — Last measured: {LastMeasured}"
            Then output EVERY row as one line: "{ParameterName}: {ParameterValue} {UnitName}" — skip UnitName if empty or null.
            Skip any row where ParameterValue is null.
            You MUST list ALL parameters from ALL rows. Do not stop after the first line.

            AQI categories: 0–50 Good, 51–100 Moderate, 101–150 Unhealthy for Sensitive Groups, 151–200 Unhealthy, 201–300 Very Unhealthy, 301+ Hazardous.
            No JSON, no bullets, no extra commentary.
            """,

        ["device_info"] = """
            Data columns: "DeviceName", "StationName", "RegionName", "SectorName", "SubSectorName", "DeviceModel", "DeployedOn", "LastMeasured", "Status".

            If NO rows returned: "No device found matching that name. Please check the device name and try again."

            Answer ONLY what the user asked. Do not output all columns — respond to the specific question:
            - "site" or "station" or "where" → "Device {DeviceName} is deployed at {StationName}."
            - "region" → "Device {DeviceName} is in the {RegionName} region."
            - "sector" → "Device {DeviceName} belongs to the {SectorName} sector." (append SubSectorName if not null/empty: " — {SubSectorName}")
            - "deployed" / "created" / "installed" / "when" / "registration" → "Device {DeviceName} was registered on {DeployedOn}."
            - "last measured" / "last data" / "last reading time" / "last measurement" → "Device {DeviceName} last measured at {LastMeasured}."
            - "status" → "Device {DeviceName} is currently {Status}. Last measured at {LastMeasured}."
            - "model" / "type" / "device model" → "Device {DeviceName} is a {DeviceModel} model."
            - General "details" / "info" / "about" / "tell me about" → output a summary:
              "Device {DeviceName}:
               Site: {StationName}
               Region: {RegionName}
               Sector: {SectorName}
               Status: {Status}
               Last Measured: {LastMeasured}
               Registered: {DeployedOn}"
              Add DeviceModel line only if not null/empty. Add SubSectorName line only if not null/empty.

            If SubSectorName is null or empty, omit it everywhere.
            If DeviceModel is null or empty, omit the model line.
            NEVER output placeholder text — always use actual values from the data.
            """,

        ["device_pollutants"] = """
            Data columns: "ParameterName", "UnitName".
            Each row is one parameter the device monitors.

            If NO rows returned: "No parameters found for that device. The device may not have any configured parameters."

            FORMAT: "Device {DeviceName} monitors {N} parameter(s): {ParameterName1} ({UnitName1}), {ParameterName2} ({UnitName2}), ..."
            List ALL parameters from all rows separated by commas.
            Group as: Physical — PM2.5, PM10, Temperature, Humidity, Noise. Chemical — CO, CO₂, NO₂, SO₂, O₃, CH₂O, VOC.
            If the device has both types: "Physical: {list}. Chemical: {list}."
            NEVER output placeholder text — always use actual values from the data.
            """,

        ["device_reading_history"] = """
            Data columns: "DeviceName", "ParameterName", "ParameterValue", "UnitName", "Timestamp".
            Each row is one reading at one point in time for one parameter.

            Pollutant groups (for context when user asks about "physical" or "chemical"):
            Physical: PM2.5, PM10, Temperature, Humidity, Noise
            Chemical: CO, CO₂, NO₂, SO₂, O₃, CH₂O, VOC/TVOC

            If NO rows returned: "No historical data found for that device and interval. The device may not have data for the requested time period."

            If rows are for ONE parameter (user asked for a specific pollutant):
            Line 1: "{DeviceName} — {ParameterName} ({interval label} readings):"
            Then one line per row: "{Timestamp}: {ParameterValue} {UnitName}"
            Skip UnitName if empty. Show up to 20 rows.

            If rows are for MULTIPLE parameters (all, physical, or chemical group):
            If user asked "physical": intro = "{DeviceName} — Physical parameters ({interval label}):"
            If user asked "chemical": intro = "{DeviceName} — Chemical parameters ({interval label}):"
            Otherwise: intro = "{DeviceName} — {interval label} readings from {earliest Timestamp} to {latest Timestamp}."
            Write the intro sentence only — the table below will show all rows.

            Interval labels: "5min"→"5-minute", "1hour"→"1-hour average", "8hour"→"8-hour average", "24hour"→"24-hour average", "monthly"→"monthly average", "yearly"→"yearly average".
            AQI categories: 0–50 Good, 51–100 Moderate, 101–150 Unhealthy for Sensitive Groups, 151–200 Unhealthy, 201–300 Very Unhealthy, 301+ Hazardous.
            NEVER output placeholder text — always use actual values from the data.
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

        ["sites_all_regions"] = """
            Data columns: "StationName", "RegionName", "SectorName", "SubSectorName", "LiveStatus".
            The table groups all sites across all regions. Write ONE intro sentence only.
            FORMAT: "Here are all {N} site(s) across {RegionCount} regions. The table below shows each site grouped by region."
            Example: "Here are all 13 site(s) across 3 regions. The table below shows each site grouped by region."
            N = total row count. RegionCount = number of distinct RegionName values.
            NEVER list individual sites in text — the table handles that.
            NEVER output placeholder text — always use actual counts from the data.
            """,

        ["sites_count_by_region"] = """
            HAWAQM has 3 regions in total: Abu Dhabi, Al Ain, and Al Dhafra.
            The data rows show only the regions the user has access to.

            If the user asked ONLY about regions (e.g. "how many regions", "what are the regions", "number of regions", "regions in this application"):
            FORMAT: "HAWAQM covers {N} regions: {RegionName1}, {RegionName2}, and {RegionName3}."
            Example: "HAWAQM covers 3 regions: Abu Dhabi, Al Ain, and Al Dhafra."
            Do NOT mention site counts in this case.

            If the user asked about sites per region or site breakdown by region:
            FORMAT: "HAWAQM has 3 regions — Abu Dhabi ({TotalSites1} sites, {ActiveSites1} active), Al Ain ({TotalSites2} sites, {ActiveSites2} active), and Al Dhafra ({TotalSites3} sites, {ActiveSites3} active)."

            If user has access to fewer than 3 regions: "HAWAQM has 3 regions in total (Abu Dhabi, Al Ain, and Al Dhafra). Based on your access permissions, you can view: {RegionName} ({TotalSites} sites). Contact your administrator for access to other regions."
            Always use actual values from the data rows. Never output placeholders.
            """,

        ["list_sectors"] = """
            Data columns: "SectorName", "TotalSites".
            If user asked how many sectors or what sectors exist:
            FORMAT: "HAWAQM has {N} sectors: {SectorName1}, {SectorName2}, and {SectorName3}."
            Example: "HAWAQM has 3 sectors: Commercial, Public & Govt-School, and Residential."
            N = number of rows. List all SectorName values. Do NOT mention site counts unless user asked for them.
            NEVER output placeholder text — always use actual values from the data.
            """,

        ["list_subsectors"] = """
            Data columns: "SubSectorName", "SectorName" (parent sector), "TotalSites".
            If user asked how many sub-sectors or what sub-sectors exist:
            FORMAT: "HAWAQM has {N} sub-sectors across {SectorCount} sectors: {list each SubSectorName grouped under its SectorName}."
            Example: "HAWAQM has 5 sub-sectors — under Public & Govt-School: Private School, Government School; under Commercial: Mall, Office; under Residential: Villa."
            N = total number of rows. Group by SectorName. Do NOT mention site counts unless user asked.
            NEVER output placeholder text — always use actual values from the data.
            """,

        ["sites_count_by_sector"] = """
            Data columns: "SectorName", "TotalSites", "ActiveSites".
            FORMAT: "The current sector distribution is {SectorName1}: {TotalSites1}, {SectorName2}: {TotalSites2}, {SectorName3}: {TotalSites3}, and {SectorName4}: {TotalSites4}."
            Use the exact SectorName and TotalSites values from each row. List all sectors from the data in one sentence.
            Example: "The current sector distribution is Public & Govt-School: 2, Residential: 1, and Commercial: 4."
            NEVER output placeholder text — always use actual values from the data.
            """,

        ["site_device_summary"] = """
            Data columns: "StationName", "RegionName", "SectorName", "TotalDevices", "ActiveDevices", "InactiveDevices", "LastReadingTime", "BestAQI".

            Adapt the response based on what the user asked:

            If user asked about SITES WITH NO DEVICES (TotalDevices = 0 for all rows, or no rows):
            If rows exist: "There are {N} site(s) with no registered devices: {StationName1}, {StationName2}."
            If no rows: "All registered sites have at least one device."

            If user asked about SITES WITH MORE/FEWER THAN N DEVICES:
            "There are {N} site(s) matching your filter: {StationName1} ({TotalDevices1} devices), {StationName2} ({TotalDevices2} devices)."

            If user asked about DEVICE STATUS PER SITE (active/inactive breakdown):
            "Here is the device status per site — {StationName1}: {TotalDevices1} total ({ActiveDevices1} active, {InactiveDevices1} inactive), {StationName2}: {TotalDevices2} total ({ActiveDevices2} active, {InactiveDevices2} inactive)."

            If user asked about LAST READING PER SITE:
            "Here is the last reading time per site — {StationName1}: {LastReadingTime1}, {StationName2}: {LastReadingTime2}."

            If user asked about BEST AQI PER SITE:
            "Here are the best (lowest) AQI readings per site — {StationName1}: AQI {BestAQI1}, {StationName2}: AQI {BestAQI2}."

            If user asked a GENERAL device summary:
            "Here is the device summary across {N} site(s). The table shows device counts, active/inactive status, last reading time, and best AQI per site."

            ALWAYS use actual values from the data — never output placeholder text.
            The table handles full details — write ONE intro sentence or short paragraph only.
            """,

        ["sites_by_sector"] = """
            Data columns: "StationName", "SubSectorName", "SectorName", "RegionName", "LiveStatus".
            SubSectorName shows the sub-type (e.g. "Private School", "Government School", "Indian School").

            If no rows: "There are no {SectorName} sites found for your query. Contact your administrator if you need access."

            If user asked about SCHOOLS (sectorName = "Public & Govt-School"):
            Group by SubSectorName to show public vs private breakdown.
            FORMAT:
            "There are {N} school site(s){region suffix}:
             {SubSectorName1}: {StationName1}, {StationName2}
             {SubSectorName2}: {StationName3}"
            Example:
            "There are 4 school site(s) in Abu Dhabi:
             Government School: Al Naeem School, Al Bateen School
             Private School: Al Saad Indian School, GEMS School"

            If user asked about COMMERCIAL or RESIDENTIAL:
            Group by RegionName if spanning multiple regions.
            FORMAT: "There are {N} {SectorName} site(s){region suffix}: {StationName1}, {StationName2}, ..."

            {region suffix} = " in {RegionName}" if filtered to one region, otherwise omit.
            NEVER output placeholder text — always use actual values from the data.
            The table below shows the full list — write ONE intro paragraph only.
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

        // Yearly AQI ranking — top N sites by highest avg AQI for a year
        ["top_sites_aqi_yearly"] = """
            Data columns: "StationName", "RegionName", "AvgAQI", "Year".
            Line 1: "The {N} sites with the highest average AQI in {Year} were:"
            Then one line per site (ranked 1 to N): "{Rank}. {StationName} ({RegionName}) — Avg AQI: {AvgAQI} ({AQICategory})"
            List EVERY row from the data. Do not skip any.
            AQI categories: 0–50 Good, 51–100 Moderate, 101–150 Unhealthy for Sensitive Groups, 151–200 Unhealthy, 201–300 Very Unhealthy, 301+ Hazardous.
            If no rows: "No yearly AQI data is available for the requested year."
            NEVER output placeholder text — always use actual values from the data.
            """,

        // Yearly AQI ranking — top N sites by lowest avg AQI for a year
        ["bottom_sites_aqi_yearly"] = """
            Data columns: "StationName", "RegionName", "AvgAQI", "Year".
            Line 1: "The {N} sites with the lowest average AQI in {Year} were:"
            Then one line per site (ranked 1 to N): "{Rank}. {StationName} ({RegionName}) — Avg AQI: {AvgAQI} ({AQICategory})"
            List EVERY row from the data. Do not skip any.
            AQI categories: 0–50 Good, 51–100 Moderate, 101–150 Unhealthy for Sensitive Groups, 151–200 Unhealthy, 201–300 Very Unhealthy, 301+ Hazardous.
            If no rows: "No yearly AQI data is available for the requested year."
            NEVER output placeholder text — always use actual values from the data.
            """,

        // Worst air-quality month at a site — Month, AvgAQI, MaxAQI + pollutant exceedances
        ["worst_month_pollutants"] = """
            Data columns: "Month" (int 1–12), "Year", "AvgAQI", "MaxAQI", "ParameterName", "ExceedanceCount".
            Multiple rows — same Month/Year repeated, one row per pollutant ordered by ExceedanceCount DESC.
            Extract: Month, Year, AvgAQI, MaxAQI from first row. Pollutants = all rows.

            Month number to name: 1=January, 2=February, 3=March, 4=April, 5=May, 6=June,
              7=July, 8=August, 9=September, 10=October, 11=November, 12=December.

            FORMAT:
            "The worst month was {MonthName} {Year}, averaging {AvgAQI}."

            If exceedance rows exist:
            "The pollutants behind it were {Pollutant1} with {ExceedanceCount1} exceedance(s){, Pollutant2 with N2, ...}, and the month's highest single reading was {MaxAQI}."

            Closing line:
            "That tells you which pollutants the poor month was made of, which is the useful starting point. It does not tell you where they came from — the measurements identify the signal, but the source needs to be established at the site itself."

            If no exceedance rows (all pollutants below threshold):
            "No pollutant exceedances were recorded that month, though the AQI of {AvgAQI} suggests elevated composite levels."

            If "Error" is present: output the error.
            NEVER output placeholder text — always use actual values from the data.
            """,

        // Region AQI trend over multiple years — one row per region per year
        ["region_aqi_trend_yearly"] = """
            Data columns: "RegionName", "Year", "AvgAQI", "HighDays", "DSR_Pct".
            Multiple rows — one per region per year. Group rows by RegionName.

            STEP 1 — compute each region's overall average AvgAQI across all years (simple mean).
            STEP 2 — identify the region with the highest overall AvgAQI = poorest region.

            Line 1: "{PoorestRegion} came out poorest across the {N} years, with an average AQI of {OverallAvg}, {TotalHighDays} days above 100 and {AvgDSR}% average data completeness."
            Line 2: "The other regions were: {Region2}: avg AQI {Avg2}, {HighDays2} days above 100, DSR {DSR2}%. {Region3}: avg AQI {Avg3}, {HighDays3} days above 100, DSR {DSR3}%."
            Line 3: "Completeness is shown for each region deliberately. A region that was reporting less of the time can look better simply because more of its bad periods went unrecorded, so any ranking should only include periods with enough data behind them to be comparable."

            Then list each region year by year:
            "{RegionName}:"
            "  - {Year}: Avg AQI {AvgAQI}, Days above 100: {HighDays}, DSR: {DSR_Pct}%"

            If no rows: "No yearly AQI data found for any region in the requested period."
            NEVER output placeholder text — always use actual values from the data.
            """,

        // Region best/worst for a single year with completeness — rows ordered best (lowest AQI) first
        ["region_aqi_single_year"] = """
            Data columns: "RegionName", "Year", "AvgAQI", "HighDays", "DSR_Pct".
            Rows ordered by AvgAQI ASC — first row = best (lowest AQI), last row = worst.

            If user asked which region performed BEST:
            Line 1: "{Row1.RegionName} had the strongest combined result in {Year} — average AQI {Row1.AvgAQI}, {Row1.HighDays} days above 100, and {Row1.DSR_Pct}% Data Success Rate."
            Line 2: "Both halves of that matter. A region can post an excellent AQI simply because it was recording less of the time, so a low average sitting on poor completeness isn't a better result, it's a less certain one."
            Then: "Setting all regions side by side with their completeness alongside keeps the comparison honest:"
            Then list all rows: "- {RegionName}: avg AQI {AvgAQI}, {HighDays} days above 100, DSR {DSR_Pct}%"

            If user asked which region performed WORST:
            Line 1: "{LastRow.RegionName} had the weakest combined result in {Year} — average AQI {LastRow.AvgAQI}, {LastRow.HighDays} days above 100, and {LastRow.DSR_Pct}% Data Success Rate."
            Then same completeness caveat and full table.

            If no rows: "No yearly AQI data found for that year."
            NEVER output placeholder text — always use actual values from the data.
            """,

        // Yearly AQI trend at a site — one row per year with AvgAQI, HighDays, DSR_Pct
        ["site_aqi_trend_yearly"] = """
            Data columns: "SiteName", "Year", "AvgAQI", "HighDays", "DSR_Pct". One row per year, ordered oldest to newest.
            Extract: Year1 = first row, YearN = last row (most recent).

            FORMAT:
            Line 1: "Across the {N} comparable years, {SiteName} is on a {TREND} trend."
              - TREND = "improving" if YearN.AvgAQI < Year1.AvgAQI, "worsening" if higher, "stable" if within 5 points.
            Line 2: "Average AQI moved from {Year1.AvgAQI} in {Year1.Year} to {YearN.AvgAQI} in {YearN.Year}, a change of {PERCENT_CHANGE}%."
              - PERCENT_CHANGE = ROUND(ABS(YearN.AvgAQI - Year1.AvgAQI) / Year1.AvgAQI * 100, 1)
            Line 3: "Days above AQI 100 went from {Year1.HighDays} to {YearN.HighDays}."
            Line 4: "Completeness across the {N} years was {DSR list joined with ', '}%, which makes this a {CONFIDENCE} conclusion."
              - CONFIDENCE = "high-confidence" if all DSR_Pct >= 75, "moderate-confidence" if any between 50–75, "low-confidence" if any below 50.
            Line 5: "If one year is noticeably thinner than the others, part of the movement may reflect a monitoring difference rather than a real change in the air."

            Then list each year: "- {Year}: Avg AQI {AvgAQI}, Days above 100: {HighDays}, DSR: {DSR_Pct}%"

            If "Error" is present: output the error message.
            AQI categories: 0–50 Good, 51–100 Moderate, 101–150 Unhealthy for Sensitive Groups, 151–200 Unhealthy, 201–300 Very Unhealthy, 301+ Hazardous.
            NEVER output placeholder text — always use actual values from the data.
            """,

        // Indoor vs nearest ambient station AQI comparison
        ["indoor_ambient_compare"] = """
            Data columns: "IndoorSite", "IndoorAQI", "AmbientStation", "AmbientAQI", "Difference", "HigherOrLower", "Distance_km", "Timestamp".
            There is exactly 1 row.

            FORMAT:
            "{IndoorSite}'s current indoor AQI is {IndoorAQI}, while the selected nearby ambient station {AmbientStation} is {AmbientAQI}. The indoor value is {Difference} AQI points {HigherOrLower} at {Timestamp}."

            Add one closing sentence:
            - If IndoorAQI > AmbientAQI: "The indoor environment is currently worse than the outdoor air at the nearest ambient station ({Distance_km} km away)."
            - If IndoorAQI < AmbientAQI: "The indoor environment is currently better than the outdoor air at the nearest ambient station ({Distance_km} km away)."
            - If equal: "The indoor and outdoor AQI are currently the same at the nearest ambient station ({Distance_km} km away)."

            If "Error" is present in the data, output the error message instead.
            AQI categories: 0–50 Good, 51–100 Moderate, 101–150 Unhealthy for Sensitive Groups, 151–200 Unhealthy, 201–300 Very Unhealthy, 301+ Hazardous.
            NEVER output placeholder text — always use actual values from the data.
            """,

        // Lowest DSR devices last month — ranked table + summary sentence
        ["lowest_dsr_devices"] = """
            Data columns: "DeviceName", "StationName", "RegionName", "DSR_Pct".
            Rows are ordered lowest DSR first.

            Line 1: "The devices with the lowest Data Success Rate last month are:"
            Then one line per row: "- {DeviceName} ({StationName}, {RegionName}): {DSR_Pct}%"
            Final line: "The lowest was {Row1.DeviceName} at {Row1.DSR_Pct}%."

            If no rows: "No Data Success Rate data found for last month."
            NEVER output placeholder text — always use actual values from the data.
            """,

        // Which device drove the peak site AQI last month
        ["device_peak_aqi_contribution"] = """
            Data columns: "DeviceName", "AQI", "PeakDateTime", "SiteAQI".
            There will be 2 rows — one per device, ordered highest AQI first.
            Row 1 = the higher device (DeviceHigh), Row 2 = the lower device (DeviceLow).

            FORMAT (fill ALL values from actual data — never use placeholder text):
            "At the site's highest AQI period on {PeakDateTime}, {DeviceHigh.DeviceName} was the higher of the two at {DeviceHigh.AQI}, against {DeviceLow.AQI} for {DeviceLow.DeviceName}. Since the site figure is an average of its devices, the site peak was partly held down by the quieter device — so the actual conditions around {DeviceHigh.DeviceName} were worse than the site number suggests."

            If only 1 row returned (single-device site):
            "The site's peak AQI last month was {AQI} at {PeakDateTime}, recorded by {DeviceName}. This site has only one device, so the site AQI equals the device reading directly."

            If no rows: "No hourly AQI data found for that site last month. The devices may have been offline."
            NEVER output placeholder text — always use actual values from the data.
            """,

        // Device comparison at a site for a specific month
        ["compare_devices_monthly"] = """
            Data columns: "DeviceName", "AvgAQI", "DSR_Pct".
            There will be exactly 2 rows (one per device).
            Extract: Device1 = first row, Device2 = second row.
            Compute: Difference = ABS(Device1.AvgAQI - Device2.AvgAQI), rounded to 1 decimal.

            FORMAT (fill ALL values from actual data — never use placeholder text):
            "For {Month} {Year}, {Device1.DeviceName} averaged AQI {Device1.AvgAQI} and {Device2.DeviceName} averaged AQI {Device2.AvgAQI}. Their average difference was {Difference} AQI points. Data Success Rates were {Device1.DSR_Pct}% and {Device2.DSR_Pct}%, respectively."

            AQI categories: 0–50 Good, 51–100 Moderate, 101–150 Unhealthy for Sensitive Groups, 151–200 Unhealthy, 201–300 Very Unhealthy, 301+ Hazardous.
            If only 1 row: "Only one device found at that site for the requested month. {DeviceName} averaged AQI {AvgAQI} with a Data Success Rate of {DSR_Pct}%."
            If no rows: "No AQI data found for that site and month. The devices may have been offline or the site name may not match."
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
        var systemPrompt = $$"""
            You are HAWAQM AI, an air quality monitoring assistant for a UAE government platform.
            Answer the user's question using ONLY the information provided below. Do not invent facts.
            When a FAQ entry directly answers the question, return that answer VERBATIM — do not paraphrase or reword it.
            EXCEPTION: if the FAQ answer contains the placeholder {SITE_NAME}, replace it with the actual site or school name the user mentioned in their question. If the user did not mention a specific site, replace {SITE_NAME} with "your site".
            Be concise — one to three sentences maximum. Output plain text only.

            HAWAQM Knowledge Base:
            {{faqContext}}
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

            PRIORITY RULE 0 — device name ALWAYS beats ALL other rules including "average", "mean", "trend":
            A device name is ONLY the patterns "BA" or "SEI100M" followed by 1–4 digits (e.g. "BA0010", "BA 0010", "ba0010", "SEI100M0014", "SEI100M 0014"). Site names, school names, and person names are NOT device names.
            NEVER treat a school name, site name, or person name as a device name — even if it contains letters and numbers. For example: "Al Reyada School", "Al Saad Indian School", "Kaltham Hasan", "Al Naeem School" are SITE names, NOT device names.
            If the question does NOT contain an explicit BA/SEI100M device code, do NOT select any device template (device_last_reading, device_status, device_info, device_pollutants, device_reading_history).
            If the question contains a device name (pattern: "BA" or "SEI100M" followed by 1–4 digits, with or without space) THEN choose one of the device templates:
            - If the question asks for METADATA (site, station, region, sector, deployed, created, installed, when, model, details, info, last measured time, last data time, where is) → select "device_info"
            - If the question asks what POLLUTANTS or PARAMETERS a device monitors/measures → select "device_pollutants"
            - If the question asks for STATUS only (active, inactive, online, offline, is it working, is it sending) WITHOUT asking for readings → select "device_status"
            - If the question mentions ANY of: a time interval (5 min, 1 hour, 1H, 8 hour, 8H, 24 hour, 24H, daily, monthly, yearly, annual), OR the words average/avg/mean/reading/value/history/historical/past/trend, OR a specific date/time (yesterday, last hour, last 24 hours, last month, last year, "at 8AM", "on 2026-07-16", any date or clock time) → select "device_reading_history"
            - Otherwise (no time interval or date, asks for current/latest/live/now readings, or no specific context) → select "device_last_reading"
            NEVER select a site template (site_aqi_single, site_readings_all, site_aqi_all, site_aqi_by_category) when the question contains a device name.
            NEVER select "average_at_station", "pollutant_trend_hourly", or any non-device template when the question contains a device name — device name is ABSOLUTE priority.
            Examples of questions that do NOT contain a device name and must NOT use device templates:
            - "what is the Site AQI at Al Reyada School?" → NO device name → site_aqi_single, stationName="Al Reyada School"
            - "AQI at Al Saad Indian School" → NO device name → site_aqi_single, stationName="Al Saad Indian School"
            - "CO2 at Kaltham Hasan Al Obeidli" → NO device name → site_aqi_single, stationName="Kaltham Hasan Al Obeidli"
            Examples — device_reading_history (historical/averaged/timestamped readings):
            - "1H averages of SEI100M 0008" → device_reading_history, deviceName="SEI100M 0008", interval="1hour"
            - "1H reading of SEI100M 0008" → device_reading_history, deviceName="SEI100M 0008", interval="1hour"
            - "1H average reading of SEI100M 0008" → device_reading_history, deviceName="SEI100M 0008", interval="1hour"
            - "what is the 1H average of device SEI100M 0008" → device_reading_history, deviceName="SEI100M 0008", interval="1hour"
            - "what is the 1H average reading of SEI100M 0008 on 2026-07-16 8AM" → device_reading_history, deviceName="SEI100M 0008", interval="1hour", targetTime="8:00AM", startDate="2026-07-16"
            - "1H average of SEI100M 0008 at 8AM yesterday" → device_reading_history, deviceName="SEI100M 0008", interval="1hour", targetTime="8:00AM", startDate=yesterday
            - "give me the PM2.5 value at 8AM at 1H average for SEI100M 0008" → device_reading_history, deviceName="SEI100M 0008", interval="1hour", parameterName="PM2.5", targetTime="8:00AM"
            - "PM2.5 reading at 1H interval for BA 0010 on 2026-07-16" → device_reading_history, deviceName="BA 0010", interval="1hour", parameterName="PM2.5", startDate="2026-07-16"
            - "CO2 value at 3PM using 1H averages for SEI100M 0014" → device_reading_history, deviceName="SEI100M 0014", interval="1hour", parameterName="CO2", targetTime="3:00PM"
            - "show me the 8H average CO2 for BA 0010 yesterday" → device_reading_history, deviceName="BA 0010", interval="8hour", parameterName="CO2", startDate=yesterday
            - "what was the temperature reading at 2AM for SEI100M 0008 2 days ago" → device_reading_history, deviceName="SEI100M 0008", parameterName="Temperature", targetTime="2:00AM", startDate=today−2
            - "average reading for BA 0010 last month" → device_reading_history, deviceName="BA 0010", interval="monthly"
            - "daily CO2 average for SEI100M 0014" → device_reading_history, deviceName="SEI100M 0014", interval="24hour", parameterName="CO2"
            - "BA0010 readings for last month" → device_reading_history, deviceName="BA0010", interval="monthly"
            - "mean PM2.5 for BA 0010 this week" → device_reading_history, deviceName="BA 0010", interval="24hour", parameterName="PM2.5"
            - "show me the 1 hour PM2.5 for BA 0010" → device_reading_history, deviceName="BA 0010", interval="1hour", parameterName="PM2.5"
            - "pollutant reading at 14:00 for SEI100M 0008 yesterday" → device_reading_history, deviceName="SEI100M 0008", targetTime="14:00", startDate=yesterday
            - "what pollutant value did BA 0010 record at noon on 2026-09-01" → device_reading_history, deviceName="BA 0010", targetTime="12:00", startDate="2026-09-01"
            Examples — device_last_reading (current/live):
            - "what is the PM2.5 for device BA0010" → device_last_reading, deviceName="BA0010"
            - "PM2.5 reading for BA 0010" → device_last_reading, deviceName="BA 0010"
            - "current readings for SEI100M 0008" → device_last_reading, deviceName="SEI100M 0008"
            - "latest CO2 for BA 0010" → device_last_reading, deviceName="BA 0010"
            Examples — device_info / device_pollutants / device_status:
            - "what site is device BA 0001" → device_info, deviceName="BA 0001"
            - "which region is BA 0010 in" → device_info, deviceName="BA 0010"
            - "when was BA 0010 deployed" → device_info, deviceName="BA 0010"
            - "what pollutants does SEI100M 0014 monitor" → device_pollutants, deviceName="SEI100M 0014"
            - "what parameters does BA 0010 have" → device_pollutants, deviceName="BA 0010"
            - "what sector is BA 0010 in" → device_info, deviceName="BA 0010"
            - "is BA 0010 active" → device_status, deviceName="BA 0010"

            PRIORITY RULE — site name beats region/generic rules:
            If the user names a SPECIFIC site or person name in their question (but NOT a device name), ALWAYS use rule 1 or 2 — never site_aqi_all, never site_aqi_by_category, never region queries, NEVER schools_latest_pollutant.
            A site name is ANY proper noun or name that is NOT just one of the three region names alone ("Abu Dhabi", "Al Ain", "Al Dhafra") and NOT a device name.
            Site names include: school names, person names (e.g. "Kaltham Hasan Al Obeidli"), place names with descriptors, institutional names.
            Words like "at", "for", "of", "in" before a name indicate the name is a site.
            If the question contains a SPECIFIC school name (e.g. "Al Reyada School", "Al Saad Indian School", "Al Naeem School") — even if the word "School" appears — treat the full phrase as a site name and use rule 1 or 2. NEVER route a question with a specific named school to schools_latest_pollutant.
            SAFETY RULE — if the user asks whether a SPECIFIC named site is safe, suitable, healthy, or okay for children/elderly/disabled/vulnerable people, or whether people should attend/visit/go — this is a safety question about a named site. Select "site_aqi_single" and extract the stationName. The formatter will provide live AQI data plus the required health authority disclaimer. NEVER route these to "faq_answer" — the FAQ cannot provide live readings.
            Examples:
            - "AQI at Kaltham Hasan AL Obeidli site" → stationName="Kaltham Hasan AL Obeidli" → site_aqi_single
            - "AQI at Al Saad Indian School in Al Ain" → stationName="Al Saad Indian School" → site_aqi_single
            - "what is the Site AQI at Al Reyada School" → stationName="Al Reyada School" → site_aqi_single
            - "Can you confirm AL Reyada School is safe for children to attend today?" → stationName="Al Reyada School" → site_aqi_single
            - "Is Al Naeem School suitable for elderly visitors?" → stationName="Al Naeem School" → site_aqi_single
            - "Is it safe to attend Al Saad Indian School today?" → stationName="Al Saad Indian School" → site_aqi_single
            - "CO2 at Abu Dhabi Residential" → stationName="Abu Dhabi Residential" → site_aqi_single
            - "readings at Al Ain Commercial Institutional" → stationName="Al Ain Commercial Institutional" → site_readings_all
            - "AQI in Abu Dhabi" (only a region word, no other name) → region query
            NEVER return a list of all sites when the user has named a specific site.

            1. If the user mentions a SPECIFIC site or station name AND asks for ONE value (AQI, AQI category, or a specific pollutant like CO2, PM2.5, NO2, temperature, humidity) OR asks a safety/health/attendance question about that site → select "site_aqi_single". Returns a single sentence — NOT a table.
               When extracting stationName: use ONLY the site name itself — strip any trailing region qualifier like "in Al Ain", "in Abu Dhabi", "in Al Dhafra". Strip trailing words like "site", "station", "location". Example: "Kaltham Hasan AL Obeidli site" → stationName = "Kaltham Hasan AL Obeidli".
            2. If the user mentions a SPECIFIC site or station name AND asks for ALL readings, ALL parameters, ALL current data, or the last 24 hours of readings → select "site_readings_all". Returns a table.
               Same stationName rule: strip region qualifier and trailing generic words from the extracted name.
            3. Only select "schools_latest_pollutant" when the user says "schools" generically with a GENERIC plural reference (e.g. "all schools", "schools in Al Ain", "what is the AQI at schools") with NO specific school name mentioned AND is asking for a specific POLLUTANT reading (CO2, PM2.5, AQI value, etc.) WITHOUT an AQI quality level filter. NEVER select schools_latest_pollutant when: (a) a specific school or site name is mentioned, or (b) the user is asking about AQI category/quality level (good, moderate, unhealthy, etc.) — use site_aqi_by_category with sectorName="Public & Govt-School" instead.
            4. If the user asks an AQI question that mentions ONLY a region name (Abu Dhabi, Abudhabi, Al Ain, Alain, Al Dhafra, Aldhafra) with NO specific site name, OR asks for "all regions", "by region", "region wise" → select "region_aqi_geographical". Extract regionName if one specific region is mentioned; omit it if all regions are requested. NEVER select region_aqi_geographical when a specific site name is also present — rule 1 takes priority.
            4b. If the user asks a generic AQI question with NO specific site name AND NO region AND NO quality level word → select "site_aqi_all". Examples: "show me the AQI", "what is the AQI", "show the AQI", "current AQI", "display AQI", "AQI at my sites". NEVER select site_aqi_all when: (a) the question names a specific site or person → use site_aqi_single (rule 1), or (b) the question contains a quality word like good, bad, poor, unhealthy, moderate, hazardous, safe, harmful, vulnerable, critical → use site_aqi_by_category (rule 8).
            5. If the user asks to LIST or SHOW schools, commercial sites, or residential sites — with or without a region filter — → ALWAYS select "sites_by_sector". Extract sectorName ("Public & Govt-School" for any school/govt/public question) and regionName if mentioned. This covers questions like "what are the different schools in Abu Dhabi?", "list schools in Al Ain", "schools present in Abudhabi", "which schools are in Al Dhafra", "commercial sites in Abu Dhabi". NEVER select schools_latest_pollutant or sites_by_region for these — those don't show school type breakdown.
            6. Extract deviceName ONLY from the CURRENT question. Never use a device name from conversation history.
            7. Extract the device name as written in the question. Device names follow these real patterns (with or without space between prefix and number): "BA 0001", "BA 0010", "BA0010", "SEI100M 0014", "SEI100M 0085", "SEI100M0085". Preserve exactly what the user wrote — the system handles all spacing/casing variants automatically.
            7b. For device_reading_history: extract interval from time words in the question.
               Interval mapping — match ANY of these phrases:
               → "5min"   : "5 min", "5minute", "5-minute", "raw", "5 minutes", "5min interval", "5-minute average", "5 min reading"
               → "1hour"  : "1 hour", "1h", "1H", "hourly", "1-hour", "one hour", "1 hour average", "1H average", "1H reading", "1 hour reading", "1H averages", "hourly average", "hourly reading", "at 1H", "1H interval", "using 1H", "1 hour interval"
               → "8hour"  : "8 hour", "8h", "8H", "8-hour", "eight hour", "8 hour average", "8H average", "8H reading", "8 hour reading", "8H interval", "8 hour interval"
               → "24hour" : "24 hour", "24h", "24H", "24-hour", "daily", "day", "one day", "24 hour average", "daily average", "daily reading", "24H interval", "per day"
               → "monthly": "monthly", "month", "1 month", "one month", "monthly average", "monthly reading", "per month"
               → "yearly" : "yearly", "year", "annual", "1 year", "one year", "yearly average", "per year", "annually"
               IMPORTANT — interval + targetTime can coexist: "PM2.5 at 8AM at 1H average" → interval="1hour", targetTime="8:00AM". Extract BOTH when both are present.
               IMPORTANT — "average"/"avg"/"mean" paired WITH a time unit → extract that interval. "average" alone without a time unit → omit interval (system defaults to 5min raw readings).
               IMPORTANT — a specific date or clock time WITHOUT an interval word → omit interval (system defaults to 5min raw readings for point-in-time lookups).
               If no interval word is found, omit interval entirely.
            7c. For device_reading_history: extract targetTime when the user mentions ANY specific clock time. Support all these patterns:
               "at 8AM", "at 8:00AM", "at 08:00" → targetTime: "8:00AM"
               "at 2:02AM", "at 2:02 AM" → targetTime: "2:02AM"
               "at 4AM", "at 4:00AM" → targetTime: "4:00AM"
               "at 3:03 AM" → targetTime: "3:03AM"
               "at 4:30PM", "at 16:30" → targetTime: "4:30PM"
               "at 14:00", "at 2PM", "at 2:00PM" → targetTime: "14:00"
               "at noon", "at 12PM" → targetTime: "12:00PM"
               "at midnight", "at 12AM" → targetTime: "12:00AM"
               Also always extract startDate as the calendar date (ISO YYYY-MM-DD) using the Today's date in the user message:
               "today" → today; "yesterday"/"1 day back"/"1 day ago" → today−1; "day before yesterday"/"2 days back" → today−2; "N days back/ago" → today−N; "1 week back/ago" → today−7; "N weeks back/ago" → today−(N×7); "last month"/"1 month back" → first day of last month; "N months back" → first day of (today−N months); "last year"/"1 year back" → Jan 1 of last year; "N years back" → Jan 1 of (current year−N); explicit date like "2026-07-16", "16 Jul 2026", "July 16" → that date in ISO format.
               The controller combines startDate (day) + targetTime (clock) to snap to nearest reading.
               Only extract targetTime for concrete clock times. Do NOT extract for vague periods like "morning", "evening", "night", "afternoon".
            8. PRIORITY RULE — AQI quality filter beats generic AQI: If the question contains ANY quality/condition word describing the AQI level (good, bad, poor, unhealthy, moderate, hazardous, safe, unsafe, clean, harmful, dangerous, critical, hotspot, vulnerable, sensitive, kids, children, elders, elderly, sick, diseased, patients, healthy, polluted, dirty, severe, alarming, toxic, deadly) → ALWAYS select "site_aqi_by_category", even if the question starts with "show me the sites" or "which sites". Extract aqiCategory using the mapping in the parameter rules below. If no specific level word is present, omit aqiCategory to return all sites grouped.
               IMPORTANT — AQI category DEFINITION questions: if the user is asking WHAT an AQI range or category means (e.g. "What AQI range is considered Good?", "What AQI range is Moderate?", "What AQI range is Unhealthy?", "What is the AQI range for Hazardous?", "What AQI is good?") → select "faq_answer". These are knowledge questions about the numeric range, NOT requests for site data. NEVER select "site_aqi_by_category" for these — that template lists sites, not definitions.
               IMPORTANT — AQI numeric threshold questions: if the user asks about sites or schools with AQI ABOVE/BELOW/MORE THAN/LESS THAN a specific number (e.g. "How many schools have an AQI above 30 in Abu Dhabi?", "sites with AQI above 100", "schools where AQI exceeds 50", "how many sites have AQI less than 80") → ALWAYS select "site_aqi_by_category". Extract: sectorName="Public & Govt-School" if "school" is mentioned, regionName if a region is mentioned, minValue from the number when "above/more than/greater than/over/exceeds/at least", maxValue from the number when "below/less than/under/at most". NEVER select "schools_latest_pollutant" for numeric threshold questions — that template cannot filter by AQI value.
               IMPORTANT — AQI category with region: if the user asks for sites/locations WHERE AQI IS a quality level (good, moderate, unhealthy, hazardous, etc.) even with a region filter (e.g. "Al Ain sites where AQI is good", "list Abu Dhabi sites with good AQI", "sites in Al Dhafra where AQI is moderate") → ALWAYS select "site_aqi_by_category". Extract aqiCategory AND regionName. NEVER select sites_by_region or list_stations for these — those return site lists without AQI data.
               IMPORTANT — schools + AQI category: if the user asks about SCHOOLS (e.g. "schools where AQI is good", "school sites with moderate AQI") AND mentions an AQI quality level → select "site_aqi_by_category" with aqiCategory set AND sectorName="Public & Govt-School". Do NOT select schools_latest_pollutant for this case.
            9. If the user asks for the current Data Success Rate, DSR value, data availability %, how many devices are active, active device count, or critical site count → select "data_success_rate". Extract year if mentioned (e.g. "2025", "2026"); omit year to default to the current year. NEVER select faq_answer for questions asking for the CURRENT or ACTUAL value of the DSR or active device count — those must come from the live API.
            9j. If the user asks which region had the poorest/worst/best air quality over the last N years considering data completeness (multi-year comparison) → select "region_aqi_trend_yearly". Extract years (default 3). NEVER select region_aqi_geographical or region_aqi_single_year for multi-year questions.
            9i. If the user asks which region performed best or worst in a SPECIFIC YEAR (e.g. "Which region performed best in 2025 when both AQI and data completeness are considered?") → select "region_aqi_single_year". Extract year. NEVER select region_aqi_trend_yearly (that covers multiple years).
            9h. If the user asks what drove the worst air-quality month at a specific site in a given year (e.g. "What drove the worst air-quality month at Al Saad Indian School in 2025?") → select "worst_month_pollutants". Extract stationName and year. NEVER select faq_answer or site_aqi_trend_yearly for this.
            9g. If the user asks whether air quality at a specific site has improved or worsened over the last N years, or asks for a yearly AQI trend/history at a site (e.g. "Has air quality at Al Saad Indian School improved or worsened over the last three years?") → select "site_aqi_trend_yearly". Extract stationName and years (default 3). NEVER select faq_answer or top_sites_aqi_yearly for this — those do not give a per-site trend.
            9f. If the user asks to compare the indoor AQI at a site with the nearest ambient (outdoor) station (e.g. "Compare the current indoor AQI at Al Saad Indian School with the nearest ambient station") → select "indoor_ambient_compare". Extract stationName. The system fetches the nearest ambient station automatically from Abu Dhabi SDI. NEVER select region_aqi_geographical or site_aqi_single for this.
            9e. If the user asks which devices had the lowest Data Success Rate last month (e.g. "which devices had the lowest DSR last month?", "worst data success rate last month") → select "lowest_dsr_devices". Extract topN if mentioned (default 10). NEVER select data_success_rate (that returns overall network DSR, not per-device) or device_compliance_rate (that requires a specific stationId).
            9d. If the user asks which device contributed most to the highest site AQI at a school or site last month (e.g. "which device contributed most to the highest site AQI at Al Saad Indian School last month?") → select "device_peak_aqi_contribution". Extract stationName only — the date range is always last month (hardcoded in SQL). NEVER select device_last_reading, compare_two_stations, or schools_latest_pollutant for this.
            9c. If the user asks to compare the two devices at a specific school or site for a specific month (e.g. "compare the two devices at Al Saad Indian School for September 2026", "device comparison at Al Naeem School for August") → select "compare_devices_monthly". Extract stationName (site/school name), month (as integer 1–12), and year (4-digit). NEVER select compare_two_stations or device_last_reading for this — those do not group by device per month.
            9b. If the user asks which sites had the HIGHEST average AQI last year, in a specific year, or wants a worst/most-polluted year-based ranking (e.g. "which 10 sites had the highest average AQI last year?", "top sites by AQI in 2024", "worst sites AQI 2023") → select "top_sites_aqi_yearly". Extract topN (default 10) and year (default: current year minus 1). If the user asks for LOWEST or BEST average AQI last year (cleanest, safest, best air quality) → select "bottom_sites_aqi_yearly" instead. NEVER select faq_answer, site_aqi_ranking, or highest_aqi_today for year-based AQI ranking questions.
            10. If the user asks to LIST or SHOW devices under a SPECIFIC SITE or SCHOOL name (e.g. "devices under Al Naeem School", "devices at Al Saad Indian School and Al Bateen School") → select "devices_with_readings". Extract stationName as the site name or a keyword from it (e.g. "Al Naeem" or "Naeem"). If the user names MULTIPLE sites, join them with a space so the LIKE filter catches any partial match (e.g. stationName="Naeem Saad" won't work — instead extract the FIRST site name only; the user will see all devices grouped by site). NEVER select sites_by_sector or list_stations for device listing questions.
            If the user asks about devices in a REGION or SECTOR only (no specific site name) → select "devices_by_filter". Extract regionName and/or sectorName. This also handles "how many active devices in Abu Dhabi" — extract statusFilter if active/inactive is mentioned.
            11. If the user asks to LIST or SHOW sites/stations across ALL regions (e.g. "give me the sites under all regions", "show sites region wise", "sites in each region", "all sites by region") → select "sites_all_regions". NEVER select sites_count_by_region (that only shows counts), schools_latest_pollutant, or sector_latest_pollutant for this.
            11e. If the user asks HOW MANY sites are in ALL THREE regions together (e.g. "how many sites are there in Abu Dhabi, Al Ain and Al Dhafra?", "how many sites in Abu Dhabi, Al Ain and Al Dhafra", "total sites in all three regions") → select "sites_count_by_region". No params needed — it returns counts for all regions. NEVER select count_sites_in_region (that takes a single regionName and would need to run 3 times). NEVER select sites_all_regions (that lists site names, not counts).
            11a. If the user asks how many SECTORS exist, what sectors are in HAWAQM, or list of sectors (with NO specific sector name) → select "list_sectors". NEVER select sites_count_by_sector (that shows sites per sector, not the sector list).
            11b. If the user asks how many SUB-SECTORS exist, what sub-sectors are in HAWAQM, or list of sub-sectors → select "list_subsectors". NEVER select sites_by_sector or list_sectors for sub-sector questions.
            11c. If the user asks for a PIE CHART or DONUT CHART or DISTRIBUTION CHART of sites or schools by AQI category (e.g. "create a pie chart of schools by their current AQI category", "show me a pie chart of AQI distribution", "AQI category breakdown as a pie chart") → select "schools_aqi_pie". If the user mentions "schools" or "school sites", extract sectorName="Public & Govt-School". If a region is mentioned, extract regionName. NEVER select site_aqi_by_category or site_aqi_all for pie/donut/distribution chart questions.
            11d. If the user asks about DEVICES PER SITE — including:
              - Sites with no/zero devices ("which sites don't have any devices", "sites without devices", "sites with no devices")
              - Sites with more/fewer than N devices ("sites with more than 2 devices", "sites with at least 1 device")
              - Active/inactive device count per site ("active and inactive devices per site", "device status per site")
              - Last reading per site ("last reading from devices per site", "when did each site last send data")
              - Best AQI device per site ("which device is better in AQI", "which device has lowest AQI")
              → select "site_device_summary".
              For "no devices" / "zero devices" → set minDeviceCount=0, maxDeviceCount=0.
              For "more than N devices" → set minDeviceCount=N+1, maxDeviceCount=9999.
              For "fewer than N devices" → set minDeviceCount=0, maxDeviceCount=N-1.
              For "at least N devices" → set minDeviceCount=N, maxDeviceCount=9999.
              For general device summary (no count filter) → omit both params (defaults: min=0, max=9999).
              NEVER select online_devices, offline_devices, active_device_count, or devices_by_filter for device-per-site questions.

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

            For dates: use ISO format YYYY-MM-DD. Today's date is provided in the user message — use it to resolve ALL relative date expressions. Rules:
              - "today"                          → startDate = today, endDate = today
              - "yesterday" / "day before today" → startDate = today−1, endDate = today−1
              - "day before yesterday"            → startDate = today−2, endDate = today−2
              - "1 day back" / "1 day ago"        → startDate = today−1, endDate = today−1
              - "2 days back" / "2 days ago"      → startDate = today−2, endDate = today−2
              - "N days back" / "N days ago"      → startDate = today−N, endDate = today−N
              - "last N days"                     → startDate = today−N, endDate = today
              - "1 week back" / "1 week ago"      → startDate = today−7, endDate = today−7
              - "N weeks back" / "N weeks ago"    → startDate = today−(N×7), endDate = today−(N×7)
              - "last N weeks"                    → startDate = today−(N×7), endDate = today
              - "1 month back" / "1 month ago" / "last month" → startDate = first day of previous month, endDate = last day of previous month
              - "N months back" / "N months ago"  → startDate = first day of (today − N months), endDate = last day of that month
              - "last N months"                   → startDate = today−(N months), endDate = today
              - "1 year back" / "1 year ago" / "last year" → startDate = first day of previous year (Jan 1), endDate = last day of previous year (Dec 31)
              - "N years back" / "N years ago"    → startDate = Jan 1 of (current year − N), endDate = Dec 31 of that year
              - "last N years"                    → startDate = today−(N years), endDate = today
              - "this month"                      → startDate = first day of current month, endDate = today
              - "this year"                       → startDate = Jan 1 of current year, endDate = today
              - "this week"                       → startDate = most recent Monday, endDate = today
              Always compute using the "Today's date" provided in the user message.
            For columns: use lowercase snake_case (e.g. pm25, not PM2.5).
            For year: extract a 4-digit year if the user mentions one (e.g. "2025", "last year"). If no year is mentioned, omit it — the system will default to the current year automatically.
            For regionName: extract the region the user mentions (words like "in", "under", "for", "within", "at" before a region name all indicate the region). Map to: "Abu Dhabi" (also: abudhabi, abu-dhabi, abudabi), "Al Ain" (also: alain, al-ain), "Al Dhafra" (also: aldhafra, al-dhafra, dhafra). If the user asks about all regions or does not mention a specific region, omit regionName.
            For sectorName: there are exactly 3 sectors. Map ALL user variations to one of these three canonical values:
              → "Commercial": "commercial", "commercial sector", "commercial sites"
              → "Public & Govt-School": ANY phrase containing "school", "schools", "govt", "gov", "government", "public & govt", "public & gov", "public and gov", "public and govt", "public and government", "govt-school", "gov-school", "government school", "govt school", "gov school", "educational", "private and gov-school", "private and govt school"
              → "Residential": "residential", "residential sector", "residential sites"
            IMPORTANT: "public & Gov School", "Public and Government School", "Public & Government school", "gov school", "govt school" ALL map to "Public & Govt-School".
            IMPORTANT: When the template is "site_aqi_by_category" and the user mentions "schools" or "school sites", ALWAYS extract sectorName="Public & Govt-School".
            IMPORTANT: When the template is "sites_by_sector" and the user mentions "schools", "school", "different schools", "schools present", "which schools", "list schools" → ALWAYS extract sectorName="Public & Govt-School".
            If the user says something that doesn't match any of these three, omit sectorName.
            For parameterName: map common user words to the allowed parameter name — "co2", "carbon dioxide" → "CO2"; "pm2.5", "fine particles", "fine dust" → "PM2.5"; "pm10", "coarse particles", "coarse dust" → "PM10"; "co", "carbon monoxide" → "CO"; "aqi", "air quality index" → "AQI Index"; "no2", "nitrogen dioxide" → "NO2"; "so2", "sulphur dioxide", "sulfur dioxide" → "SO2"; "o3", "ozone" → "O3"; "tvoc", "voc", "volatile organic" → "VOC"; "ch2o", "formaldehyde" → "CH2O"; "temperature", "temp" → "Temperature"; "humidity" → "Humidity"; "noise", "sound" → "Noise".
            IMPORTANT — pollutant groups: if the user asks for "physical" readings/pollutants/parameters, do NOT extract a single parameterName — omit it (the system will filter to physical group: PM2.5, PM10, Temperature, Humidity, Noise). If the user asks for "chemical" readings/pollutants/parameters, omit parameterName as well (system filters to chemical group: CO, CO2, NO2, SO2, O3, CH2O, VOC). Only extract a specific parameterName when the user names ONE specific pollutant.
            For stationName: extract the site or station name the user mentions. CRITICAL — strip any region qualifier appended after the site name. Examples:
              - "Al Saad Indian School in Al Ain" → stationName = "Al Saad Indian School"
              - "Abu Dhabi Residential site" → stationName = "Abu Dhabi Residential"
              - "Al Ain Commercial Institutional" → stationName = "Al Ain Commercial Institutional" (Al Ain is part of the site name here, not a region qualifier)
              - "readings at Al Bateen School in Abu Dhabi" → stationName = "Al Bateen School"
            If the user says ONLY a region name ("Abu Dhabi", "Al Ain", "Al Dhafra") with no site name, do NOT extract stationName — that is a region query, not a site query.
            Never extract or guess a numeric ID — always use the name.
            For deviceName: extract the device name exactly as stated in the CURRENT question only. Real device name patterns in this system are: "BA 0001", "BA 0005", "BA 0010" (prefix "BA" + space + 4-digit number) and "SEI100M 0014", "SEI100M 0085", "SEI100M 0148" (prefix "SEI100M" + space + 4-digit number). Users may omit the space or use different casing — extract whatever they wrote (e.g. "ba0010", "BA0010", "BA 0010", "sei100m 0014" are all valid). Never guess or invent a device name — only extract what is explicitly stated in the current question.
            For parameterName (device readings only): if the user asks for a specific parameter, map it — "aqi", "air quality index" → "AQI Index"; "pm2.5", "fine particles", "fine dust" → "PM2.5"; "pm10", "coarse particles" → "PM10"; "co2", "carbon dioxide" → "CO2"; "co", "carbon monoxide" → "CO"; "no2", "nitrogen dioxide" → "NO2"; "so2", "sulphur dioxide", "sulfur dioxide" → "SO2"; "o3", "ozone" → "O3"; "temperature", "temp" → "Temperature"; "humidity" → "Humidity"; "voc", "tvoc", "volatile organic" → "VOC"; "noise", "sound" → "Noise"; "ch2o", "formaldehyde" → "CH2O".
            CRITICAL — pollutant group words for device queries: if the user says "physical" (or "physical readings", "physical pollutants", "physical parameters"), omit parameterName — the controller will return PM2.5, PM10, Temperature, Humidity, Noise. If the user says "chemical" (or "chemical readings", "chemical pollutants", "chemical parameters"), omit parameterName — the controller will return CO, CO2, NO2, SO2, O3, CH2O, VOC. Only extract a specific parameterName when the user names a single explicit pollutant.
            If the user asks for "reading", "readings", "all readings", "all parameters", "all data", "physical", or "chemical" — omit parameterName entirely.
            For statusFilter (devices_by_filter only): extract "Active" if user says "active", "online", "working", "sending"; extract "Inactive" if user says "inactive", "offline", "not sending", "down"; omit if user asks for all devices with no status preference.
            For aqiCategory (site_aqi_by_category only): map the user's words to one or more of the six categories below.
              If the user asks for MORE THAN ONE category (e.g. "Moderate and Very Unhealthy", "Good or Unhealthy"), output ALL matched categories joined by "|" (pipe).
              Examples: "Moderate and Very Unhealthy" → "Moderate|Very Unhealthy"; "Good or Hazardous" → "Good|Hazardous"; "just Moderate" → "Moderate".
              Category mapping (use MOST SPECIFIC match for each word/phrase):
              → "Good"                          : good, healthy, safe, clean, great, excellent, fine, acceptable, green
              → "Moderate"                      : moderate, average, medium, okay, ok, acceptable (AQI 51–100)
              → "Unhealthy for Sensitive Groups": sensitive, sensitive groups, unhealthy for sensitive, vulnerable, kids, children, elders, elderly, old people, sick, diseased, patients, asthma, heart, lung, at-risk, at risk, vulnerable groups, special groups, special needs, weak, fragile, very vulnerable, dangerous for kids, bad for kids, bad for elderly, harmful for children, harmful for elderly, harmful for sensitive
              → "Unhealthy"                     : unhealthy, poor, bad, bad air, bad aqi, not good, harmful, not safe, polluted, dirty air, bad quality
              → "Very Unhealthy"                : very unhealthy, very bad, very poor, very polluted, extremely bad, very harmful, very dangerous, severe, serious, alarming
              → "Hazardous"                     : hazardous, dangerous, extremely dangerous, deadly, toxic, emergency, catastrophic, worst, extreme, critical air, red alert, purple, maroon
              → "critical" (AQI > 100, all tiers above Moderate combined): critical, hotspot, hotspots, most polluted, above 100, over 100, exceeding 100, priority, high risk
              If user says "all" or doesn't specify any category, omit aqiCategory entirely.
            For minValue (site_aqi_by_category and schools_latest_pollutant): extract a numeric AQI lower bound when user says "more than X", "greater than X", "above X", "over X", "exceeding X". Use the number only (e.g. "AQI more than 30" → minValue: "30", "above 50" → minValue: "50").
            For maxValue (site_aqi_by_category and schools_latest_pollutant): extract a numeric AQI upper bound when user says "less than X", "below X", "under X", "at most X". Use the number only (e.g. "AQI less than 100" → maxValue: "100").
            IMPORTANT — for site_aqi_by_category with AQI range: when user says "AQI more than 30 in Al Ain" → minValue="30", regionName="Al Ain". When "AQI less than 100 in Abu Dhabi" → maxValue="100", regionName="Abu Dhabi". When "AQI between 50 and 100" → minValue="50", maxValue="100". Do NOT extract aqiCategory when the user specifies a numeric range — use minValue/maxValue instead.
            IMPORTANT — regionName for site_aqi_by_category: extract regionName when user mentions a region ("in Al Ain", "in Abu Dhabi", "in Al Dhafra", "under Abu Dhabi", "within Al Ain") for AQI category or AQI range queries. Map to: "Abu Dhabi", "Al Ain", "Al Dhafra".
            IMPORTANT — pronoun resolution: if the user says "this site", "this station", "this location", "it", or similar pronouns without naming a site, look at the conversation history above to find the most recently mentioned site or station name, and use that as stationName.
            For deviceName pronoun resolution ONLY: if the user says "this device", "it", or similar pronouns WITHOUT explicitly naming a device in the current question, look at history for the most recently mentioned device name. But if the current question contains an explicit device name, ALWAYS use that — never the history device.
            For interval (device_reading_history only): map user time words to allowed values:
              → "5min"   : "5 min", "5minute", "5-minute", "raw", "5 minutes", "5min interval", "5-minute average", "5 min reading"
              → "1hour"  : "1 hour", "1h", "1H", "hourly", "1-hour", "one hour", "1 hour average", "1H average", "1H reading", "1 hour reading", "1H averages", "hourly average", "hourly reading", "at 1H", "1H interval", "using 1H", "1 hour interval"
              → "8hour"  : "8 hour", "8h", "8H", "8-hour", "eight hour", "8 hour average", "8H average", "8H reading", "8 hour reading", "8H interval", "8 hour interval"
              → "24hour" : "24 hour", "24h", "24H", "24-hour", "daily", "day", "one day", "24 hour average", "daily average", "daily reading", "24H interval", "per day"
              → "monthly": "monthly", "month", "1 month", "one month", "monthly average", "monthly reading", "per month"
              → "yearly" : "yearly", "year", "annual", "1 year", "one year", "yearly average", "per year", "annually"
              IMPORTANT — interval + targetTime coexist: "PM2.5 at 8AM at 1H average" → interval="1hour", targetTime="8:00AM". Extract BOTH when both are present.
              IMPORTANT: "average"/"avg"/"mean" alone (without a time unit) do NOT imply an interval — omit interval. Only map to an interval when a time unit (hour, day, month, year) is explicitly stated alongside average/avg/mean.
              If no interval word is found in the question, omit the interval parameter entirely.
            For targetTime (device_reading_history only): extract when the user mentions ANY specific clock time. Output in HH:MMAM/PM or 24-hour HH:MM format.
              "at 8AM", "at 8:00AM", "at 08:00" → "8:00AM"
              "at 2:02AM", "at 2:02 AM" → "2:02AM"
              "at 4AM" → "4:00AM"
              "at 3AM" → "3:00AM"
              "at 2:59AM" → "2:59AM"
              "at 4:30PM", "at 16:30" → "4:30PM"
              "at 14:00", "at 2PM" → "14:00"
              "at noon", "at 12PM" → "12:00PM"
              "at midnight", "at 12AM" → "12:00AM"
              "at 1:05" → "1:05"
              "at 13:00" → "13:00"
              IMPORTANT: always also extract startDate as the calendar date (ISO YYYY-MM-DD) using the Today's date in the user message:
              "yesterday at 4AM" → startDate: today−1, targetTime: "4:00AM"
              "2 days back at 8AM" → startDate: today−2, targetTime: "8:00AM"
              "1 week back at 3PM" → startDate: today−7, targetTime: "3:00PM"
              "11-Aug-2026 at 2:02AM", "2026-08-11 at 2AM" → startDate: "2026-08-11", targetTime: "2:02AM"
              "on 2026-07-16 at 8AM" → startDate: "2026-07-16", targetTime: "8:00AM"
              "today at noon" → startDate: today, targetTime: "12:00PM"
              The controller combines startDate (day) + targetTime (clock) to snap to the nearest reading.
              Only extract targetTime for concrete clock times. Do NOT extract for "morning", "evening", "night", "afternoon". Omit targetTime if no specific time is given.
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

        var today = DateTime.UtcNow.Date;
        var todayStr = $"Today's date (UTC): {today:yyyy-MM-dd}";

        return $"Question: {question}\n{todayStr}\n{dateStr}\n{regionStr}";
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
