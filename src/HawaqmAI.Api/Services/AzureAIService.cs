using System.Security.Cryptography.X509Certificates;
using System.Text;
using System.Text.Json;
using Azure.AI.OpenAI;
using Azure.Identity;
using HawaqmAI.Api.Configuration;
using HawaqmAI.Api.Models;
using Microsoft.AspNetCore.Http;
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
    private readonly IHttpContextAccessor _httpContextAccessor;
    private readonly ChatClient _chatClient;      // large model — summarization
    private readonly ChatClient _fastChatClient;  // fast/cheap model — routing, param extraction

    public AzureAIService(IOptions<AzureAIOptions> options, IHttpContextAccessor httpContextAccessor)
    {
        _options = options.Value;
        _httpContextAccessor = httpContextAccessor;
        _chatClient = BuildChatClient(_options.DeploymentName);

        // If a separate fast model is configured, build a second client for it.
        // Otherwise fall back to the same model (no perf gain but no regression either).
        var fastDeploy = string.IsNullOrWhiteSpace(_options.FastDeploymentName)
            ? _options.DeploymentName
            : _options.FastDeploymentName;
        _fastChatClient = BuildChatClient(fastDeploy);

        _log.Information("Azure AI: summary model={Main}, fast model={Fast}",
            _options.DeploymentName, fastDeploy);
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
        // Don't pass history when candidates include device_live_reading — history causes
        // the LLM to pick a device name from a previous turn instead of the current question
        var hasDeviceTemplate = candidates.Any(c => c.Id == "device_live_reading");
        var messages = BuildMessages(systemPrompt, history, userPrompt, includeHistory: !hasDeviceTemplate);

        var response = await CallWithRetryAsync(messages, ct, client: _fastChatClient);
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

        var response = await CallWithRetryAsync(messages, ct, client: _fastChatClient);
        return ParseParameterResponse(response);
    }

    // ── Format templates keyed by template ID ───────────────────────────────
    // These match the exact Q&A format patterns specified by users in response-formats.json.
    // Placeholder syntax: {ColumnName} is replaced by the LLM from the actual data values.
    private static readonly Dictionary<string, string> _formatTemplates = new(StringComparer.OrdinalIgnoreCase)
    {
        // Offline/inactive devices — matches dashboard 1440-min (24-hr) cutoff
        ["offline_devices"] = """
            Data columns: "DeviceName", "StationName", "RegionName", "SectorName", "LastActive".
            Each row is a device that has not sent a reading in the last 24 hours (offline/inactive).

            FORMAT: "There are {N} offline device(s) right now. The table below lists them with their last known reading time."
            If no rows: "All devices are currently online — no devices have been offline in the last 24 hours."

            NEVER list individual devices in text — the table handles that.
            NEVER output placeholder text — always use the actual row count from the data.
            """,

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

        // Priority hotspots from Executive Dashboard
        ["priority_hotspots"] = """
            Two possible data shapes:

            CASE 1 — Hotspots exist: data has "Site Name" column with real site names and "Site ID".
            FORMAT: "There are {count} site(s) currently in the Hazardous air quality category (AQI above 300):"
            Then bullet list: • {Site Name} (ID: {Site ID})

            CASE 2 — No hotspots: data has "Status" = "No priority hotspots currently" and "Network AQI".
            FORMAT: "There are currently no sites in the Hazardous air quality category (AQI > 300). The overall network Air Quality Index is {Network AQI} ({Air Quality Category})."
            Do NOT show a table for this case. Output plain text only.

            NEVER output placeholder text — always use actual values from the data.
            """,

        // AQI category filter — columns: SiteName, RegionName, AQI, AQICategory, LastUpdated
        ["site_aqi_by_category"] = """
            Data columns: "SiteName", "RegionName", "AQI", "AQICategory", "LastUpdated".
            AQI categories: 0–50 Good, 51–100 Moderate, 101–150 Unhealthy for Sensitive Groups, 151–200 Unhealthy, 201–300 Very Unhealthy, 301+ Hazardous.
            The table below will show the full list — write ONE intro sentence only.

            SITE LABEL: If the user asked specifically about "schools" use "school site(s)". If about "commercial" use "commercial site(s)". If about "residential" use "residential site(s)". Otherwise use "site(s)".
            REGION SUFFIX: " in {RegionName}" if a region was filtered, otherwise omit.
            SECTOR SUFFIX: " in the {SectorLabel} sector" if a sector was filtered and no region, otherwise omit when region is already stated.

            If user asked for AQI RANGE (above X / below X / more than X / less than X / between X and Y)
            OR a COMPARATIVE category (worse than Moderate, better than Unhealthy, not Good, etc.):
            For comparative: map back to human language — "worse than Moderate" means AQI > 100 (i.e. Unhealthy for Sensitive Groups or higher).
            FORMAT: "There are {N} {site label} with air quality worse than Moderate{region suffix}. The table below shows the full list."
            Examples:
              "There are 7 school site(s) with air quality worse than Moderate in Abu Dhabi. The table below shows the full list."
              "There are 12 site(s) with air quality worse than Good (AQI above 50){region suffix}. The table below shows the full list."
              "There are 4 site(s) with AQI above 30 in Al Ain. The table below shows the full list."
              "There are 2 site(s) with AQI below 100 in Abu Dhabi. The table below shows the full list."
              "There are 3 site(s) with AQI between 50 and 150. The table below shows the full list."
            If no rows: "No {site label} currently match the requested AQI range{region suffix}."

            If ONE specific category was requested (e.g. "Moderate", "Unhealthy"):
            FORMAT: "There are {N} {site label} currently with {AQICategory} air quality{region suffix}. The table below shows the full list."
            If no rows: "No {site label} currently have {AQICategory} air quality{region suffix}."

            If MULTIPLE categories were requested (e.g. "Moderate and Very Unhealthy"):
            Count sites per category from the data.
            FORMAT: "There are {N1} {site label} with {Cat1} and {N2} {site label} with {Cat2} air quality{region suffix}. The table below shows the full list."

            For "critical" or "hotspot" (Hazardous, AQI > 300):
            FORMAT: "There are {N} air quality hotspot(s){region suffix} in the Hazardous category (AQI above 300). The table below shows the full list."
            If empty: "No {site label}{region suffix} are currently in the Hazardous category (AQI above 300)."

            If NO filter at all:
            FORMAT: "Current AQI status across {N} accessible {site label}. The table below shows all sites."

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

        // Sites by region pie/bar — columns: Region, Count
        ["sites_by_filter_chart"] = """
            Data columns: "Region" (region name), "Count" (number of sites in that region).
            The chart below will visualise the distribution — write ONE short summary sentence only.
            FORMAT: "Here is the breakdown of {TotalCount} site(s) across {N} region(s). The chart below shows the count per region."
            {TotalCount} = sum of all Count values. {N} = number of rows.
            NEVER list individual sites or regions in text — the chart handles that.
            NEVER output placeholder text — always use actual totals from the data.
            """,

        // Sites by sector pie/bar — columns: Sector, Count
        ["sites_by_filter_chart"] = """
            Data columns: "Sector" (sector name), "Count" (number of sites in that sector).
            The chart below will visualise the distribution — write ONE short summary sentence only.
            FORMAT: "Here is the breakdown of {TotalCount} site(s) across {N} sector(s). The chart below shows the count per sector."
            {TotalCount} = sum of all Count values. {N} = number of rows.
            NEVER list individual sites or sectors in text — the chart handles that.
            NEVER output placeholder text — always use actual totals from the data.
            """,

        // AQI by region bar/pie — columns: Region, AQI (average)
        ["aqi_by_region_chart"] = """
            Data columns: "Region" (region name), "AQI" (average AQI index for that region).
            The chart below will visualise the AQI comparison — write ONE short summary sentence only.
            FORMAT: "Here is the average AQI across {N} region(s). The chart below shows the AQI level per region."
            {N} = number of rows.
            NEVER list individual regions or values in text — the chart handles that.
            NEVER output placeholder text — always use actual counts from the data.
            """,

        // Sites by status pie/bar — columns: Status (Active/Offline), Count
        ["sites_by_status_chart"] = """
            Data columns: "Status" (Active or Offline), "Count" (number of sites with that status).
            The chart below will visualise the breakdown — write ONE short summary sentence only.
            FORMAT: "Here is the current status breakdown across {TotalCount} site(s). The chart below shows Active vs Offline counts."
            {TotalCount} = sum of all Count values.
            NEVER list individual sites in text — the chart handles that.
            NEVER output placeholder text — always use actual totals from the data.
            """,

        // Pollutant by sector bar — columns: Sector, Value (average pollutant)
        ["pollutant_by_sector_chart"] = """
            Data columns: "Sector" (sector name), "Value" (average pollutant value for that sector).
            The chart below will visualise the comparison — write ONE short summary sentence only.
            FORMAT: "Here is the average {ParameterName} across {N} sector(s). The chart below shows the value per sector."
            {N} = number of rows. {ParameterName} = the pollutant name from the user's question (e.g. PM2.5, AQI).
            NEVER list individual values in text — the chart handles that.
            NEVER output placeholder text — always use actual counts from the data.
            """,

        // Devices by model pie/bar — columns: Model, DeviceCount
        ["devices_by_model_chart"] = """
            Data columns: "Model", "DeviceCount".
            The chart below will visualise the distribution — write ONE short summary sentence only.
            FORMAT: "Here is the distribution of {TotalCount} device(s) across {N} model(s). The chart below shows the count per model."
            {TotalCount} = sum of all DeviceCount values. {N} = number of rows.
            NEVER list individual models in text — the chart handles that.
            NEVER output placeholder text — always use actual totals from the data.
            """,

        // Sites by sub-sector pie/bar — columns: SubSector, SiteCount
        ["sites_by_subsector_chart"] = """
            Data columns: "SubSector", "SiteCount".
            The chart below will visualise the distribution — write ONE short summary sentence only.
            FORMAT: "Here is the breakdown of {TotalCount} school site(s) across {N} sub-sector(s). The chart below shows the count per sub-sector."
            {TotalCount} = sum of all SiteCount values. {N} = number of rows.
            NEVER list individual sub-sectors in text — the chart handles that.
            NEVER output placeholder text — always use actual totals from the data.
            """,

        // Sites by AQI level pie — columns: Category, Count
        ["sites_aqi_level_chart"] = """
            Data columns: "Category" (AQI category name), "Count" (number of sites in that category).
            The chart below will visualise the AQI level distribution — write ONE short summary sentence only.
            FORMAT: "Here is the current AQI level distribution across {TotalCount} site(s). The chart below shows the count per AQI category."
            {TotalCount} = sum of all Count values.
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

            GOOD OR BAD / QUALITY JUDGEMENT QUESTION — if the user asks whether the air is good or bad, good or poor, how is the air, is the air quality good, is the AQI good/bad, is it ok/fine/healthy/unhealthy at a site:
            Find the AQI Index row. Use the AQI value to give a direct verdict based on MOCCAE categories, then state the value.
            AQI verdict map: 0–50 = "good" | 51–100 = "acceptable" | 101–150 = "a concern for sensitive groups" | 151–200 = "unhealthy" | 201–300 = "very unhealthy" | 301+ = "hazardous"
            FORMAT: "The air at {Site Name} is currently {verdict} — AQI {Value} ({AQI category}) as of {Last Updated}."
            Examples:
              "The air at Latifa Mohammed Ahmed Al Dhaheri is currently good — AQI 38 (Good) as of 2026-10-01 14:35:00."
              "The air at Al Reyada School is currently a concern for sensitive groups — AQI 118 (Unhealthy for Sensitive Groups) as of 2026-10-01 09:10:00."
              "The air at Al Bateen School is currently unhealthy — AQI 165 (Unhealthy) as of 2026-10-01 11:25:00."
            If no AQI data: "No current AQI reading is available for {Site Name}. The device may be offline."
            Keywords that trigger this format: good or bad, is it good, is it bad, is the air good, is the air bad, how is the air, air quality good, air quality bad, is the aqi good, is it healthy, is it ok, is it fine.

            SAFETY / HEALTH / ATTENDANCE QUESTION — if the user asks whether the site is safe, good for health, safe for children/elderly/disabled/vulnerable/sensitive groups, or whether people should attend, go, visit, or use the space:
            Find the AQI Index row and 1–2 key pollutant readings (e.g. PM2.5, CO2) from the data.
            FORMAT (use actual values — never placeholder text):
            "I can share the measurements: AQI {AQI_VALUE} ({AQI_LEVEL}) as of {Last Updated}, with PM2.5 at {PM2.5_VALUE} µg/m³ and CO2 at {CO2_VALUE} PPM. Whether the building is safe to attend is a decision for the responsible health authority — the data alone cannot decide that. I can prepare the full data picture to support the decision."
            If PM2.5 or CO2 is not available, omit that pollutant and use whatever key pollutant IS available.
            If no AQI data at all: "No current readings are available for {Site Name}. The device may be offline. Whether the building is safe to attend is a decision for the responsible health authority."
            Keywords that trigger this format: safe, safety, attend, attendance, suitable, good for children, good for elderly, good for disabled, can children, should children, can people, should people.

            User asked for AQI or air quality category (NOT a safety/quality question) → find the row where Parameter = "AQI Index":
            FORMAT: "The current AQI at {Site Name} is {Value}, classified as {AQI category}, last updated at {Last Updated}."
            Example: "The current AQI at Al Saad Indian School is 72, classified as Moderate, last updated at 2026-09-15 18:45:00."

            User asked for ONE specific pollutant (CO2, PM2.5, PM10, CO, NO2, O3, TVOC) → find that parameter's row:
            FORMAT: "The latest {Parameter} reading at {Site Name} is {Value} {Unit}, recorded at {Last Updated}."
            Example: "The latest CO2 reading at Al Bateen School is 487 PPM, recorded at 2026-09-01 18:45:00."

            User asked for TWO or THREE specific parameters (e.g. temperature and humidity, NO2 and O3 and CO, PM10 and PM2.5):
            CRITICAL: For EACH parameter, find the row where the "Parameter" column EXACTLY matches the parameter name.
            Parameter name aliases: "temp" or "temperature" → row where Parameter = "Temperature". "humidity" → row where Parameter = "Humidity". "PM2.5" → row where Parameter = "PM2.5". "PM10" → row where Parameter = "PM10". Never swap or guess values.
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

        // Historical daily averages for a specific site on a past date
        ["site_daily_averages"] = """
            Data columns: "StationName", "ParameterName", "DailyAverage", "MainPollutant", "DeviceCount", "Date".
            NEVER show StationName, Date, DeviceCount, or MainPollutant in the table — those go in text only.
            NEVER output placeholder text — always use actual values from the data.

            MainPollutant format rules:
            - If DeviceCount = 1 (single device): MainPollutant is a plain parameter name (e.g. "CO2"). Use it as-is.
              FORMAT: "Here are the 24-hour average readings at {StationName} on {Date}. The main pollutant for the day was {MainPollutant}."
            - If DeviceCount > 1 (multiple devices): MainPollutant is a comma-separated list of "DeviceName - ParameterName" pairs (e.g. "BA 0001 - CO2, BA 0002 - NO2"). List each on its own line.
              FORMAT: "Here are the 24-hour average readings at {StationName} on {Date}. Main pollutant per device:\n• {DeviceName1} — {ParameterName1}\n• {DeviceName2} — {ParameterName2}"
            - If MainPollutant is NULL or missing for all rows, omit the main pollutant sentence entirely.

            The table must have exactly two columns: Parameter | Daily Average
            Fill one row per data row using ParameterName and DailyAverage only.

            If no rows: "No daily average readings found for that site on the selected date."
            """,

        // School / sector pollutant queries — data columns: Site Name, Region, Parameter, Value, Unit, Last Updated
        ["schools_latest_pollutant"] = """
            Data columns: "Site Name", "Region", "Parameter", "Value", "Unit", "Last Updated".
            The data rows are pre-sorted and pre-limited: highest first for "top N / highest N / worst N", lowest first for "bottom N / lowest N / best N".

            If user asked for "top N" / "highest N" / "worst N" (sortOrder=desc):
            "Here are the top {N} schools with the highest {Parameter} right now:"
            Then one line per row: "{Rank}. {Site Name} ({Region}): {Value} {Unit}"
            If Parameter is "AQI Index", omit the Unit.

            If user asked for "bottom N" / "lowest N" / "best N" (sortOrder=asc):
            "Here are the {N} schools with the lowest {Parameter} right now:"
            Then one line per row: "{Rank}. {Site Name} ({Region}): {Value} {Unit}"
            If Parameter is "AQI Index", omit the Unit.

            Otherwise:
            "There are {Count} school site(s) with {Parameter} data."
            Then one line per row: "{Site Name} in {Region}: {Parameter} = {Value} {Unit} (last updated: {Last Updated})."
            If Parameter is "AQI Index", omit the Unit (do NOT show hPa or any unit for AQI).
            If no rows: "There are 0 school site(s) with {Parameter} data."
            Keep under 150 words. List every site from the data.
            """,

        ["sector_latest_pollutant"] = """
            Data columns: "Site Name", "Region", "Parameter", "Value", "Unit", "Last Updated".
            The data rows are already sorted by AQI/Value descending when the user asked for "top N".

            If user asked for "top N" / "highest N" / "worst N":
            "Here are the top {N} {SectorName} sites with the highest {Parameter} right now:"
            Then one line per row: "{Site Name} ({Region}): {Value} {Unit}"
            If Parameter is "AQI Index", omit the Unit.

            Otherwise:
            "There are {Count} {SectorName} site(s) with {Parameter} data."
            Then one line per row: "{Site Name} in {Region}: {Parameter} = {Value} {Unit} (last updated: {Last Updated})."
            If Parameter is "AQI Index", omit the Unit (do NOT show hPa or any unit for AQI).
            If no rows: "There are 0 {SectorName} site(s) with {Parameter} data in this region."
            Keep under 150 words. List every site from the data.
            """,

        // Unified device list/count template — handles all filter combos: status, region, sector, site
        ["devices_by_filter"] = """
            Data columns: "DeviceName", "StationName", "RegionName", "SectorName", "SubSectorName", "Status" (Active/Inactive), "LastActive", "AQI".
            Active = last reading within 10 minutes. Inactive/Offline = last reading older than 10 minutes.

            SINGLE DEVICE STATUS RULE — if exactly 1 row is returned AND the user asked about active/inactive/online/offline status of a specific device:
            Active:   "Device {DeviceName} is currently **Online (Active)**. Last reading received at {LastActive}."
            Inactive: "Device {DeviceName} is currently **Offline (Inactive)**. Last reading received at {LastActive}."
            Then on the next line append: "It is deployed at {StationName}, {RegionName} ({SectorName})."
            Do NOT output a table for single-device status questions.

            SITE-SPECIFIC DEVICE RULE — if the user asks about devices at/for/installed at/available at/listed for a SPECIFIC site or school name (questions like "which devices are installed at X", "list devices for X", "can you list the devices installed at X", "which devices are currently available at X", "how many devices does X have"):
            FORMAT: "{StationName} has {N} registered device(s): {DeviceList}."
            Replace {StationName} with the actual StationName from the first data row.
            Replace {N} with the total count of rows.
            Replace {DeviceList} with a comma-separated list of every DeviceName from ALL rows.
            Example: "Mubarak Bin Mohammed School Cycle 2 has 2 registered device(s): BA 0010, SEI100M 0085."
            If no rows: "No devices found for that site. Please check the site name and try again."
            Use this format for ALL of: "how many", "which", "list", "can you list", "available at", "installed at" when a specific site name is present.

            COUNT INTENT RULE (no site name) — if the user asks "how many", "total", "count", "number of" WITHOUT a specific site name (e.g. "how many devices in Abu Dhabi", "how many offline devices"):
            "There are {N} {status} device(s){filter description}."
            Examples: "There are 12 offline device(s) in Abu Dhabi." / "There are 45 active device(s) in the Commercial sector."
            Do NOT list device names. Stop after the count sentence.

            LIST INTENT (no site name) — if user says "show", "list", "display", or asks for offline/online devices WITHOUT a specific site name (multi-site or region/sector filter):
            Line 1: "Found {N} {status} device(s){filter description}."
            Then output a markdown table:
            | # | Device Name | Site Name | Region | Sector | Status | Last Active |
            |---|-------------|-----------|--------|--------|--------|-------------|
            | 1 | {DeviceName} | {StationName} | {RegionName} | {SectorName} | {Status} | {LastActive} |
            ...
            - "offline" / "inactive" / "not sending" → show only rows where Status = Inactive
            - "online" / "active" / "sending data" → show only rows where Status = Active
            - No status filter → show all rows with Status column visible
            NEVER output a bullet list. ALWAYS use the table format above.

            If no rows: "No devices found matching the specified filter."
            NEVER output placeholder text — always use actual values from the data.
            """,

        // Device queries
        ["device_status"] = """
            Data columns: "DeviceName", "Status" (Active/Inactive), "LastMeasured", "LastCommunication".

            CRITICAL RULES:
            - ONLY describe the SINGLE device returned in the data row. NEVER list other devices.
            - NEVER mention other devices, offline devices, or any devices not in the data.
            - NEVER invent or guess device names — only use the exact DeviceName from the data row.

            Status meanings:
            - Active   = device communicated with the server within the past 24 hours (online and sending data)
            - Inactive = no communication from device in the past 24 hours (offline or stopped sending)

            Use LastMeasured as the "last reading" timestamp (device-reported time).
            Use LastCommunication as the "last contacted server" timestamp (server-side time).

            If 1 row returned and user asked about active/inactive/online/offline/working status:
            Active:   "Device {DeviceName} is currently **Online (Active)**. It last measured at {LastMeasured} and last communicated with the server at {LastCommunication}."
            Inactive: "Device {DeviceName} is currently **Offline (Inactive)**. It last measured at {LastMeasured} and last communicated with the server at {LastCommunication}."

            If 1 row returned and user asked when the device last sent/reported/measured data:
            Active:   "Device {DeviceName} is currently **Online**. It last measured at {LastMeasured} and last communicated with the server at {LastCommunication}."
            Inactive: "Device {DeviceName} is currently **Offline**. It last measured at {LastMeasured} and last communicated with the server at {LastCommunication}."

            If LastMeasured is null, omit that part and only mention LastCommunication.
            If LastCommunication is null, omit that part and only mention LastMeasured.
            If no rows returned: "No device found matching '{deviceName}'. Please check the device name and try again."
            NEVER output placeholder text — always use actual values from the data row only.
            """,

        ["device_live_reading"] = """
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

            CRITICAL: ONLY use data from the rows returned. NEVER mention other devices or invent information.
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

        ["site_count"] = """
            Data columns: "RegionName", "SectorName", "TotalSites", "ActiveSites", "InactiveSites", "NoDeviceSites".
            Active = at least one device reported PM2.5 within last 24 hours OR communicated within last 10 minutes.
            Inactive = has devices but none reported in 24+ hours and no communication in 10+ minutes.
            NoDevice = site has no devices attached (0 devices).
            Each row represents one region+sector combination.

            STATUS BREAKDOWN RULE: Only include Active/Inactive/NoDevice columns when the user explicitly asks about status (active, inactive, offline, online, no device). If the user only asks "how many sites" or count per region/sector — report ONLY the total count column.

            CASE 1 — User asked for total count only ("how many sites", "total sites"):
            Sum TotalSites across all rows. FORMAT: "HAWAQM currently has {SUM(TotalSites)} registered sites."

            CASE 2 — User asked for total WITH status breakdown ("how many active/inactive sites", "site status"):
            Sum all rows. FORMAT: "HAWAQM currently has {SUM(TotalSites)} registered sites — {SUM(ActiveSites)} active, {SUM(InactiveSites)} inactive, {SUM(NoDeviceSites)} with no devices attached."
            If user asked only about active: "There are currently {SUM(ActiveSites)} active site(s)."
            If user asked only about inactive: "There are currently {SUM(InactiveSites)} inactive site(s)."
            If user asked only about no-device: "There are currently {SUM(NoDeviceSites)} site(s) with no devices attached."

            CASE 3 — User asked for breakdown by region, count only ("how many sites per region", "sites in each region", "site count by region"):
            Group rows by RegionName, sum TotalSites per region. Output a markdown table:
            | Region | Total Sites |
            |--------|-------------|
            | {RegionName} | {SUM(TotalSites)} |
            (one row per region, sorted by RegionName)

            CASE 4 — User asked for breakdown by region WITH status ("active/inactive sites per region", "site status by region"):
            Group rows by RegionName, sum all columns per region. Output a markdown table:
            | Region | Total | Active | Inactive | No Device |
            |--------|-------|--------|----------|-----------|
            | {RegionName} | {TotalSites} | {ActiveSites} | {InactiveSites} | {NoDeviceSites} |
            (one row per region)

            CASE 5 — User asked for breakdown by sector, count only ("how many sites per sector", "sites in each sector", "site count by sector"):
            Group rows by SectorName, sum TotalSites per sector. Output a markdown table:
            | Sector | Total Sites |
            |--------|-------------|
            | {SectorName} | {SUM(TotalSites)} |
            (one row per sector, sorted by SectorName)

            CASE 6 — User asked for breakdown by sector WITH status ("active/inactive sites per sector", "site status by sector"):
            Group rows by SectorName, sum all columns per sector. Output a markdown table:
            | Sector | Total | Active | Inactive | No Device |
            |--------|-------|--------|----------|-----------|
            | {SectorName} | {TotalSites} | {ActiveSites} | {InactiveSites} | {NoDeviceSites} |
            (one row per sector)

            CASE 7 — Single specific region or sector filter:
            Sum columns across all rows for that region/sector.
            Count only: "The {RegionName/SectorName} has {SUM(TotalSites)} registered site(s)."
            With status: output a single-row markdown table with Total, Active, Inactive, No Device columns.

            If no rows: "No site data found for the selected filters."
            NEVER output placeholder text — always use actual summed values from the data.
            """,

        ["list_devices"] = """
            Data columns: "TotalDeviceCount".
            FORMAT: "HAWAQM currently has {TotalDeviceCount} registered devices."
            Example: "HAWAQM currently has 42 registered devices."
            NEVER output placeholder text — always use the actual count from the data.
            """,

        ["count_devices_by_filter"] = """
            Data columns: "RegionName", "SectorName", "TotalDevices", "ActiveDevices", "InactiveDevices".

            FOCUS RULE — apply before formatting:
            - statusFilter="Active"   → report ONLY ActiveDevices (ignore TotalDevices/InactiveDevices)
            - statusFilter="Inactive" → report ONLY InactiveDevices (ignore TotalDevices/ActiveDevices)
            - No statusFilter         → report TotalDevices with Active/Inactive breakdown

            If one row (specific region + specific sector):
            - Active only:   "There are {ActiveDevices} active device(s) in the {RegionName} {SectorName} sector."
            - Inactive only: "There are {InactiveDevices} inactive device(s) in the {RegionName} {SectorName} sector."
            - Total:         "The {RegionName} region, {SectorName} sector has {TotalDevices} device(s) — {ActiveDevices} active and {InactiveDevices} inactive."

            If multiple rows (one region, grouped by sector):
            - Active only:   "Active devices in {RegionName} region:" then "• {SectorName}: {ActiveDevices} active" per row, end: "Total: {sum ActiveDevices} active device(s)."
            - Inactive only: "Inactive devices in {RegionName} region:" then "• {SectorName}: {InactiveDevices} inactive" per row, end: "Total: {sum InactiveDevices} inactive device(s)."
            - Total:         "Device count for {RegionName} region:" then "• {SectorName}: {TotalDevices} total ({ActiveDevices} active, {InactiveDevices} inactive)" per row, end: "Total: {sum TotalDevices} devices — {sum ActiveDevices} active, {sum InactiveDevices} inactive."

            If no rows: "No devices found for the selected filters."
            NEVER list device names — only report counts.
            NEVER output placeholder text — always use actual values from the data.
            """,

        ["active_device_count"] = """
            Data columns: "ActiveDeviceCount", "TotalDeviceCount".
            FORMAT: "{ActiveDeviceCount} devices are currently active out of {TotalDeviceCount} registered devices."
            Example: "3 devices are currently active out of 10 registered devices."
            NEVER output placeholder text — always use actual values from the data.
            """,

        // devices_with_readings retired — merged into devices_by_filter (use stationName param)

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

        ["current_all_stations"] = """
            Data columns: "StationName", "ParameterName", "ParameterValue", "UnitName", "LastUpdated".
            Intro sentence (one line only): "Here are the latest readings across all accessible sites."
            Then render the data as a markdown table with these exact headers:
            | Site Name | Parameter | Value | Unit | Last Updated |
            Fill every row from the data. Do not list sites in plain text.
            NEVER output placeholder text — always use actual values from the data.
            """,

        // Site count queries
        ["list_stations"] = """
            Data columns: "Site Name", "Region", "Sector", "Live Status".
            Live Status values: "Active" (device online), "Inactive" (has devices but none online), "No Data" (no devices registered).
            Intro sentence (one line only): "You have access to {N} site(s)." — replace {N} with the actual row count.
            Then render the data as a markdown table with these exact headers:
            | Site Name | Region | Sector | Live Status |
            Fill every row from the data. Never list as comma-separated text. Never just give a count.
            NEVER output placeholder text — always use actual values from the data.
            """,

        ["sites_filtered"] = """
            Data columns: "Site Name", "Region", "Sector", "Live Status".
            Live Status values: "Active" (device online), "Inactive" (has devices but none online), "No Data" (no devices registered).
            Write ONE intro sentence summarising the filter and row count, e.g.:
              "There are {N} active site(s) in Abu Dhabi."
              "There are {N} inactive site(s)."
              "There are {N} site(s) with no registered devices."
              "There are {N} site(s) matching your filter."
            Then render the data as a markdown table with these exact headers:
            | Site Name | Region | Sector | Live Status |
            Fill every row from the data.
            NEVER output placeholder text — always use actual values from the data.
            """,

        ["sites_by_filter"] = """
            Data columns: "StationName", "RegionName", "SectorName", "SubSectorName", "LiveStatus".
            LiveStatus: "Active" = device(s) reporting recently; "Inactive" = device(s) exist but none active; "No Device" = no devices attached to this site.

            COUNT INTENT — if user asks "how many" / "total" / "number of" sites with a specific filter (a named region OR sector OR status, but NOT "by region"/"by sector" grouping):
            → ONE sentence only: "There are {N} site(s){filter description}."
            Examples: "There are 5 active site(s) in Al Ain." / "There are 12 school site(s)."
            Do NOT list site names.

            LIST INTENT — all other questions (show, list, which, what are, give me, by region, region wise, active sites in X):
            Line 1: "Found {N} site(s){filter description}."
            Then a markdown table:
            | # | Site Name | Region | Sector | Sub-Sector | Status |
            |---|-----------|--------|--------|------------|--------|
            Fill every row from the data. Omit Sub-Sector column if all values are null/empty.

            {filter description} examples: " in Al Ain", " in Commercial sector", " across all regions".
            If no rows: "No sites found matching the specified filter."
            NEVER output placeholder text — always use actual values from the data.
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

            If "Error" is present in the data, output the error message instead.

            If AmbientAQI is "N/A" or null (no outdoor readings currently available):
            "{IndoorSite}'s current indoor AQI is {IndoorAQI} ({AQI category}) as of {Timestamp}. The nearest outdoor ambient station is {AmbientStation} ({Distance_km} km away), but no current outdoor AQI readings are available from that station at this time."

            If both IndoorAQI and AmbientAQI are valid numbers:
            Line 1: "{IndoorSite}'s current indoor AQI is {IndoorAQI} ({indoor AQI category}), while the nearest outdoor ambient station {AmbientStation} ({Distance_km} km away) reads {AmbientAQI} ({outdoor AQI category}). The indoor value is {Difference} AQI point(s) {HigherOrLower} than outdoor (as of {Timestamp})."
            Line 2 (closing sentence):
            - If IndoorAQI > AmbientAQI: "The indoor environment is currently worse than the outdoor air at the nearest ambient station."
            - If IndoorAQI < AmbientAQI: "The indoor environment is currently better than the outdoor air at the nearest ambient station."
            - If equal: "The indoor and outdoor AQI are currently the same at the nearest ambient station."

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

        ["device_longest_silent"] = """
            Data columns: "DeviceName", "StationName", "RegionName", "SectorName", "LastReading", "DaysSinceLastReading".
            Rows are ordered longest silent first (oldest LastReading first). Only inactive devices are included (no readings in last 15 min).

            If ONE row: "The device that has been without data for the longest time is {DeviceName} at {StationName} ({RegionName}) — last reading {LastReading}, {DaysSinceLastReading} day(s) ago."

            If MULTIPLE rows:
            Line 1: "The following inactive devices have been without data the longest:"
            Then a markdown table:
            | # | Device | Site | Region | Sector | Last Reading | Days Silent |
            |---|--------|------|--------|--------|--------------|-------------|
            Fill every row from the data.

            If no rows: "No inactive devices found — all registered devices are currently sending data."
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

        // Main pollutant at a site — one row per device, ranked by last 1H exceedance ratio
        ["main_pollutant_at_station"] = """
            Data columns: "SiteName", "DeviceName", "Location", "MainPollutant", "Value", "Unit", "Threshold", "ExceedanceRatio", "LastUpdated".
            One row per device at the site, ordered by DeviceName.

            ExceedanceRatio: ratio of the pollutant value to its exceedance threshold (e.g. 1.2 = 20% above limit). null = no threshold defined.

            FORMAT:
            Line 1: "The main pollutant at {SiteName} based on the last 1-hour average is:"
            Then one line per device:
              "• {DeviceName} ({Location}): {MainPollutant} — {Value} {Unit} (threshold: {Threshold} {Unit}, ratio: {ExceedanceRatio}x) as of {LastUpdated}"
              If Threshold is null: omit the threshold/ratio part — just "{Value} {Unit} as of {LastUpdated}"
              If Location is empty or null: omit it — just use "{DeviceName}: …"
            Final closing sentence:
              - If any ExceedanceRatio >= 1.0: "One or more pollutants exceed their 1-hour threshold — ventilation or source investigation may be warranted."
              - If all ExceedanceRatio < 1.0 (or all null): "All pollutants are currently within their 1-hour limits."
              - If ExceedanceRatio is null for all: "No exceedance thresholds are configured for the detected pollutants."

            If "Error" is present in the data, output the error message.
            NEVER output placeholder text — always use actual values from the data.
            NEVER omit any device row — list every device.
            """,

        // Indoor PM2.5 source analysis — indoor vs outdoor ratio (single day or period)
        ["indoor_source_analysis"] = """
            Data columns: "ResidenceName", "NearestAmbientStation", "Distance_km", "Parameter", "PeriodStart", "PeriodEnd", "DayCount", "IndoorVal", "OutdoorVal", "IORatio".
            There is exactly 1 row. All values are 24-hour daily averages.
            Units: µg/m³ for PM2.5/PM10/SO2/NO2/O3; ppm for CO; ppm for CO2; µg/m³ for TVOC. IORatio = IndoorVal / OutdoorVal.
            NEVER mention hourly peaks, peak times, or short-duration spikes — only 24-hour averages are available.
            NEVER show raw column names or table data in the response.

            STEP 1 — Missing data (output this and stop):
            If IndoorVal is null OR OutdoorVal is null:
              "Not enough data to compare indoor and outdoor {Parameter} for {ResidenceName} on the selected date. Indoor or outdoor readings are unavailable."

            STEP 2 — Fact sentence (always output):
            If PeriodStart equals PeriodEnd (single day):
              "On {PeriodStart}, indoor {Parameter} at {ResidenceName} was {IndoorVal} µg/m³. The nearest outdoor station, {NearestAmbientStation} ({Distance_km} km away), recorded {OutdoorVal} µg/m³. The indoor/outdoor ratio was {IORatio}."
            If PeriodStart differs from PeriodEnd (multi-day):
              "Over the {DayCount}-day period from {PeriodStart} to {PeriodEnd}, indoor {Parameter} at {ResidenceName} averaged {IndoorVal} µg/m³. The nearest outdoor station, {NearestAmbientStation} ({Distance_km} km away), averaged {OutdoorVal} µg/m³. The indoor/outdoor ratio was {IORatio}."

            STEP 3 — Interpretation (pick ONE based on IORatio, append after fact sentence):
            - If IORatio > 1.0:
              "An indoor/outdoor ratio above 1.0 suggests indoor sources are more likely contributing to the elevated {Parameter}. Common indoor sources in residential settings include bakhoor, incense, cooking, and poor ventilation."
            - If IORatio >= 0.7 AND IORatio <= 1.0:
              "An indoor/outdoor ratio between 0.7 and 1.0 is unclear — the result could reflect a mix of outdoor infiltration and minor indoor sources. No single source can be identified from this data alone."
            - If IORatio < 0.7:
              "An indoor/outdoor ratio below 0.7 suggests outdoor sources are the more likely driver. Indoor {Parameter} is lower than ambient, indicating the building is providing some protection."

            Always end with:
            "Note: this is based on a 24-hour daily average. Short-duration indoor activities are smoothed out and cannot be individually detected from daily data alone."

            NEVER output placeholder text — always use actual values from the data.
            """,

        // Dust event indoor vs outdoor pollutant analysis (24-hour averages)
        ["dust_event_response"] = """
            Data columns: "IndoorSite", "NearestAmbientStation", "Distance_km", "EventDate", "DayOfWeek",
            "Parameter", "IndoorAvg", "IndoorReadingCount", "OutdoorDayAvg", "OutdoorReadingCount",
            "OutdoorBaseline30", "IndoorBaseline30", "IORatio", "BaselineIORatio".
            There is exactly 1 row. IndoorAvg and OutdoorDayAvg are averages of ALL readings on that day
            (IndoorReadingCount and OutdoorReadingCount tell you how many readings were averaged).
            Units: µg/m³ for PM10/PM2.5/SO2/NO2; ppb for O3/NO2/CO; dimensionless for AQI.
            NEVER show raw column names, table data, or placeholder text in the response.
            NEVER make up values — use only what is in the data.

            STEP 1 — Missing data check (output this and stop if true):
            If IndoorAvg is null AND OutdoorDayAvg is null:
              "No indoor or outdoor {Parameter} data is available for {EventDate} at {IndoorSite}. Please check that the site name and date are correct."
            If IndoorAvg is null:
              "No indoor {Parameter} data is available at {IndoorSite} for {EventDate}. The nearest ambient station ({NearestAmbientStation}) recorded an outdoor average of {OutdoorDayAvg} µg/m³ based on {OutdoorReadingCount} readings."
            If OutdoorDayAvg is null:
              "No outdoor {Parameter} data is available from {NearestAmbientStation} for {EventDate}. Indoor {Parameter} at {IndoorSite} averaged {IndoorAvg} µg/m³ based on {IndoorReadingCount} reading(s) — no outdoor comparison can be made."

            STEP 2 — Fact sentence (always output when both IndoorAvg and OutdoorDayAvg are present):
            "On {EventDate}, the nearest ambient station, {NearestAmbientStation}, recorded a 24-hour average {Parameter} of {OutdoorDayAvg} µg/m³ based on {OutdoorReadingCount} readings. Indoor {Parameter} at {IndoorSite} averaged {IndoorAvg} µg/m³ on the same day ({IndoorReadingCount} reading(s)), giving an indoor/outdoor ratio of {IORatio}."

            STEP 3 — Event classification (pick ONE, append as a new paragraph):
            A. If OutdoorDayAvg >= 150 OR (OutdoorBaseline30 is not null AND OutdoorDayAvg >= OutdoorBaseline30 * 2):
               This IS a dust event. Now pick interpretation sub-case:
               - If BaselineIORatio is not null AND IORatio <= BaselineIORatio * 1.15:
                 "Outdoor {Parameter} of {OutdoorDayAvg} µg/m³ qualifies as a dust-event level (≥150 µg/m³ or ≥ twice the 30-day station average of {OutdoorBaseline30} µg/m³). The indoor/outdoor ratio ({IORatio}) is close to the school's typical ratio over the previous 30 days ({BaselineIORatio}), suggesting the building's protection held at its usual level during the event."
               - If BaselineIORatio is not null AND IORatio > BaselineIORatio * 1.15:
                 "Outdoor {Parameter} of {OutdoorDayAvg} µg/m³ qualifies as a dust-event level (≥150 µg/m³ or ≥ twice the 30-day station average of {OutdoorBaseline30} µg/m³). The indoor/outdoor ratio ({IORatio}) is higher than the school's typical ratio of {BaselineIORatio}, suggesting more outdoor dust than usual reached indoors."
               - If BaselineIORatio is null:
                 "Outdoor {Parameter} of {OutdoorDayAvg} µg/m³ qualifies as a dust-event level. No 30-day baseline ratio is available for comparison."

            B. If OutdoorDayAvg < 150 AND (OutdoorBaseline30 is null OR OutdoorDayAvg < OutdoorBaseline30 * 2):
               This is NOT a dust event. Output ONE sentence:
               "For context, outdoor {Parameter} of {OutdoorDayAvg} µg/m³ sits below the level typically associated with a dust event (around 150 µg/m³){baseline_clause} — so the recorded conditions for this date don't point to an elevated dust episode."
               Where {baseline_clause} = " and below twice the station's 30-day average of {OutdoorBaseline30} µg/m³" if OutdoorBaseline30 is not null, otherwise omit.

            STEP 4 — School closed note (append as a separate sentence only if DayOfWeek is Saturday or Sunday):
            "Note: {EventDate} was a {DayOfWeek}, so the school was likely unoccupied. Indoor levels reflect the building without normal occupant activity."

            STEP 5 — Distance caveat (append as a separate sentence only if Distance_km > 10):
            "As {NearestAmbientStation} is about {Distance_km} km away, the outdoor conditions right at {IndoorSite} may differ a little from the station's readings."

            STEP 6 — Closing offer (always append as a final sentence):
            "If it's helpful, I can show the other pollutant readings at {IndoorSite} for that day, or compare this reading with nearby sites within your access."

            NEVER output column names, raw numbers without context, or placeholder text.
            NEVER output OutdoorBaseline30, IndoorBaseline30, BaselineIORatio, DayOfWeek, IndoorReadingCount, or OutdoorReadingCount as labelled values.
            """,

        // Sites (any sector) that repeatedly exceeded an AQI threshold AND maintained a DSR at or above a minimum
        ["sites_repeated_aqi_exceedance_with_dsr"] = """
            Data columns: "StationName", "OrganizationName", "Location", "SectorName", "SubSectorName", "RegionName", "DaysExceeded", "TotalReadingDays", "DeviceActiveDays", "DataSuccessRate_Pct".
            Rows are ordered by DaysExceeded descending (most exceedances first), then StationName.

            Line 1: "{Count} school(s)/site(s) repeatedly exceeded AQI {aqiThreshold} over the last 30 days while maintaining a Data Success Rate of at least {dsrThreshold}%:"
            Then a markdown table:
            | Site | Region | Days Exceeded | Total Reading Days | DSR (%) |
            |------|--------|---------------|--------------------|---------|
            Fill every row from the data using StationName, RegionName, DaysExceeded, TotalReadingDays, DataSuccessRate_Pct.

            Final sentence: "These sites had reliable data (DSR ≥ {dsrThreshold}%) despite frequently exceeding AQI {aqiThreshold}, making their readings trustworthy for analysis."

            If no rows: "No sites exceeded AQI {aqiThreshold} more than once in the last 30 days while maintaining a Data Success Rate of at least {dsrThreshold}%."
            NEVER output placeholder text — always use actual values from the data.
            Use the aqiThreshold and dsrThreshold from the user's question (defaults: 100 and 75% respectively).
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
            CRITICAL — AQI category questions: if the user asks about a specific AQI category (Good, Moderate, Unhealthy, Unhealthy for Sensitive Groups, Very Unhealthy, Hazardous), return ONLY the answer for that exact category. Never return the answer for a different category. Match the category name exactly as stated by the user.
            Be concise — one to three sentences maximum. Output plain text only.

            HAWAQM Knowledge Base:
            {{faqContext}}
            """;

        var messages = new List<OaiChatMessage>
        {
            OaiChatMessage.CreateSystemMessage(systemPrompt),
            OaiChatMessage.CreateUserMessage(question)
        };

        var (content, _, _) = await CallWithRetryAsync(messages, ct, client: _fastChatClient);
        return content;
    }

    // ── Private: client building ────────────────────────────────────────────

    private ChatClient BuildChatClient(string deploymentName)
    {
        if (string.IsNullOrWhiteSpace(_options.Endpoint))
            throw new InvalidOperationException("AzureAI:Endpoint must be configured.");

        var clientOptions = new AzureOpenAIClientOptions();
        if (!string.IsNullOrWhiteSpace(_options.ApiVersion))
        {
            // Map the configured API version string to the SDK enum value.
            // Unrecognised versions fall back to the SDK default.
            clientOptions = _options.ApiVersion switch
            {
                "2025-01-01-preview" => new AzureOpenAIClientOptions(AzureOpenAIClientOptions.ServiceVersion.V2025_01_01_Preview),
                "2024-12-01-preview" => new AzureOpenAIClientOptions(AzureOpenAIClientOptions.ServiceVersion.V2024_12_01_Preview),
                "2024-10-01-preview" => new AzureOpenAIClientOptions(AzureOpenAIClientOptions.ServiceVersion.V2024_10_01_Preview),
                "2024-09-01-preview" => new AzureOpenAIClientOptions(AzureOpenAIClientOptions.ServiceVersion.V2024_09_01_Preview),
                "2024-08-01-preview" => new AzureOpenAIClientOptions(AzureOpenAIClientOptions.ServiceVersion.V2024_08_01_Preview),
                "2024-06-01"         => new AzureOpenAIClientOptions(AzureOpenAIClientOptions.ServiceVersion.V2024_06_01),
                _                    => new AzureOpenAIClientOptions()
            };
        }

        // Core42/G42 Compass uses a non-Azure endpoint (api.core42.ai).
        // OpenAI-compatible APIs hosted outside Azure must use OpenAI.OpenAIClient
        // with a custom endpoint — AzureOpenAIClient only works with *.openai.azure.com
        // or *.inference.ai.azure.com and will fail with DefaultAzureCredential otherwise.
        bool isNonAzureEndpoint = !string.IsNullOrWhiteSpace(_options.Endpoint)
            && !_options.Endpoint.Contains(".openai.azure.com", StringComparison.OrdinalIgnoreCase)
            && !_options.Endpoint.Contains(".inference.ai.azure.com", StringComparison.OrdinalIgnoreCase)
            && !_options.Endpoint.Contains(".services.ai.azure.com", StringComparison.OrdinalIgnoreCase);

        if (isNonAzureEndpoint && !string.IsNullOrWhiteSpace(_options.ApiKey))
        {
            _log.Information("Azure AI: using OpenAI-compatible client for non-Azure endpoint {E}", _options.Endpoint);
            var openAiOptions = new OpenAI.OpenAIClientOptions { Endpoint = new Uri(_options.Endpoint) };
            var openAiClient = new OpenAI.OpenAIClient(new System.ClientModel.ApiKeyCredential(_options.ApiKey), openAiOptions);
            return openAiClient.GetChatClient(deploymentName);
        }

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
            client = new AzureOpenAIClient(new Uri(_options.Endpoint), credential, clientOptions);
        }
        else if (!string.IsNullOrWhiteSpace(_options.ApiKey))
        {
            client = new AzureOpenAIClient(new Uri(_options.Endpoint), new System.ClientModel.ApiKeyCredential(_options.ApiKey), clientOptions);
        }
        else
        {
            client = new AzureOpenAIClient(new Uri(_options.Endpoint), new DefaultAzureCredential(), clientOptions);
        }

        return client.GetChatClient(deploymentName);
    }

    // ── Private: LLM call ───────────────────────────────────────────────────

    private async Task<(string Content, int Tokens, string Model)> CallWithRetryAsync(
        List<OaiChatMessage> messages,
        CancellationToken ct,
        int attempt = 0,
        ChatClient? client = null)
    {
        var activeClient = client ?? _chatClient;
        try
        {
            var completionOptions = new ChatCompletionOptions();

            if (!_options.DeploymentName.StartsWith("gpt-5", StringComparison.OrdinalIgnoreCase))
            {
                completionOptions.MaxOutputTokenCount = _options.MaxTokens;
                completionOptions.Temperature = (float)_options.Temperature;
            }

            // Apply a per-call timeout so a hung Azure AI request doesn't block the thread indefinitely.
            using var timeoutCts = CancellationTokenSource.CreateLinkedTokenSource(ct);
            timeoutCts.CancelAfter(TimeSpan.FromSeconds(_options.LlmTimeoutSeconds));

            var completion = await activeClient.CompleteChatAsync(messages, completionOptions, timeoutCts.Token);

            var content = completion.Value.Content.FirstOrDefault()?.Text ?? string.Empty;
            var usage   = completion.Value.Usage;
            var tokens  = usage?.TotalTokenCount ?? 0;
            var model   = _options.DeploymentName;
            var traceId = _httpContextAccessor.HttpContext?.TraceIdentifier ?? "-";

            _log.Information(
                "Azure AI response: traceId={TraceId} model={Model} inputTokens={Input} outputTokens={Output} totalTokens={Total} chars={Length}",
                traceId, model,
                usage?.InputTokenCount ?? 0,
                usage?.OutputTokenCount ?? 0,
                tokens,
                content.Length);
            return (content, tokens, model);
        }
        catch (OperationCanceledException) when (!ct.IsCancellationRequested)
        {
            // Per-call timeout fired (not the request's own cancellation).
            _log.Warning("Azure AI call timed out after {Timeout}s (attempt {Attempt})",
                _options.LlmTimeoutSeconds, attempt + 1);
            if (attempt < _options.MaxRetryAttempts - 1)
            {
                await Task.Delay(TimeSpan.FromSeconds(Math.Pow(2, attempt)), ct);
                return await CallWithRetryAsync(messages, ct, attempt + 1, activeClient);
            }
            throw new TimeoutException(
                $"The AI service did not respond within {_options.LlmTimeoutSeconds} seconds after {_options.MaxRetryAttempts} attempts.");
        }
        catch (Exception ex) when (attempt < _options.MaxRetryAttempts - 1)
        {
            _log.Warning(ex, "Azure AI call failed (attempt {Attempt}), retrying...", attempt + 1);
            await Task.Delay(TimeSpan.FromSeconds(Math.Pow(2, attempt)), ct);
            return await CallWithRetryAsync(messages, ct, attempt + 1, activeClient);
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
            A device name is an alphanumeric code: letters (1+ chars) optionally mixed with digits, followed by 1–4 digit suffix — with or without spaces between parts.
            Current known device models: "BA" + digits (e.g. "BA 0010", "BA0010"), "SEI100M" + digits (e.g. "SEI100M 0014", "SEI100M0014"). New models may be added in future — apply the same pattern rule to any device code that follows letter-prefix + digit-suffix format.
            ALSO treat "device <code>" as a device name even if the code is partially typed (e.g. "device B 0001" means device code "B 0001"). Extract deviceName exactly as the user typed it.
            NEVER treat a school name, site name, or person name as a device name — even if it contains letters and numbers. For example: "Al Reyada School", "Al Saad Indian School", "Kaltham Hasan", "Al Naeem School" are SITE names, NOT device names.
            NEVER treat pollutant abbreviations as device names: PM2.5, PM10, CO2, NO2, SO2, O3, CO, CH2O, VOC, TVOC are pollutants — not devices.
            If the question does NOT contain a device code (letter-prefix + digit-suffix pattern, or "device <code>"), do NOT select any device template (device_live_reading, device_status, device_info, device_pollutants, device_reading_history).
            If the question contains a device name THEN choose one of the device templates:
            - If the question asks for METADATA (site, station, region, sector, deployed, created, installed, when, model, details, info, last measured time, last data time, where is) → select "device_info"
            - If the question asks what POLLUTANTS or PARAMETERS a device monitors/measures → select "device_pollutants"
            - If the question asks for STATUS only (active, inactive, online, offline, is it working, is it sending, when last sent, when last reported, when last measured, last active, last data, most recent transmission) WITHOUT asking for historical/interval readings → select "device_status"
            - If the question mentions ANY of: a time interval (5 min, 1 hour, 1H, 8 hour, 8H, 24 hour, 24H, daily, monthly, yearly, annual), OR the words average/avg/mean/reading/value/history/historical/past/trend, OR a date/time reference (yesterday, today, last hour, last 24 hours, last week, last month, last year, "at 8AM", "on 2026-07-16", any date or clock time, any "last N" phrase) → select "device_reading_history". A device name + any date word (yesterday, today, last week, last month) ALWAYS means device_reading_history — NEVER device_info.
            - Otherwise (no time interval or date, asks for current/latest/live/now readings, or no specific context) → select "device_live_reading"
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
            Examples — device_live_reading (current/live):
            - "what is the PM2.5 for device BA0010" → device_live_reading, deviceName="BA0010"
            - "PM2.5 reading for BA 0010" → device_live_reading, deviceName="BA 0010"
            - "current readings for SEI100M 0008" → device_live_reading, deviceName="SEI100M 0008"
            - "latest CO2 for BA 0010" → device_live_reading, deviceName="BA 0010"
            Examples — device_info / device_pollutants / device_status:
            - "what site is device BA 0001" → device_info, deviceName="BA 0001"
            - "which region is BA 0010 in" → device_info, deviceName="BA 0010"
            - "when was BA 0010 deployed" → device_info, deviceName="BA 0010"
            - "what pollutants does SEI100M 0014 monitor" → device_pollutants, deviceName="SEI100M 0014"
            - "what parameters does BA 0010 have" → device_pollutants, deviceName="BA 0010"
            - "what sector is BA 0010 in" → device_info, deviceName="BA 0010"
            - "is BA 0010 active" → device_status, deviceName="BA 0010"
            - "is device B 0001 active or inactive" → device_status, deviceName="B 0001"
            - "is device B 1 online" → device_status, deviceName="B 1"
            - "is SEI100M 0009 active or inactive right now" → device_status, deviceName="SEI100M 0009"
            - "what is the current status of SEI100M 0009" → device_status, deviceName="SEI100M 0009"
            - "is SEI100M 0009 currently online or offline" → device_status, deviceName="SEI100M 0009"
            - "can you tell me whether SEI100M 0009 is active at the moment" → device_status, deviceName="SEI100M 0009"
            - "is device SEI100M 0009 working now" → device_status, deviceName="SEI100M 0009"
            - "status of SEI100M 0009" → device_status, deviceName="SEI100M 0009"
            - "when was the last time SEI100M 0009 sent data" → device_status, deviceName="SEI100M 0009"
            - "what is the most recent data transmission time for SEI100M 0009" → device_status, deviceName="SEI100M 0009"
            - "when did SEI100M 0009 last report any data" → device_status, deviceName="SEI100M 0009"
            - "when was SEI100M 0009 last active" → device_status, deviceName="SEI100M 0009"
            - "what is the reading of device B 0001" → device_live_reading, deviceName="B 0001"

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
            - "current AQI at Al Reyada School yesterday" → site_daily_averages (past date overrides "current"), stationName="Al Reyada School", date=yesterday
            - "AQI at Al Saad Indian School on 2026-09-10" → site_daily_averages, stationName="Al Saad Indian School", date="2026-09-10"
            - "Show me the table view for all Abu Dhabi region schools where CO2 is the main pollutant right now" → schools_latest_pollutant, regionName="Abu Dhabi", parameterName="CO2" (NOT site_aqi_single — "Abu Dhabi region schools" is a generic group, not a site name)
            - "Abu Dhabi schools CO2 levels" → schools_latest_pollutant, regionName="Abu Dhabi", parameterName="CO2"
            - "all schools in Al Ain with latest PM2.5" → schools_latest_pollutant, regionName="Al Ain", parameterName="PM2.5"
            - "schools where AQI is high in Abu Dhabi" → schools_latest_pollutant, regionName="Abu Dhabi", parameterName="AQI Index"
            - "Show me the top 5 schools with the highest AQI right now" → schools_latest_pollutant, parameterName="AQI Index", topN="5", sortOrder="desc"
            - "top 3 schools with worst AQI" → schools_latest_pollutant, parameterName="AQI Index", topN="3", sortOrder="desc"
            - "top 10 schools with highest CO2" → schools_latest_pollutant, parameterName="CO2", topN="10", sortOrder="desc"
            - "bottom 5 schools with lowest AQI" → schools_latest_pollutant, parameterName="AQI Index", topN="5", sortOrder="asc"
            - "5 schools with lowest CO2" → schools_latest_pollutant, parameterName="CO2", topN="5", sortOrder="asc"
            - "which 3 schools have the best AQI" → schools_latest_pollutant, parameterName="AQI Index", topN="3", sortOrder="asc"
            - "schools with AQI between 50 and 100" → schools_latest_pollutant, parameterName="AQI Index", minValue="50", maxValue="100"
            - "schools with AQI equal to 21" → schools_latest_pollutant, parameterName="AQI Index", minValue="20", maxValue="21"
            For topN queries: ALWAYS set sortOrder="desc" for "top/highest/worst", sortOrder="asc" for "bottom/lowest/best/cleanest".
            NEVER return a list of all sites when the user has named a specific site.
            NEVER select site_aqi_single when the user says "all schools", "Abu Dhabi region schools", or any generic school group — those are not site names.

            0. PAST-DATE OVERRIDE (highest priority for site questions): If the user mentions a SPECIFIC site or station name AND the question contains ANY past-date reference — including "yesterday", "last Monday", "previous day", "day before", "on [date]", "on 2026-…", "on [month] [day]", a specific calendar date, or any phrase implying a historical/past reading — ALWAYS select "site_daily_averages". Extract stationName and date. NEVER select site_aqi_single for past-date site questions. This rule overrides rules 1 and 2 whenever a past date is present.
               CRITICAL: The word "yesterday" ALWAYS triggers this rule, even if the question also contains words like "current", "AQI", or a specific pollutant name. Past-date signal beats everything else.
               Examples:
               - "current AQI at Al Reyada School yesterday" → site_daily_averages (NOT site_aqi_single), stationName="Al Reyada School", date=yesterday
               - "what is the current AQI at Al Saad Indian School yesterday?" → site_daily_averages, stationName="Al Saad Indian School", date=yesterday
               - "what were the readings at Al Reyada School yesterday?" → site_daily_averages, stationName="Al Reyada School", date=yesterday
               - "air quality at Al Saad Indian School on 2026-09-25" → site_daily_averages, stationName="Al Saad Indian School", date="2026-09-25"
               - "what was the AQI at Kaltham Hasan Al Obeidli last Monday?" → site_daily_averages, stationName="Kaltham Hasan Al Obeidli", date=last Monday
               - "show me the readings at Al Naeem School on October 5" → site_daily_averages, stationName="Al Naeem School", date="2026-10-05"
               - "previous day readings at Abu Dhabi Residential" → site_daily_averages, stationName="Abu Dhabi Residential", date=yesterday
               - "what was the PM2.5 at Al Reyada School yesterday?" → site_daily_averages (NOT site_aqi_single), stationName="Al Reyada School", date=yesterday
               - "AQI at Al Naeem School last Monday" → site_daily_averages, stationName="Al Naeem School", date=last Monday
            1. If the user mentions a SPECIFIC site or station name AND asks for ONE value (AQI, AQI category, or a specific pollutant like CO2, PM2.5, NO2, temperature, humidity) OR asks a safety/health/attendance question about that site — AND the question does NOT contain a past date reference — → select "site_aqi_single". Returns a single sentence — NOT a table.
               When extracting stationName: use ONLY the site name itself — strip any trailing region qualifier like "in Al Ain", "in Abu Dhabi", "in Al Dhafra". Strip trailing words like "site", "station", "location". Example: "Kaltham Hasan AL Obeidli site" → stationName = "Kaltham Hasan AL Obeidli".
            2. If the user mentions a SPECIFIC site or station name AND asks for ALL readings, ALL parameters, ALL current data, or the last 24 hours of readings → select "site_readings_all". Returns a table.
               Same stationName rule: strip region qualifier and trailing generic words from the extracted name.
            2b. If the user asks which schools or sites repeatedly exceeded a given AQI threshold AND had a Data Success Rate (DSR) above a minimum (e.g. "schools that repeatedly exceeded AQI 100 but still had Data Success Rate of at least 75%", "sites that crossed AQI 100 more than once with good data", "exceeded AQI threshold with acceptable DSR") → select "sites_repeated_aqi_exceedance_with_dsr". Extract aqiThreshold (default 100) and dsrThreshold (default 75). NEVER select schools_latest_pollutant for this case — schools_latest_pollutant only shows current/live readings, not historical exceedance counts combined with DSR.
            3. Only select "schools_latest_pollutant" when the user says "schools" generically with a GENERIC plural reference (e.g. "all schools", "schools in Al Ain", "what is the AQI at schools") with NO specific school name mentioned AND is asking for a specific POLLUTANT reading (CO2, PM2.5, AQI value, etc.) WITHOUT an AQI quality level filter AND WITHOUT any historical exceedance or DSR condition. NEVER select schools_latest_pollutant when: (a) a specific school or site name is mentioned, (b) the user is asking about AQI category/quality level (good, moderate, unhealthy, etc.) — use site_aqi_by_category with sectorName="Public & Govt-School" instead, or (c) the user asks about "repeatedly exceeded", "exceedance", "Data Success Rate", or "DSR" combined with AQI — use sites_repeated_aqi_exceedance_with_dsr instead.
            4. If the user asks an AQI question that mentions ONLY a region name (Abu Dhabi, Abudhabi, Al Ain, Alain, Al Dhafra, Aldhafra) with NO specific site name, OR asks for "all regions", "by region", "region wise" → select "region_aqi_geographical". Extract regionName ONLY if exactly ONE specific region is mentioned and the user is asking about that region alone. OMIT regionName (return all regions) when: the user names TWO OR MORE regions (e.g. "Abu Dhabi, Al Ain, and Al Dhafra"), says "each region", "all regions", "every region", "each of the regions", "all three regions", "region wise", or any phrasing that implies the complete set. NEVER extract regionName when the question implies all three. NEVER select region_aqi_geographical when a specific site name is also present — rule 1 takes priority.
            4b. If the user asks a generic AQI question with NO specific site name AND NO region AND NO quality level word → select "site_aqi_all". Examples: "show me the AQI", "what is the AQI", "show the AQI", "current AQI", "display AQI", "AQI at my sites". NEVER select site_aqi_all when: (a) the question names a specific site or person → use site_aqi_single (rule 1), or (b) the question contains a quality word like good, bad, poor, unhealthy, moderate, hazardous, safe, harmful, vulnerable, critical → use site_aqi_by_category (rule 8).
            5. If the user asks to LIST or SHOW which schools/sites exist (no pollutant reading requested) — with or without a region filter — → ALWAYS select "sites_by_filter". Extract BOTH sectorName AND regionName when both are present. NEVER select schools_latest_pollutant, list_stations, or sites_filtered for these. "sites_filtered" must NEVER be used for sector-based questions — it has no sectorName parameter and cannot filter by sector.
               Sector extraction for sites_by_filter:
               - "residential sites in [region]" → sectorName="Residential", regionName=[region]
               - "commercial sites in [region]" → sectorName="Commercial", regionName=[region]
               - "schools in [region]" / "government schools in [region]" / "public and government schools in [region]" / "MOE schools in [region]" / "moe school in [region]" → sectorName="Public & Govt-School", regionName=[region]
               - "residential sites" (no region) → sectorName="Residential", no regionName
               - "commercial sites" (no region) → sectorName="Commercial", no regionName
               Examples: "show me the residential sites in Al Dhafra" → sites_by_filter, sectorName="Residential", regionName="Al Dhafra"
                         "residential sites in Al Ain" → sites_by_filter, sectorName="Residential", regionName="Al Ain"
                         "commercial sites in Abu Dhabi" → sites_by_filter, sectorName="Commercial", regionName="Abu Dhabi"
                         "what are the different schools in Abu Dhabi?" → sites_by_filter, sectorName="Public & Govt-School", regionName="Abu Dhabi"
                         "list schools in Al Ain" → sites_by_filter, sectorName="Public & Govt-School", regionName="Al Ain"
                         "which schools are in Al Dhafra" → sites_by_filter, sectorName="Public & Govt-School", regionName="Al Dhafra"
               EXCEPTION to rule 5: if the user says "schools" generically AND asks for a specific pollutant reading or data (e.g. "show me a table of all Abu Dhabi schools with their latest CO2 reading", "Abu Dhabi schools CO2", "schools in Al Ain with AQI") → select "schools_latest_pollutant" (rule 3), NOT sites_by_filter. Rule 3 wins when a pollutant/reading is explicitly requested.
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
               IMPORTANT — AQI category DEFINITION questions: if the user is asking WHAT an AQI range or category means — including short phrases like "Hazardous AQI value", "Very Unhealthy AQI range", "Good AQI value", "Moderate AQI value", "Unhealthy AQI value", or any phrase of the form "[category] AQI [value/range/number]" or "AQI [value/range] for [category]" — → select "faq_answer". These are knowledge questions about the numeric range or classification definition, NOT requests for site data. NEVER select "site_aqi_by_category" for these — that template lists sites, not definitions. Examples that must route to faq_answer: "Hazardous AQI value", "Very Unhealthy AQI range", "What AQI range is considered Good?", "Which AQI values are classified as Unhealthy?", "What range of AQI is classified as Unhealthy for Sensitive Groups?"
               IMPORTANT — AQI numeric threshold questions: if the user asks about sites or schools with AQI ABOVE/BELOW/MORE THAN/LESS THAN a specific number (e.g. "How many schools have an AQI above 30 in Abu Dhabi?", "sites with AQI above 100", "schools where AQI exceeds 50", "how many sites have AQI less than 80", "Show me all Al Ain schools where the current AQI is above 10", "Al Ain schools with AQI above 50", "schools in Abu Dhabi where AQI exceeds 100") → ALWAYS select "site_aqi_by_category". Extract: sectorName="Public & Govt-School" if "school" or "schools" is mentioned, regionName if a region is mentioned (e.g. "Al Ain", "Abu Dhabi", "Al Dhafra"), minValue from the number when "above/more than/greater than/over/exceeds/at least", maxValue from the number when "below/less than/under/at most". NEVER select "schools_latest_pollutant" for numeric threshold questions — that template cannot filter by AQI value. AQI numeric threshold ALWAYS wins over schools_latest_pollutant even when a region is mentioned.
               IMPORTANT — AQI category with region: if the user asks for sites/locations WHERE AQI IS a quality level (good, moderate, unhealthy, hazardous, etc.) even with a region filter (e.g. "Al Ain sites where AQI is good", "list Abu Dhabi sites with good AQI", "sites in Al Dhafra where AQI is moderate") → ALWAYS select "site_aqi_by_category". Extract aqiCategory AND regionName. NEVER select sites_by_filter or list_stations for these — those return site lists without AQI data.
               IMPORTANT — schools + AQI category: if the user asks about SCHOOLS (e.g. "schools where AQI is good", "school sites with moderate AQI") AND mentions an AQI quality level → select "site_aqi_by_category" with aqiCategory set AND sectorName="Public & Govt-School". Do NOT select schools_latest_pollutant for this case.
            9. If the user asks for the current Data Success Rate, DSR value, data availability %, how many devices are active, active device count, or critical site count → select "data_success_rate". Extract year if mentioned (e.g. "2025", "2026"); omit year to default to the current year. NEVER select faq_answer for questions asking for the CURRENT or ACTUAL value of the DSR or active device count — those must come from the live API.
            9j. If the user asks which region had the poorest/worst/best air quality over the last N years considering data completeness (multi-year comparison) → select "region_aqi_trend_yearly". Extract years (default 3). NEVER select region_aqi_geographical or region_aqi_single_year for multi-year questions.
            9i. If the user asks which region performed best or worst in a SPECIFIC YEAR (e.g. "Which region performed best in 2025 when both AQI and data completeness are considered?") → select "region_aqi_single_year". Extract year. NEVER select region_aqi_trend_yearly (that covers multiple years).
            9h. If the user asks what drove the worst air-quality month at a specific site in a given year (e.g. "What drove the worst air-quality month at Al Saad Indian School in 2025?") → select "worst_month_pollutants". Extract stationName and year. NEVER select faq_answer or site_aqi_trend_yearly for this.
            9g. If the user asks whether air quality at a specific site has improved or worsened over the last N years, or asks for a yearly AQI trend/history at a site (e.g. "Has air quality at Al Saad Indian School improved or worsened over the last three years?") → select "site_aqi_trend_yearly". Extract stationName and years (default 3). NEVER select faq_answer or top_sites_aqi_yearly for this — those do not give a per-site trend.
            9f. If the user asks to compare the indoor AQI at a site with the nearest ambient (outdoor) station (e.g. "Compare the current indoor AQI at Al Saad Indian School with the nearest ambient station") → select "indoor_ambient_compare". Extract stationName. The system fetches the nearest ambient station automatically from Abu Dhabi SDI. NEVER select region_aqi_geographical or site_aqi_single for this.
            9f-iii. If the user asks whether any pollutant (PM2.5, PM10, CO2, CO, NO2, O3, TVOC, etc.) at a residential or indoor site is coming from indoor sources (bakhoor, cooking, incense, ventilation) or from outdoor infiltration/dust, or asks for a source analysis of any parameter over a period (e.g. "Is the PM2.5 at Al Reem Residence more likely coming from indoors or outdoor dust over the last 30 days?", "Is the PM10 at Al Reem Residence from indoor or outdoor sources from 2026-03-01?", "What is causing elevated CO2 at the residence from 2026-02-01 to 2026-02-28?") → select "indoor_source_analysis". Extract stationName, parameterName (default PM2.5; use exactly as stated by user, e.g. PM10, PM2.5, CO2, CO, NO2, O3, TVOC), startDate and endDate (as YYYY-MM-DD). The query compares indoor 24-hour average (SiteParameterAverages) against the nearest outdoor ambient station (AmbientData) and computes an I/O ratio. NEVER select dust_event_response (that is for dust events on a specific date), site_aqi_single, or faq_answer for this.
            9f-ii. If the user asks how indoor air quality at a specific school or site responded during a dust event, dust storm, or sandstorm on a specific date (e.g. "How did indoor air quality at Al Saad Indian School respond during the dust event on 2026-03-15?") → select "dust_event_response". Extract stationName and date (as YYYY-MM-DD). Optionally extract parameterName (default PM10; allowed: PM10, PM25, O3, SO2, NO2, CO, AQI). The key signals are: a site name + a specific date + the words "dust event", "dust storm", "sandstorm", or "respond during". NEVER select indoor_ambient_compare (that is live/current only), site_parameter_trend (that shows multi-day history with no outdoor comparison), site_aqi_single, or faq_answer for this.
            9k. If the user asks what the main, dominant, primary, worst, or highest pollutant is at a specific named site RIGHT NOW (e.g. "What is the main pollutant at Al Reyada School?", "Which pollutant is driving the AQI at Al Saad Indian School?", "What is the dominant pollutant at Kaltham Hasan Al Obeidli?") → select "main_pollutant_at_station". Extract stationName only. This uses the last 1-hour averaged readings to rank pollutants by their exceedance ratio. NEVER select site_aqi_single or faq_answer for this — those do not return pollutant ranking data.
            9e. If the user asks which devices had the lowest Data Success Rate last month (e.g. "which devices had the lowest DSR last month?", "worst data success rate last month") → select "lowest_dsr_devices". Extract topN if mentioned (default 10). NEVER select data_success_rate (that returns overall network DSR, not per-device) or device_compliance_rate (that requires a specific stationId).
            9f. If the user asks which device has been without data / offline / silent for the longest time → select "device_longest_silent". Set topN=1 when user says "which device" (singular). Set topN to the number mentioned if user says "top 5", "show me 10", etc. Default is 1. Extract regionName, sectorName, stationName if the user filters by region/sector/site. Only inactive devices (no reading in last 15 min) are returned.
            9d. If the user asks which device contributed most to the highest site AQI at a school or site last month (e.g. "which device contributed most to the highest site AQI at Al Saad Indian School last month?") → select "device_peak_aqi_contribution". Extract stationName only — the date range is always last month (hardcoded in SQL). NEVER select device_live_reading, compare_two_stations, or schools_latest_pollutant for this.
            9c. If the user asks to compare the two devices at a specific school or site for a specific month (e.g. "compare the two devices at Al Saad Indian School for September 2026", "device comparison at Al Naeem School for August") → select "compare_devices_monthly". Extract stationName (site/school name), month (as integer 1–12), and year (4-digit). NEVER select compare_two_stations or device_live_reading for this — those do not group by device per month.
            9b. If the user asks which sites had the HIGHEST average AQI last year, in a specific year, or wants a worst/most-polluted year-based ranking (e.g. "which 10 sites had the highest average AQI last year?", "top sites by AQI in 2024", "worst sites AQI 2023") → select "top_sites_aqi_yearly". Extract topN (default 10) and year (default: current year minus 1). If the user asks for LOWEST or BEST average AQI last year (cleanest, safest, best air quality) → select "bottom_sites_aqi_yearly" instead. NEVER select faq_answer, site_aqi_ranking, or highest_aqi_today for year-based AQI ranking questions.
            10. DEVICE COUNT vs DEVICE LIST routing — this is critical:
            - "HOW MANY / TOTAL / COUNT" + region/sector/status (no site name) → ALWAYS select "count_devices_by_filter"
              Examples:
              - "how many inactive devices in Abu Dhabi" → count_devices_by_filter, regionName="Abu Dhabi", statusFilter="Inactive"
              - "how many active devices in Al Ain" → count_devices_by_filter, regionName="Al Ain", statusFilter="Active"
              - "total inactive devices in Al Ain residential sector" → count_devices_by_filter, regionName="Al Ain", sectorName="Residential", statusFilter="Inactive"
              - "how many inactive devices in commercial sector" → count_devices_by_filter, sectorName="Commercial", statusFilter="Inactive"
              - "how many active devices in Abu Dhabi school sector" → count_devices_by_filter, regionName="Abu Dhabi", sectorName="Public & Govt-School", statusFilter="Active"
              - "device count in Al Ain" → count_devices_by_filter, regionName="Al Ain"
              - "how many devices in commercial" → count_devices_by_filter, sectorName="Commercial"
            - "LIST / SHOW / WHICH / display" + region/sector/status (or a site name) → select "devices_by_filter"
              Examples:
              - "inactive devices under Abu Dhabi region" → devices_by_filter, regionName="Abu Dhabi", statusFilter="Inactive"
              - "list active devices in commercial sector" → devices_by_filter, sectorName="Commercial", statusFilter="Active"
            NEVER select count_devices_by_filter when the user wants to see device names. NEVER select devices_by_filter when the user only wants a number.
            10b. If the user asks to LIST, SHOW, or COUNT devices under a SPECIFIC SITE or SCHOOL name (e.g. "devices under Al Naeem School", "devices at Al Saad Indian School", "devices available at Al Ain Commerical Com site") → select "devices_by_filter". Extract stationName as the site name keyword. The SQL uses LIKE so a partial keyword works. NEVER select list_devices (system-wide total) for site-specific questions.
            "devices_by_filter" is the SINGLE template for ALL device listing and status questions — regardless of filter combination. Use it for:
            - "which devices are offline/online/active/inactive" (no filter) → statusFilter="Inactive"/"Active"
            - "offline devices in Abu Dhabi" → regionName="Abu Dhabi", statusFilter="Inactive"
            - "active devices in commercial sector" → sectorName="Commercial", statusFilter="Active"
            - "devices at Al Reyada School" → stationName="Al Reyada School"
            - "inactive devices in Al Ain commercial sector" → regionName="Al Ain", sectorName="Commercial", statusFilter="Inactive"
            - "school devices that are offline" → sectorName="Public & Govt-School", statusFilter="Inactive"
            - "online devices in all schools" → sectorName="Public & Govt-School", statusFilter="Active"
            - "online devices in schools" → sectorName="Public & Govt-School", statusFilter="Active"
            - "active devices in all schools" → sectorName="Public & Govt-School", statusFilter="Active"
            - "offline devices in all schools" → sectorName="Public & Govt-School", statusFilter="Inactive"
            CRITICAL: "online/offline/active/inactive devices in [sector/schools]" is ALWAYS a device query → devices_by_filter. NEVER route to schools_latest_pollutant — that shows pollutant readings, not device status.
            - "how many offline devices in Abu Dhabi" → regionName="Abu Dhabi", statusFilter="Inactive" (LLM returns count from rows)
            - "which devices are currently available at Al Ain Commercial site" → regionName="Al Ain", sectorName="Commercial" (NO stationName — "Al Ain" is a region, "Commercial" is a sector)
            - "devices in Abu Dhabi Residential" → regionName="Abu Dhabi", sectorName="Residential" (NOT a stationName)
            - "devices in Al Ain Commercial Com site" → regionName="Al Ain", sectorName="Commercial"
            CRITICAL for regionName/sectorName vs stationName disambiguation (devices_by_filter only):
            - If the phrase contains BOTH a known region ("Abu Dhabi", "Al Ain", "Al Dhafra") AND a known sector word ("Commercial", "Residential", "School"), treat them as regionName + sectorName — do NOT set stationName.
            - Only set stationName if the user mentions a real site name that is NOT just a region+sector combo (e.g. "Al Reyada School", "Al Bateen Airport").
            Always extract: statusFilter="Inactive" for offline/inactive/not sending/down/disconnected; statusFilter="Active" for online/active/sending/working/connected. NEVER set statusFilter for "available", "installed", "at", "under", "in" alone — those are location prepositions, not status words. The status word must appear explicitly: "offline devices under Abu Dhabi" → statusFilter="Inactive" (triggered by "offline"), regionName="Abu Dhabi" (triggered by "under Abu Dhabi").
            Leave statusFilter empty when no status filter is mentioned (show all devices with status column).
            11. For site listing/filtering questions where the user wants full details (site names, show/list/display) → "sites_by_filter". Use it for: "show me sites", "list sites in Al Ain", "sites in commercial sector", "what are the active sites", "sites region wise". NEVER select schools_latest_pollutant or sector_latest_pollutant for plain site listing questions. NEVER select sites_by_filter when the user asks for COUNT/NUMBER of sites per region or sector — that must be site_count.
            11e. For ALL site COUNT questions → select "site_count". Set params as follows:
              - "how many sites total", "total sites in HAWAQM" → no params (LLM reports total only, no breakdown)
              - "how many sites per region", "site count by region", "how many sites in each region" → no params (LLM groups by RegionName, reports totals only)
              - "how many sites per sector", "site count by sector", "how many sites in each sector" → no params (LLM groups by SectorName, reports totals only)
              - "how many sites in Abu Dhabi" (specific region) → regionName="Abu Dhabi"
              - "how many active sites" → liveStatus="Active"
              - "how many inactive sites" → liveStatus="Inactive"
              - "how many sites have no devices", "sites without devices", "sites with no device attached" → liveStatus="NoDevice"
              - "how many active sites in Al Ain" → regionName="Al Ain", liveStatus="Active"
              - "site status breakdown", "active inactive and no device count" → no params (LLM reports full breakdown with Active/Inactive/NoDevice)
              - "how many schools are active" → sectorName="Public & Govt-School", liveStatus="Active"
              - "how many schools are active under Al Ain" → sectorName="Public & Govt-School", regionName="Al Ain", liveStatus="Active"
              - "how many schools are inactive in Abu Dhabi" → sectorName="Public & Govt-School", regionName="Abu Dhabi", liveStatus="Inactive"
              - "how many active schools in Al Dhafra" → sectorName="Public & Govt-School", regionName="Al Dhafra", liveStatus="Active"
              - "how many commercial sites are active" → sectorName="Commercial", liveStatus="Active"
              - "how many residential sites are inactive in Al Ain" → sectorName="Residential", regionName="Al Ain", liveStatus="Inactive"
              - "total active schools" → sectorName="Public & Govt-School", liveStatus="Active"
              - "count of active school sites" → sectorName="Public & Govt-School", liveStatus="Active"
              CRITICAL: "how many schools", "how many active schools", "how many inactive schools", "total schools", "count of schools" are COUNT questions → ALWAYS select site_count with sectorName="Public & Govt-School". NEVER select sites_by_filter for these.
              NEVER select sites_by_filter for count questions — it returns individual site rows, not counts.
            11f. If the user asks to LIST/SHOW which sites are active, inactive, or have no device (e.g. "what are the active sites in Al Ain", "show me inactive sites in Abu Dhabi", "which sites are active", "sites with no device", "sites without devices", "sites with zero devices", "which sites have no sensor", "list sites without any device") → select "sites_by_filter" with liveStatus="Active", "Inactive", or "NoDevice" respectively. NEVER select site_count for list questions.
              - "no device" / "without device" / "zero device" / "without any device" / "no sensor" / "without sensor" → liveStatus="NoDevice"
              - "active" / "online" → liveStatus="Active"
              - "inactive" / "offline" → liveStatus="Inactive"
            CRITICAL — "sites_filtered" must NEVER be selected when the user mentions a sector name ("residential", "commercial", "school", "government school"). "sites_filtered" has no sectorName parameter and will ignore the sector filter entirely. Always use "sites_by_filter" for any question that includes a sector word.
            11a. If the user asks how many SECTORS exist, what sectors are in HAWAQM, or list of sectors (with NO specific sector name) → select "list_sectors". NEVER select site_count (that shows sites per sector, not the sector list).
            11b. If the user asks how many SUB-SECTORS exist, what sub-sectors are in HAWAQM, or list of sub-sectors → select "list_subsectors". NEVER select sites_by_filter or list_sectors for sub-sector questions.
            11c. If the user asks for a PIE CHART or DONUT CHART or DISTRIBUTION CHART of sites or schools by AQI category (e.g. "create a pie chart of schools by their current AQI category", "show me a pie chart of AQI distribution", "AQI category breakdown as a pie chart") → select "schools_aqi_pie". If the user mentions "schools" or "school sites", extract sectorName="Public & Govt-School". If a region is mentioned, extract regionName. NEVER select site_aqi_by_category or site_aqi_all for pie/donut/distribution chart questions.
            11c-ii. If the user asks for a PIE CHART or BAR CHART showing the NUMBER or COUNT of sites in each REGION (e.g. "pie chart of sites by region", "how many sites per region as a chart", "show number of sites in each region") → select "sites_by_filter_chart". No parameters needed.
            11c-iii. If the user asks for a PIE CHART or BAR CHART showing the NUMBER or COUNT of sites in each SECTOR (e.g. "pie chart of sites by sector", "how many sites per sector as a chart", "show number of sites in each sector") → select "sites_by_filter_chart". No parameters needed.
            11c-iv. If the user asks for a chart of AVERAGE AQI by REGION (e.g. "bar chart of AQI by region", "average AQI per region chart", "AQI comparison across regions") → select "aqi_by_region_chart". No parameters needed.
            11c-v. If the user asks for a chart of SITE STATUS (Active vs Offline/Inactive) (e.g. "pie chart of site status", "active vs offline sites chart", "site availability breakdown") → select "sites_by_status_chart". No parameters needed.
            11c-vi. If the user asks for a chart of a POLLUTANT or AQI by SECTOR (e.g. "bar chart of PM2.5 by sector", "average AQI by sector", "pollutant levels per sector") → select "pollutant_by_sector_chart". Extract parameterName (e.g. "PM2.5", "PM10", "AQI Index"). Default to "AQI Index" if not specified.
            11c-vii. If the user asks for a pie or bar chart of devices grouped by model/type (e.g. "show me the distribution of devices by model", "pie chart of device models", "how many devices of each model") → select "devices_by_model_chart". No parameters needed. NEVER select site_device_summary or list_devices for this.
            11c-viii. If the user asks for a pie or bar chart based on school sub-sectors or type of school (private/public/government) (e.g. "plot a pie chart based on school subsectors", "pie chart of type of school", "private vs public school chart") → select "sites_by_subsector_chart". No parameters needed. NEVER select sites_by_filter_chart (that groups by main sector, not sub-sector).
            11c-ix. If the user asks for a pie or bar chart of sites based on AQI levels/categories (e.g. "sites based on AQI levels", "AQI level distribution chart", "pie chart of sites by AQI category") → select "sites_aqi_level_chart". No parameters needed. NEVER select schools_aqi_pie (that is school-sector only) or site_aqi_by_category (that returns a filtered table, not a chart) for this.
            11d. If the user asks about TOTAL DEVICES or DEVICES PER SITE across ALL sites (e.g. "total number of devices per site in the application", "how many devices does each site have", "device count per site", "total devices across all sites") — even without naming a specific site — select "site_device_summary". No parameters needed. NEVER select site_aqi_single for this type of question.
            11d-i. If the user asks about DEVICES PER SITE — including:
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
              NEVER select active_device_count or devices_by_filter for device-per-site questions.

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
              - "from YYYY-MM-DD to YYYY-MM-DD"  → startDate = first date, endDate = second date (extract both exactly)
              - "from YYYY-MM-DD" (no end date)  → startDate = that date, endDate = that same date (single-day query)
              - "since YYYY-MM-DD" (no end date) → startDate = that date, endDate = that same date
              CRITICAL: NEVER invent or guess an endDate beyond the stated date. If only one date is given with no range, startDate = endDate = that date.
              Always compute using the "Today's date" provided in the user message.
            For columns: use lowercase snake_case (e.g. pm25, not PM2.5).
            For year: extract a 4-digit year if the user mentions one (e.g. "2025", "last year"). If no year is mentioned, omit it — the system will default to the current year automatically.
            For regionName: extract the region(s) the user mentions (words like "in", "under", "for", "within", "at", "and" between region names all indicate regions). Map to: "Abu Dhabi" (also: abudhabi, abu-dhabi, abudabi), "Al Ain" (also: alain, al-ain), "Al Dhafra" (also: aldhafra, al-dhafra, dhafra). If the user mentions MORE THAN ONE region (e.g. "Abu Dhabi and Al Dhafra", "Al Ain or Abu Dhabi"), output ALL matched regions joined by "|" (pipe). Examples: "Abu Dhabi and Al Dhafra" → "Abu Dhabi|Al Dhafra"; "Al Ain and Al Dhafra" → "Al Ain|Al Dhafra"; "all three regions" → omit. If the user asks about all regions or does not mention a specific region, omit regionName.
            For sectorName: there are exactly 3 sectors in HAWAQM — Commercial, Residential, Public & Govt-School. There is NO Industrial sector. Map ALL user variations to one of these three canonical values only:
              → "Commercial": "commercial", "commercial sector", "commercial sites", "commercial sites in"
              → "Public & Govt-School": ANY phrase containing "school", "schools", "govt", "gov", "government", "public & govt", "public & gov", "public and gov", "public and govt", "public and government", "govt-school", "gov-school", "government school", "govt school", "gov school", "educational", "private and gov-school", "private and govt school", "public and government schools", "government schools", "which public and government schools", "show me public and government schools", "list government schools", "moe", "moe school", "moe schools", "MOE", "MOE school", "MOE schools", "ministry of education school", "ministry of education schools"
              → "Residential": "residential", "residential sector", "residential sites"
            IMPORTANT: "public & Gov School", "Public and Government School", "Public & Government school", "gov school", "govt school" ALL map to "Public & Govt-School".
            IMPORTANT: When the template is "site_aqi_by_category" and the user mentions "schools" or "school sites", ALWAYS extract sectorName="Public & Govt-School".
            IMPORTANT: When the template is "sites_by_filter" and the user mentions "schools", "school", "different schools", "schools present", "which schools", "list schools", "government schools", "govt schools", "public and government schools", "public & government schools", "public and govt schools", "moe", "moe school", "moe schools", "MOE schools", "ministry of education schools" → ALWAYS extract sectorName="Public & Govt-School".
            IMPORTANT: When the template is "sites_by_filter", there are exactly 3 valid sectors. Map the user's words and extract BOTH sectorName AND regionName when a region is mentioned:
              → sectorName="Residential": user says "residential", "residential sites", "residential sector", "residential sites in [region]"
                Examples: "residential sites in Al Dhafra" → sectorName="Residential", regionName="Al Dhafra"
                          "residential sites in Al Ain" → sectorName="Residential", regionName="Al Ain"
                          "residential sites in Abu Dhabi" → sectorName="Residential", regionName="Abu Dhabi"
                          "show me all residential sites" → sectorName="Residential", no regionName
              → sectorName="Commercial": user says "commercial", "commercial sites", "commercial sector", "commercial sites in [region]"
                Examples: "commercial sites in Al Ain" → sectorName="Commercial", regionName="Al Ain"
                          "commercial sites in Abu Dhabi" → sectorName="Commercial", regionName="Abu Dhabi"
                          "show all commercial sites" → sectorName="Commercial", no regionName
              → sectorName="Public & Govt-School": user says "schools", "government schools", "public and government schools", "govt schools", "public & govt schools", "school sites in [region]", "moe schools", "moe school", "MOE schools", "ministry of education schools"
                Examples: "public and government schools in Al Ain" → sectorName="Public & Govt-School", regionName="Al Ain"
                          "government schools in Al Dhafra" → sectorName="Public & Govt-School", regionName="Al Dhafra"
                          "schools in Abu Dhabi" → sectorName="Public & Govt-School", regionName="Abu Dhabi"
                          "MOE schools in Al Ain" → sectorName="Public & Govt-School", regionName="Al Ain"
                          "moe schools in Abu Dhabi" → sectorName="Public & Govt-School", regionName="Abu Dhabi"
            ALWAYS extract regionName separately when a region (Al Ain, Abu Dhabi, Al Dhafra) is mentioned alongside a sector. Never let the region override the sector or vice versa — extract both.
            IMPORTANT: When the template is "sites_by_filter" and user mentions MULTIPLE sectors (e.g. "residential and commercial sites", "schools and residential") → omit sectorName so all sectors are shown, but mention the filter in the intro sentence.
            IMPORTANT: When the template is "sites_by_filter":
               - Extract regionName ONLY when user explicitly names a region ("in Al Ain", "in Abu Dhabi", "in Al Dhafra"). If user says "by region", "all regions", "each region", "region wise", "site count by region" — omit regionName (all regions shown).
               - Extract sectorName ONLY when user explicitly names a SINGLE sector ("commercial", "residential", "school"). If user says "by sector", "each sector", "site count by sector" — omit sectorName (all sectors shown).
               - Extract liveStatus ONLY when user explicitly says "active" or "inactive" as a site status filter.
               - Do NOT default any param. Omitted params = show all (no filter applied).
            If the user says something that doesn't match any of these three, omit sectorName.
            For parameterName: map common user words to the allowed parameter name — "co2", "carbon dioxide" → "CO2"; "pm2.5", "fine particles", "fine dust" → "PM2.5"; "pm10", "coarse particles", "coarse dust" → "PM10"; "co", "carbon monoxide" → "CO"; "aqi", "air quality index" → "AQI Index"; "no2", "nitrogen dioxide" → "NO2"; "so2", "sulphur dioxide", "sulfur dioxide" → "SO2"; "o3", "ozone" → "O3"; "tvoc", "voc", "volatile organic" → "VOC"; "ch2o", "formaldehyde" → "CH2O"; "temperature", "temp" → "Temperature"; "humidity" → "Humidity"; "noise", "sound" → "Noise".
            IMPORTANT — pollutant groups: if the user asks for "physical" readings/pollutants/parameters, do NOT extract a single parameterName — omit it (the system will filter to physical group: PM2.5, PM10, Temperature, Humidity, Noise). If the user asks for "chemical" readings/pollutants/parameters, omit parameterName as well (system filters to chemical group: CO, CO2, NO2, SO2, O3, CH2O, VOC). Only extract a specific parameterName when the user names ONE specific pollutant.
            For stationName: extract the site or station name the user mentions. CRITICAL — strip ONLY a trailing "in [region]" qualifier that follows the site name. Do NOT strip a region word that is part of the site name itself. Examples:
              - "Al Saad Indian School in Al Ain" → stationName = "Al Saad Indian School"   ← strip trailing "in Al Ain"
              - "readings at Al Bateen School in Abu Dhabi" → stationName = "Al Bateen School"   ← strip trailing "in Abu Dhabi"
              - "Abu Dhabi Residential site" → stationName = "Abu Dhabi Residential"   ← "Abu Dhabi" is part of the site name
              - "Al Ain Commercial Institutional" → stationName = "Al Ain Commercial Institutional"   ← "Al Ain" is part of the site name
              - "AQI at Abu Dhabi Commercial Institutional Site in Abu Dhabi" → stationName = "Abu Dhabi Commercial Institutional Site"   ← strip trailing "in Abu Dhabi" only
              - "Mubarak Bin Mohammed School Cycle 2" → stationName = "Mubarak Bin Mohammed School Cycle 2"   ← extract full name including cycle number
              - "air quality at Mleih Cycle 1 School" → stationName = "Mleih Cycle 1 School"   ← include cycle number
              RULE: if the region word appears BEFORE the site-type word (school, site, building, etc.), it is part of the name. If it appears AFTER with "in"/"at", it is a qualifier — strip it.
            If the user says ONLY a region name ("Abu Dhabi", "Al Ain", "Al Dhafra") with no site name, do NOT extract stationName — that is a region query, not a site query.
            Never extract or guess a numeric ID — always use the name.
            For deviceName: extract the device name exactly as stated in the CURRENT question only. Real device name patterns in this system are: "BA 0001", "BA 0005", "BA 0010" (prefix "BA" + space + 4-digit number) and "SEI100M 0014", "SEI100M 0085", "SEI100M 0148" (prefix "SEI100M" + space + 4-digit number). Users may omit the space or use different casing — extract whatever they wrote (e.g. "ba0010", "BA0010", "BA 0010", "sei100m 0014" are all valid). Never guess or invent a device name — only extract what is explicitly stated in the current question.
            For parameterName (device readings only): if the user asks for a specific parameter, map it — "aqi", "air quality index" → "AQI Index"; "pm2.5", "fine particles", "fine dust" → "PM2.5"; "pm10", "coarse particles" → "PM10"; "co2", "carbon dioxide" → "CO2"; "co", "carbon monoxide" → "CO"; "no2", "nitrogen dioxide" → "NO2"; "so2", "sulphur dioxide", "sulfur dioxide" → "SO2"; "o3", "ozone" → "O3"; "temperature", "temp" → "Temperature"; "humidity" → "Humidity"; "voc", "tvoc", "volatile organic" → "VOC"; "noise", "sound" → "Noise"; "ch2o", "formaldehyde" → "CH2O".
            CRITICAL — pollutant group words for device queries: if the user says "physical" (or "physical readings", "physical pollutants", "physical parameters"), omit parameterName — the controller will return PM2.5, PM10, Temperature, Humidity, Noise. If the user says "chemical" (or "chemical readings", "chemical pollutants", "chemical parameters"), omit parameterName — the controller will return CO, CO2, NO2, SO2, O3, CH2O, VOC. Only extract a specific parameterName when the user names a single explicit pollutant.
            If the user asks for "reading", "readings", "all readings", "all parameters", "all data", "physical", or "chemical" — omit parameterName entirely.
            For statusFilter (devices_by_filter AND count_devices_by_filter): extract "Active" if user says "active", "online", "working", "sending", "connected"; extract "Inactive" if user says "inactive", "offline", "not sending", "down", "disconnected"; omit (do NOT set statusFilter) if the question uses ONLY location words like "available", "installed", "at", "under", "in" with no explicit active/inactive/online/offline word. The status word must be present explicitly.
              Examples:
              - "offline devices under Abu Dhabi region" → statusFilter="Inactive", regionName="Abu Dhabi"
              - "inactive devices under Al Ain" → statusFilter="Inactive", regionName="Al Ain"
              - "active devices under Al Dhafra" → statusFilter="Active", regionName="Al Dhafra"
              - "offline devices in commercial sector" → statusFilter="Inactive", sectorName="Commercial"
              - "inactive devices in Abu Dhabi residential sector" → statusFilter="Inactive", regionName="Abu Dhabi", sectorName="Residential"
              - "devices under Abu Dhabi" (no status word) → regionName="Abu Dhabi", NO statusFilter
              - "devices in commercial" (no status word) → sectorName="Commercial", NO statusFilter
            IMPORTANT — for devices_by_filter: when the user mentions "school", "schools", "school sites", "school sector", "government school", "govt school", "public school", "private school", "MOE school", "MOE schools", "moe school", "moe schools", "moe", "ministry of education school" → ALWAYS extract sectorName="Public & Govt-School". The school sector in HAWAQM is always "Public & Govt-School" and covers both private and government schools. Never leave sectorName empty when the user asks about schools in a device query.
            For liveStatus (sites_by_filter only): extract "Active" ONLY if user explicitly says "active sites", "active stations", "online sites", "currently active"; extract "Inactive" ONLY if user explicitly says "inactive sites", "inactive stations", "offline sites", "not active". If the user mentions MORE THAN ONE status (e.g. "active and inactive sites"), join with "|": "Active|Inactive". CRITICAL: if the user says "all sites", "show sites", "list sites", "give all sites", or any generic site list phrase WITHOUT a status word — omit liveStatus entirely. Never guess or infer liveStatus from context. Example: "active sites in Al Ain" → liveStatus="Active", regionName="Al Ain".
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
            For topN (schools_latest_pollutant and sector_latest_pollutant): extract a positive integer when user says "top N", "top 5", "top 10", "highest N", "best N", "worst N", "first N", "show me N". Examples: "top 5 schools" → topN: "5", "top 10 sites with highest AQI" → topN: "10". Omit if user does not specify a count limit.
            For minValue (site_aqi_by_category and schools_latest_pollutant): extract a numeric AQI lower bound when user says "more than X", "greater than X", "above X", "over X", "exceeding X", "higher than X", "worse than X". Use the number only (e.g. "AQI more than 30" → minValue: "30", "above 50" → minValue: "50", "higher than 50" → minValue: "50").
            For maxValue (site_aqi_by_category and schools_latest_pollutant): extract a numeric AQI upper bound when user says "less than X", "below X", "under X", "at most X", "lower than X", "better than X". Use the number only (e.g. "AQI less than 100" → maxValue: "100", "lower than 50" → maxValue: "50").
            IMPORTANT — comparative category phrases map to minValue/maxValue (NOT aqiCategory). AQI category upper bounds: Good=50, Moderate=100, Unhealthy for Sensitive Groups=150, Unhealthy=200, Very Unhealthy=300. Use the upper bound of the named category as the threshold:
              "worse than Good" / "more than Good" / "beyond Good" / "above Good" → minValue="50" (AQI > 50)
              "worse than Moderate" / "more than Moderate" / "beyond Moderate" / "above Moderate" → minValue="100" (AQI > 100)
              "worse than Unhealthy for Sensitive Groups" → minValue="150"
              "worse than Unhealthy" → minValue="200"
              "worse than Very Unhealthy" → minValue="300"
              "better than Moderate" / "less than Moderate" / "below Moderate" → maxValue="100" (AQI ≤ 100)
              "better than Unhealthy for Sensitive Groups" → maxValue="150"
              "better than Unhealthy" → maxValue="200"
              "better than Very Unhealthy" → maxValue="300"
              "not Good" / "no longer Good" / "outside Good" → minValue="50"
              Do NOT set aqiCategory when using comparative phrases — use minValue/maxValue instead.
            IMPORTANT — for site_aqi_by_category with AQI range: when user says "AQI more than 30 in Al Ain" → minValue="30", regionName="Al Ain". When "AQI less than 100 in Abu Dhabi" → maxValue="100", regionName="Abu Dhabi". When "AQI between 50 and 100" → minValue="50", maxValue="100". Do NOT extract aqiCategory when the user specifies a numeric range — use minValue/maxValue instead.
            IMPORTANT — for schools_latest_pollutant with numeric threshold: when user says "schools with AQI higher than 50", "Al Ain schools with AQI above 30", "schools where AQI is more than 100" → select schools_latest_pollutant, extract parameterName="AQI Index", minValue="50"/"30"/"100", regionName if mentioned. NEVER select site_aqi_by_category for school-specific threshold questions.
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
