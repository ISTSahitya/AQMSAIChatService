# HAWAQM AI Chat Service — Complete Backend Flow

> **Purpose:** Developer reference document describing exactly what happens in the backend from the moment a user submits a chat message to the moment a response is returned.

---

## Overview

```
User types question
       ↓
HTTP POST /api/chat
       ↓
JWT Authentication → UserContext (role, permitted site0s)
       ↓
Scope Validation
       ↓
Session Management
       ↓
Query Routing (keyword scoring)
       ↓
LLM: Template Selection OR Parameter Filling
       ↓
RBAC Enforcement (ValidateSelection + BuildSafeQuery)
       ↓
[FAQ short-circuit] OR [SQL path] OR [API path]
       ↓
Natural Language Summary (LLM)
       ↓
Conversation Persistence
       ↓
Response Formatting → HTTP 200
```

---

## Step 1 — HTTP Request Entry

The frontend sends:
```
POST /api/chat
Authorization: Bearer <jwt>
Content-Type: application/json

{
  "message": "What is the AQI at Al Ain Industrial Site?",
  "sessionId": "<optional existing session GUID>",
  "scope": { "siteId": null, "startDate": null, "endDate": null, "region": null }
}
```

The ASP.NET **JwtBearer middleware** validates the Bearer token using the configured signing key (symmetric secret in dev, certificate/DefaultAzureCredential in prod). If the token is invalid, expired, or missing → **401 Unauthorized** immediately.

---

## Step 2 — JWT Parsing and UserContext Building

### 2a — JWT Bearer configuration

**File:** `Program.cs` (lines 77–136)

Before any request reaches the controller, ASP.NET's built-in JwtBearer middleware validates the token. Configuration is read from `appsettings.json` under the `"Jwt"` section:

```csharp
// Lines 78–81: read config values
var jwtSection  = builder.Configuration.GetSection("Jwt");
var jwtSecret   = jwtSection["Secret"];    // symmetric HMAC-SHA256 key
var jwtIssuer   = jwtSection["Issuer"];   // e.g. "http://localhost:3000"
var jwtAudience = jwtSection["Audience"]; // e.g. "http://localhost:3000"
```

**Safety guard** (line 83): if `Jwt:Secret` is empty in Production → app throws at startup and refuses to start. In dev with no secret → no-op auth is registered instead (line 135) and a warning is logged.

```csharp
// Lines 88–130: register JwtBearer when secret is configured
builder.Services.AddAuthentication(...)
    .AddJwtBearer(options =>
    {
        options.MapInboundClaims = false;  // ← CRITICAL: keeps claim names as-is
                                           //   without this, "Role" becomes the long
                                           //   http://schemas.microsoft.com/... URI
        options.TokenValidationParameters = new TokenValidationParameters
        {
            ValidateIssuerSigningKey = true,
            IssuerSigningKey = new SymmetricSecurityKey(
                Encoding.UTF8.GetBytes(jwtSecret)),   // same key AQMS API uses to sign

            ValidateIssuer   = !string.IsNullOrWhiteSpace(jwtIssuer),
            ValidIssuer      = jwtIssuer,

            ValidateAudience = !string.IsNullOrWhiteSpace(jwtAudience),
            ValidAudience    = jwtAudience,

            ValidateLifetime = true,
            ClockSkew        = TimeSpan.FromMinutes(5),  // 5-min grace on expiry

            NameClaimType = "UserName",  // User.Identity.Name resolves from "UserName" claim
            RoleClaimType = "Role"       // User.IsInRole() resolves from "Role" claim
        };

        // Events for logging (lines 111–129)
        options.Events = new JwtBearerEvents
        {
            OnAuthenticationFailed = ctx => { Log.Warning("JWT auth failed: {Error}", ...); },
            OnTokenValidated       = ctx => { Log.Information("Token validated for: {User}", ...); },
            OnChallenge            = ctx => { Log.Warning("JWT challenge: {Error}", ...); }
        };
    });
```

**Middleware pipeline order** (`Program.cs` lines 222–262):
```
UseAuthentication()                        ← JwtBearer validates token, populates User.Claims
UseAuthorization()                         ← [Authorize] attribute enforced on ChatController
UseMiddleware<RequestLoggingMiddleware>()   ← builds UserContext from validated claims
MapControllers()                           ← ChatController.Chat() is called
```

---

### 2b — UserContext construction

**File:** `Middleware/RequestLoggingMiddleware.cs` → `InvokeAsync()` (line 24)

After JWT validation populates `context.User.Claims`, the custom middleware runs:

```csharp
public async Task InvokeAsync(HttpContext context, IUserSiteService userSiteService)
{
    // 1. Extract UserId — check DOH claim first, then standard JWT claims
    var userIdStr = context.User?.FindFirst("UserId")?.Value          // DOH-issued token
                 ?? context.User?.FindFirst("sub")?.Value             // standard JWT
                 ?? context.User?.FindFirst("...nameidentifier")?.Value
                 ?? "anonymous";

    if (context.User?.Identity?.IsAuthenticated == true)
    {
        var userCtx = BuildUserContext(context, userIdStr, ip);       // line 41

        // 2. Resolve permitted sites from AQMS DB (requires numeric UserId)
        if (int.TryParse(userIdStr, out var numericUserId) && numericUserId > 0)
            userCtx.PermittedSiteIds = await userSiteService
                .GetPermittedSiteIdsAsync(numericUserId, ct);         // line 46

        context.Items["UserContext"] = userCtx;                       // line 50
    }
    // ...audit log written in finally block
}
```

**File:** `Middleware/RequestLoggingMiddleware.cs` → `BuildUserContext()` (line 68)

Extracts all `UserContext` properties from claims with fallback chains:

| Property | Claim lookup chain | Default |
|---|---|---|
| `UserId` | `"UserId"` → `"sub"` → nameidentifier URI | `"anonymous"` |
| `Role` | `"Role"` → `"role"` → MS role URI | `"viewer"` |
| `DisplayName` | `"UserName"` → `"name"` → `"preferred_username"` | userId |
| `Email` | `"Email"` → `"email"` → `"upn"` | `""` |
| `IpAddress` | `HttpContext.Connection.RemoteIpAddress` | `"unknown"` |

**Category permissions assigned by role** (line 88):

| Role | `PermittedCategories` |
|---|---|
| `"admin"` | `air_quality`, `device_health`, `admin` |
| `"device_admin"` | `air_quality`, `device_health` |
| anything else | `air_quality` only |

---

### 2c — Permitted site resolution

**File:** `Services/UserSiteService.cs` → `GetPermittedSiteIdsAsync(int userId)` (line 33)

Two sequential SQL queries against the AQMS database:

```sql
-- Query 1: resolve the user's permission group (line 47)
SELECT TOP 1 GroupID FROM Users WHERE ID = @userId AND Status = 1

-- Query 2: fetch all station IDs assigned to that group (line 62)
SELECT StationID FROM RoleStations WHERE UserGroupID = @groupId
```

The result list is cached **in-memory for 5 minutes** per user ID (line 24: `CacheDuration = TimeSpan.FromMinutes(5)`).

**Edge cases:**
- No active user row → empty list cached, warning logged
- DB failure → empty list returned (non-fatal, error logged)
- Empty `PermittedSiteIds` list → `UserContext.HasAllSitesAccess = true` (admin/unrestricted)

The completed `UserContext` is stored in `HttpContext.Items["UserContext"]` and retrieved in `ChatController` via `GetUserContext()` (line 270).

---

## Step 3 — Controller Entry and Model Validation

**File:** `Controllers/ChatController.cs`

- `ModelState` is validated against `ChatRequest`. Fails → **400 Bad Request**.
- `UserContext` is retrieved from `HttpContext.Items`. Missing → **401 Unauthorized**.

---

## Step 4 — Scope Validation and Resolution

**File:** `Validators/ScopeValidator.cs`

`ScopeValidator.Resolve(request.Scope, user)` validates the scope bar filters:

| Field | Validation |
|---|---|
| `siteId` | Must be in `user.PermittedSiteIds` (or user has all-sites access) |
| Date range | `from` must not be after `to`; span must not exceed 365 days |
| `pollutants` | Each must be in allowlist: `pm25, pm10, co, co2, no2, o3, so2, ch2o, tvoc, aqi, temperature, humidity` |
| `region` | Must be one of: `Abudhabi`, `AL Ain`, `AlDhafra` (case-insensitive) |
| `sensors` | Alphanumeric + hyphen/underscore only |

Any validation failure → session is created → error response returned immediately with code `INVALID_SCOPE`.

---

## Step 5 — Session Management

**File:** `Services/SessionService.cs`

`GetOrCreateSessionAsync(sessionId, userId)`:
- If `sessionId` provided and belongs to this user → returns existing session
- Otherwise → creates a new `ChatSession` entity (title: "New conversation", `IsActive = true`) in `ChatHistoryDbContext`

`GetConversationHistoryAsync(sessionId)`:
- Retrieves the **last 15 turns** from the DB (user + assistant messages), ordered chronologically
- Cached in-memory for the session duration
- Provides context for follow-up detection and LLM parameter extraction

---

## Step 6 — Query Routing

**File:** `Services/QueryRouterService.cs`

`RouteAsync(question, user, history)`:

1. Load approved templates from `Knowledge/approved-queries.json` (cached **1 hour**)
2. Filter to only templates whose `category` is in `user.PermittedCategories`

### 6a — Follow-Up Detection

If conversation history exists AND the question contains any of these phrases:

> `above`, `previous`, `that list`, `the list`, `same`, `again`, `sort`, `order`, `ascending`, `descending`, `asc`, `desc`, `filter`, `group`, `make it`, `can you`, `change`, `modify`, `update`, `show only`, `now show`, `instead`, `but only`, `also show`, `add to`, `remove from`, `without`, `exclude`, `include only`

→ Returns `NeedsLlmConfirmation = true` with all permitted templates as candidates and the previous template ID for context.

### 6b — Standard Keyword Scoring

For each template, compute the max score across all patterns:
- **Exact substring match** (cleaned pattern found in question) → score `0.95`
- **Token overlap** → `intersection(questionTokens, patternTokens) / len(patternTokens)`

| Score Range | Action |
|---|---|
| `≥ HighConfidenceThreshold` (0.85) | Route directly, go to **Step 7B** |
| `≥ LowConfidenceThreshold` (0.50) | Top 3 candidates → go to **Step 7A** |
| `< LowConfidenceThreshold` | Generate disambiguation prompt → return `CLARIFY` error |

**Disambiguation prompt** (when confidence too low):
- If question has site/device/region keywords → generic "Try asking about: AQI at a site, readings for a device, or AQI for a region."
- Otherwise → specific question: "Are you asking about a site, a device, or a region?"

---

## Step 7 — LLM Interaction

**File:** `Services/AzureAIService.cs`

Azure OpenAI is called via the Azure OpenAI SDK. Auth: API key (dev) or certificate credential (prod). Retries up to 2 times on failure with exponential backoff (2, 4 seconds).

### Step 7A — Template Selection (medium confidence or follow-up)

`SelectTemplateAsync(question, candidates, history, scope, previousTemplateId)`:
- System prompt: instructs LLM to pick best template from candidates and fill parameters
- **History is excluded** if any candidate is `device_last_reading` (prevents wrong device name from history)
- Returns JSON: `{ "selectedQueryId": "...", "parameters": {...}, "confidence": 0.95, "summary": "..." }`
- Validates selected ID is in candidates list; if not, falls back to first candidate

### Step 7B — Parameter Filling (high confidence)

`FillParametersAsync(question, template, history, scope)`:
- System prompt describes the template's parameters and extraction rules:
  - **dates** → ISO `YYYY-MM-DD`
  - **columns** → lowercase snake_case, validated against allowlist
  - **regionName** → maps to canonical: `Abudhabi`, `AL Ain`, `AlDhafra`
  - **sectorName** → maps: `"schools"` → `"Public & Govt-School"`, etc.
  - **parameterName** → maps: `"co2"` → `"CO2"`, `"aqi"` → `"AQI Index"`, etc.
  - **stationName** → exact name as stated by user
  - **deviceName** → exact as stated (pattern: `"BA 0001"`, `"SEI100M 0008"`)
  - **pronoun resolution** → `"this site"`, `"it"` → resolved from history
- **History is excluded** if template has `deviceName` or `stationName` params (current question always wins)
- Returns JSON: `{ "parameters": {...}, "confidence": 0.95 }`

---

## Step 8 — RBAC Enforcement

**File:** `Services/RbacEngine.cs`

### 8a — Template Validation
`ValidateSelection(template, user)`: verifies `template.Category` is in `user.PermittedCategories`. Fails → `UNAUTHORIZED` error.

### 8b — Safe Query Building
`BuildSafeQuery(template, llmParams, user, scope)` — produces the final executable SQL:

**Step 1: Column placeholder substitution**
- For `type = "column"` params, validate against allowlist and sanitize (alphanumeric + underscore, lowercase)
- Substitute `{paramName}` placeholders in SQL text

**Step 2: Typed parameter binding**
- Resolve value from LLM params first, then scope bar (siteId, startDate, endDate, regionName)
- Convert to typed value (`int`, date string, string)
- For missing **optional** params → remove their `WHERE`/`AND` condition from SQL (prevents unbound variable errors)

**Step 3: Site ID injection (core RBAC)**
- `effectiveSiteIds` = `scope.SiteIds` if present, else `user.PermittedSiteIds`
- If `template.SkipSiteFilter = false` AND `effectiveSiteIds` is non-empty AND user doesn't have all-sites access:
  - Detect which table alias to use based on SQL (`pr.StationID`, `pa.StationID`, `pam.StationID`, `s.ID`, or `StationID`)
  - Inject `WHERE {column} IN @siteIds` or `AND {column} IN @siteIds`
  - Bind `siteIds` as `List<int>` (Dapper handles `IN` expansion)

**Step 4: Date range defaults**
- If `@startDate`/`@endDate` referenced but not bound → default to **start of current month** to **today**

**Step 5: Row limit**
- Append `TOP N` if not present
- Limit = `min(viewer → DefaultRowLimit, MaxRowLimit)`

---

## Step 9 — FAQ Short-Circuit

If `template.IsFaq = true`:
- Load `Knowledge/faq.json`
- Call `AzureAIService.AnswerFaqAsync(question, faqContext)` → 1-3 sentence answer
- Save to session and return immediately (**no SQL or API execution**)

---

## Step 10 — SQL vs API Execution

### Path A: SQL Execution

**File:** `Services/SqlExecutorService.cs` + `Validators/SqlValidator.cs`

`SqlValidator.Validate(sql)` checks:
- Must start with `SELECT`
- Must not contain forbidden keywords: `INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, `CREATE`, `EXEC`, `EXECUTE`, `TRUNCATE`, `MERGE`, `GRANT`, `REVOKE`, `XP_`, `SP_`, `OPENROWSET`, `OPENQUERY`, `BULK`, `BACKUP`, `RESTORE`
- All tables must be in the allowlist: `ParameterReadings`, `ParameterAverages`, `ParameterAveragesMonth`, `ParameterAveragesYear`, `DMN_Stations`, `DMN_Devices`, `DMN_Parameters`, `DMN_Flags`, `DeviceCompliance`, `Parameters_Excedence_Values`, `ReportedUnits`, `Sectors`, `SubSectors`, `Regions`, and chat tables
- Max 3 nested `SELECT` statements

Validation failure → `SQL_INVALID` error.

`SqlExecutorService.ExecuteAsync(sql, params)`:
- Opens **read-only** connection to AQMS SQL Server
- Executes via Dapper with bound parameters
- Returns `List<Dictionary<string, object?>>` (NULLs handled explicitly)
- SQL timeout → `SQL_ERROR` error
- Other exceptions → generic `SQL_ERROR` error

### Path B: External API Call

**File:** `Services/ExternalApiService.cs`

`CallAsync(apiCall, llmParams, scope, bearerToken, user)`:

#### Site queries (path contains `{siteId}`):

1. Call `GET /api/AirQuality/GetAllSiteData` (returns **all sites — no server-side filtering**)
2. **RBAC filter**: restrict candidate stations to `user.PermittedSiteIds` only
3. Match user-provided site name against permitted stations using:
   - Name normalisation (e.g., `"abudhabi"` → `"abu dhabi"`, fix DB typos like `"Commerical"`)
   - Tokenization (3+ char words) + **Jaccard similarity** (intersection / union × 100)
   - Hard-exclude stations from wrong region (if query mentions Abu Dhabi, skip Al Ain stations)
   - Exact normalised match → score 1000; otherwise highest Jaccard score wins
4. If `matched == null`:
   - Query `DMN_Stations` DB to check if the site exists but is outside the user's permitted sites → return `"You do not have access to '{site}'"` error
   - If site doesn't exist at all → return `"doesn't match any known site"` with available site list
5. If matched but site ID not in `user.PermittedSiteIds` → `"You do not have access to '{site}'"` error
6. Call `GET /api/AirQuality/GetSiteData?siteId={id}` → fetch device-level data for permitted site
7. Flatten response by `responseShape`

#### Device queries (`responseShape == "device_latest"`):

1. Extract `deviceName` from LLM params
2. Query `DMN_Devices` to resolve name → device ID **and station ID**
3. **RBAC check**: if `stationId` is not in `user.PermittedSiteIds` → return `"You do not have access to device '{name}'"` error
4. Substitute device ID into API path, call `GET /api/AirQuality/GetDeviceLatestData?DeviceID={id}`
5. If user requested a specific parameter → filter rows to that parameter only

#### Other queries:
- Region AQI: default `year` to current year if not provided
- Bearer token forwarded in `Authorization: Bearer` header to AQMS API for its own auth checks
- Unresolved `{placeholder}` in path → error

---

## Step 11 — Natural Language Summary

**File:** `Services/AzureAIService.cs` → `SummarizeResultsAsync()`

- Looks up template-specific **format rule** (e.g., `site_aqi_single`, `device_last_reading`)
- If no specific rule → uses generic formatting rules for COUNT, LIST, SINGLE VALUE, RANKING, TREND, COMPARISON, GENERAL questions
- Sends first **10 rows** of data (to limit token usage for large results)
- LLM generates plain-text summary: no markdown, no bullet points, no repeating column names, no recommendations
- AQI categories: `0-50 Good`, `51-100 Moderate`, `101-150 Unhealthy for Sensitive Groups`, `151-200 Unhealthy`, `201-300 Very Unhealthy`, `301+ Hazardous`

---

## Step 12 — Conversation Persistence

**File:** `Services/SessionService.cs`

**Save user message:**
- `ChatMessage` entity: `role = "user"`, `content = question`
- Appended to in-memory cache + saved to `ChatHistoryDbContext`
- `session.MessageCount++`, `session.LastMessageAt = now`

**Save assistant message:**
- `ChatMessage` entity: `role = "assistant"`
- Fields: `content`, `sql` (API label if API call), `responseType`, `chartType`, `chartDataJson`, `dataSource`, `dateRange`, `modelUsed`, `executionTimeMs`, `tokenCount`

**Auto-title:** If first message in session → title = question truncated to 60 chars.

**Background cleanup:** Sessions older than `HistoryRetentionDays` (30 days) are deleted nightly.

---

## Step 13 — Response Formatting

**File:** `Services/ResponseFormatterService.cs`

`Format(sessionId, messageId, template, llmSummary, rows, ...)` builds `ChatResponse`:

- Enriches AQI numeric values with category labels
- For `device_last_reading`: strips `DeviceName` and `LastMeasured` from table columns (already in summary text)
- For `region_aqi_geographical`: strips `AQICategory` column (already in text)
- Extracts column headers from first row's keys
- For `chart` responses: serializes rows as JSON for `ChartData` field
- Builds `ResponseMetadata`:

| Field | Value |
|---|---|
| `DataSource` | `template.TableUsed` (e.g., `"DMN_Parameters"`, `"AirQuality API"`) |
| `DateRange` | Human-readable label (e.g., `"Jul 1 – Jul 31, 2026"`) |
| `RowCount` | Number of result rows |
| `ExecutionTimeMs` | Query/API execution time |
| `ModelUsed` | Azure OpenAI deployment name |
| `TokenCount` | Total LLM tokens consumed |
| `QueryTemplateId` | Template ID used (e.g., `"device_last_reading"`) |
| `Disclaimer` | Standard AI disclaimer text |

Returns `HTTP 200` with:
```json
{
  "success": true,
  "sessionId": "...",
  "messageId": "...",
  "message": "<plain text summary>",
  "responseType": "table|chart|text",
  "columns": ["Col1", "Col2"],
  "data": [{ "Col1": "...", "Col2": "..." }],
  "chartData": [...],
  "metadata": { ... }
}
```

---

## Step 14 — Error Paths Summary

All errors return **HTTP 200** (not 4xx/5xx) with `Success = false` to allow the frontend to display the message in the chat UI.

| Error Code | Trigger | User-Facing Message Example |
|---|---|---|
| `INVALID_SCOPE` | Site not in permitted list, bad date range, unknown region/pollutant | "You do not have access to site ID 5." |
| `CLARIFY` | Query confidence too low; ambiguous question | "Are you asking about a site, a device, or a region?" |
| `UNAUTHORIZED` | Template category not in user's permitted categories | "Your role does not have access to the 'device_health' query category." |
| `API_ERROR` | Site/device not found, access denied, AQMS API error | "You do not have access to 'Al Ain Industrial Site'." |
| `SQL_INVALID` | Forbidden SQL keyword, non-allowlisted table, non-SELECT | "The query could not be safely constructed." |
| `SQL_ERROR` | Timeout, syntax error, DB connection failure | SQL error message |

All errors are logged:
- Permission denials → `Warning` level
- Technical failures → `Error` level
- All requests → `RequestLoggingMiddleware` audit log (method, path, user ID, IP, status, elapsed ms)

---

## Step 15 — Middleware Pipeline and DI

**File:** `Program.cs`

### Middleware order:
1. Serilog request logging
2. CORS (allowed origins from config)
3. Authentication (JWT Bearer)
4. Authorization (`[Authorize]` attribute on `ChatController`)
5. `RequestLoggingMiddleware` (custom) — builds UserContext, resolves sites, audit log
6. Controllers

### Key service lifetimes:

| Service | Lifetime | Notes |
|---|---|---|
| `IQueryRouterService` | Singleton | Templates cached 1 hour |
| `IAzureAIService` | Singleton | Reuses `ChatClient` connection |
| `IRbacEngine` | Scoped | Stateless |
| `ISqlExecutorService` | Scoped | New connection per request |
| `ISessionService` | Scoped | Uses EF DbContext |
| `IUserSiteService` | Scoped | Permitted sites cached 5 min |
| `IStationResolverService` | Scoped | Station names cached 10 min |
| `IExternalApiService` | Scoped | Uses `IHttpClientFactory` |

### Database connections:
- **Chat history DB** → Entity Framework Core (`ChatHistoryDbContext`), read-write, auto-migrates on startup
- **AQMS Air Quality DB** → Dapper, **read-only** connection string (`SqlExecutorService`, `UserSiteService`, device resolution in `ExternalApiService`)

### HTTP clients:
- `"AqmsApi"` → AQMS Web API, timeout from config (default 30s)

---

*Generated: 2026-08-27 | HAWAQM AI Chat Service*
