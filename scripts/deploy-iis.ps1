#Requires -RunAsAdministrator
<#
.SYNOPSIS
    Deploy HAWAQM AI Chat Service to IIS on the DOH Windows Server.

.DESCRIPTION
    Creates a dedicated IIS application pool and web application for the
    HAWAQM AI service. Deploys the published .NET 8 application.

.PREREQUISITES
    - .NET 8 Hosting Bundle installed (https://dotnet.microsoft.com/download/dotnet/8.0)
    - IIS with ASP.NET Core Module V2 installed
    - Run as Administrator

.EXAMPLE
    .\deploy-iis.ps1 -PublishPath "C:\builds\hawaqm-ai\publish" -SiteName "Default Web Site"
#>

param(
    [Parameter(Mandatory = $false)]
    [string]$PublishPath = "C:\inetpub\wwwroot\HawaqmAI",

    [Parameter(Mandatory = $false)]
    [string]$SiteName = "Default Web Site",

    [Parameter(Mandatory = $false)]
    [string]$AppName = "HawaqmAI",

    [Parameter(Mandatory = $false)]
    [string]$AppPoolName = "HawaqmAI_Pool",

    [Parameter(Mandatory = $false)]
    [string]$SourcePath = ""  # Path to dotnet publish output. Leave empty to build locally.
)

$ErrorActionPreference = "Stop"
Import-Module WebAdministration

Write-Host "=== HAWAQM AI IIS Deployment ===" -ForegroundColor Cyan

# ── 1. Verify .NET 8 Hosting Bundle ────────────────────────────────────────
Write-Host "Checking .NET 8 Hosting Bundle..." -ForegroundColor Yellow
$dotnetVersion = dotnet --version 2>$null
if (-not $dotnetVersion -or -not $dotnetVersion.StartsWith("8.")) {
    Write-Error ".NET 8 is required. Install the Hosting Bundle from https://dotnet.microsoft.com/download/dotnet/8.0"
}
Write-Host "  .NET version: $dotnetVersion" -ForegroundColor Green

# ── 2. Publish the application ──────────────────────────────────────────────
if ($SourcePath -ne "") {
    Write-Host "Copying published files from $SourcePath to $PublishPath..." -ForegroundColor Yellow
    if (-not (Test-Path $PublishPath)) {
        New-Item -ItemType Directory -Path $PublishPath -Force | Out-Null
    }
    Copy-Item -Path "$SourcePath\*" -Destination $PublishPath -Recurse -Force
    Write-Host "  Files copied." -ForegroundColor Green
} else {
    Write-Host "No SourcePath specified. Assuming files already at $PublishPath." -ForegroundColor Yellow
}

# ── 3. Create Application Pool ──────────────────────────────────────────────
Write-Host "Configuring IIS Application Pool: $AppPoolName..." -ForegroundColor Yellow

if (Test-Path "IIS:\AppPools\$AppPoolName") {
    Write-Host "  App pool already exists — stopping it..." -ForegroundColor Yellow
    Stop-WebAppPool -Name $AppPoolName -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
} else {
    New-WebAppPool -Name $AppPoolName | Out-Null
    Write-Host "  App pool created." -ForegroundColor Green
}

# .NET 8 ASP.NET Core runs without classic .NET CLR
Set-ItemProperty "IIS:\AppPools\$AppPoolName" -Name "managedRuntimeVersion" -Value ""
Set-ItemProperty "IIS:\AppPools\$AppPoolName" -Name "managedPipelineMode" -Value "Integrated"
Set-ItemProperty "IIS:\AppPools\$AppPoolName" -Name "startMode" -Value "AlwaysRunning"
Set-ItemProperty "IIS:\AppPools\$AppPoolName" -Name "autoStart" -Value $true

# Set process model — run as ApplicationPoolIdentity (least privilege)
Set-ItemProperty "IIS:\AppPools\$AppPoolName" -Name "processModel.identityType" -Value "ApplicationPoolIdentity"
Set-ItemProperty "IIS:\AppPools\$AppPoolName" -Name "processModel.idleTimeout" -Value "00:00:00"  # Never idle
Set-ItemProperty "IIS:\AppPools\$AppPoolName" -Name "recycling.periodicRestart.time" -Value "00:00:00"  # No periodic restart

Write-Host "  App pool configured." -ForegroundColor Green

# ── 4. Create or update IIS Application ────────────────────────────────────
Write-Host "Configuring IIS Application: $AppName under '$SiteName'..." -ForegroundColor Yellow

$appPath = "IIS:\Sites\$SiteName\$AppName"
if (Test-Path $appPath) {
    Write-Host "  Application already exists — updating..." -ForegroundColor Yellow
    Set-ItemProperty $appPath -Name "physicalPath" -Value $PublishPath
    Set-ItemProperty $appPath -Name "applicationPool" -Value $AppPoolName
} else {
    New-WebApplication `
        -Name $AppName `
        -Site $SiteName `
        -PhysicalPath $PublishPath `
        -ApplicationPool $AppPoolName | Out-Null
    Write-Host "  Application created." -ForegroundColor Green
}

# ── 5. Set environment variable ─────────────────────────────────────────────
Write-Host "Setting ASPNETCORE_ENVIRONMENT=Production..." -ForegroundColor Yellow

$envVarPath = "system.webServer/aspNetCore/environmentVariables/environmentVariable"
try {
    Add-WebConfigurationProperty `
        -PSPath $appPath `
        -Filter "system.webServer/aspNetCore/environmentVariables" `
        -Name "." `
        -Value @{name = "ASPNETCORE_ENVIRONMENT"; value = "Production"} `
        -ErrorAction SilentlyContinue
} catch {
    # Variable may already be set in web.config — that's fine
}

# ── 6. Start the application pool ──────────────────────────────────────────
Write-Host "Starting application pool..." -ForegroundColor Yellow
Start-WebAppPool -Name $AppPoolName
Start-Sleep -Seconds 3

$poolState = (Get-WebAppPoolState -Name $AppPoolName).Value
Write-Host "  App pool state: $poolState" -ForegroundColor $(if ($poolState -eq "Started") { "Green" } else { "Red" })

# ── 7. Verify deployment ────────────────────────────────────────────────────
Write-Host ""
Write-Host "=== Deployment Summary ===" -ForegroundColor Cyan
Write-Host "  App Pool:    $AppPoolName" -ForegroundColor White
Write-Host "  IIS App:     $SiteName/$AppName" -ForegroundColor White
Write-Host "  Physical:    $PublishPath" -ForegroundColor White
Write-Host "  Status:      $poolState" -ForegroundColor White
Write-Host ""
Write-Host "Health check URL: http://localhost/$AppName/api/health" -ForegroundColor Cyan
Write-Host ""
Write-Host "IMPORTANT: Set sensitive configuration as environment variables" -ForegroundColor Red
Write-Host "  [System.Environment]::SetEnvironmentVariable('Database__AirQualityConnection', '...', 'Machine')" -ForegroundColor Yellow
Write-Host "  [System.Environment]::SetEnvironmentVariable('Database__ChatHistoryConnection', '...', 'Machine')" -ForegroundColor Yellow
Write-Host "  [System.Environment]::SetEnvironmentVariable('Jwt__Secret', '...', 'Machine')" -ForegroundColor Yellow
Write-Host "  Then restart the app pool: Restart-WebAppPool -Name $AppPoolName" -ForegroundColor Yellow
