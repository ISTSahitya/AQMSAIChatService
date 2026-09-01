# HAWAQM AI Chat Service Configuration

The service does not store credentials in source control. Supply sensitive settings through IIS environment variables, a managed secret store, or .NET user secrets for local development.

## Required settings

Use double underscores in environment variable names for nested configuration:

- `Database__ChatHistoryConnection`
- `Database__AirQualityConnection`
- `AzureAI__Endpoint`
- `AzureAI__DeploymentName`
- `AzureAI__AuthMethod` (`certificate` in production, `apikey` only for controlled local development)
- `AzureAI__TenantId`, `AzureAI__ClientId`, and `AzureAI__CertificateThumbprint` for certificate authentication
- `AzureAI__ApiKey` only when using API-key authentication
- `Jwt__Secret`
- `Jwt__Issuer`
- `Jwt__Audience`

Production requires `Jwt__Secret` and a complete Azure AI configuration. The development token endpoint has been removed; use the organization's normal identity provider to obtain JWTs.

## IIS

Set the values as system or application-pool environment variables. Do not place passwords, API keys, certificates, or JWT secrets in `web.config` or `appsettings*.json`.

## Local development

Use .NET user secrets from the API project directory:

```powershell
dotnet user-secrets set "AzureAI:ApiKey" "<value>"
dotnet user-secrets set "Database:AirQualityConnection" "<connection string>"
dotnet user-secrets set "Database:ChatHistoryConnection" "<connection string>"
dotnet user-secrets set "Jwt:Secret" "<development-only secret>"
```
