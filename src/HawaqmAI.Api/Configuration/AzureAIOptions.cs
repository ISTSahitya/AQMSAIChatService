namespace HawaqmAI.Api.Configuration;

/// <summary>
/// Configuration for Azure AI Foundry (LLM) connectivity.
/// Bound from appsettings.json "AzureAI" section.
/// </summary>
public sealed class AzureAIOptions
{
    /// <summary>Azure AI Foundry endpoint URL.</summary>
    public string Endpoint { get; set; } = string.Empty;

    /// <summary>Deployed model name (e.g. "gpt-4o-mini").</summary>
    public string DeploymentName { get; set; } = string.Empty;

    /// <summary>Azure OpenAI API version (e.g. "2024-06-01").</summary>
    public string ApiVersion { get; set; } = "2024-06-01";

    /// <summary>Authentication method: "certificate" or "apikey".</summary>
    public string AuthMethod { get; set; } = "certificate";

    /// <summary>Entra tenant ID (used for certificate auth).</summary>
    public string TenantId { get; set; } = string.Empty;

    /// <summary>App registration client ID.</summary>
    public string ClientId { get; set; } = string.Empty;

    /// <summary>Certificate thumbprint in Windows cert store (production only).</summary>
    public string CertificateThumbprint { get; set; } = string.Empty;

    /// <summary>API key fallback for local development. Leave empty in production.</summary>
    public string ApiKey { get; set; } = string.Empty;

    /// <summary>Max tokens to request in each LLM completion.</summary>
    public int MaxTokens { get; set; } = 1000;

    /// <summary>Temperature for LLM responses (lower = more deterministic).</summary>
    public double Temperature { get; set; } = 0.1;
}
