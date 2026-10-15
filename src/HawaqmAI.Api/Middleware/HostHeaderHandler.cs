namespace HawaqmAI.Api.Middleware;

/// <summary>
/// A delegating handler that overrides the Host header on outgoing HTTP requests.
/// Used when the AQMS Web API is on the same IIS server as the AI Chat Service:
/// requests go to localhost/127.0.0.1 but IIS requires the Host header to match
/// its configured site binding (the public domain name), otherwise it returns
/// HTTP 400 "Bad Request — Invalid Hostname".
/// </summary>
public sealed class HostHeaderHandler : DelegatingHandler
{
    private readonly string _siteHostName;

    public HostHeaderHandler(string siteHostName)
    {
        _siteHostName = siteHostName;
    }

    protected override Task<HttpResponseMessage> SendAsync(
        HttpRequestMessage request,
        CancellationToken cancellationToken)
    {
        if (!string.IsNullOrWhiteSpace(_siteHostName))
        {
            // Replace whatever Host the HttpClient would send with the IIS binding name
            request.Headers.Host = _siteHostName;
        }

        return base.SendAsync(request, cancellationToken);
    }
}
