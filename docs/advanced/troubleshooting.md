# Troubleshooting

## Rate Limit Behavior

The router uses two-level rate limiting:

- **Credential level** — RPM (requests per minute) and TPM (tokens per minute) per API key
- **Model level** — additional limits for specific (credential + model) pairs

When a limit is reached:

1. Router cascades to the next credential in the priority order for that model
2. Last-resort credentials (`priority: 999`) are tried after every lower tier
3. If every tier is exhausted, returns `429 Too Many Requests`

### Check Current Usage

```bash
curl http://localhost:8080/health | jq '.credentials'
```

## Common HTTP Errors

### 503 Service Unavailable

- Every credential for the model — including the last-resort tier — is banned or rate-limited
- **Fix**: increase RPM/TPM limits, add more credentials, or wait for the next minute reset

### 429 Too Many Requests

- Current credential hit its TPM limit
- No alternative credentials available for the model
- **Fix**: add additional credentials for the same model, or increase TPM limits

### 401 / 403 Unauthorized

- Invalid API key in the request
- Invalid master key configuration
- API key revoked by the provider
- **Fix**: check your config, update the API key

## Last-Resort Behavior

Last-resort credentials (`priority: 999`) are reached when every lower priority tier for
the model is unavailable because it:

- exhausted its RPM/TPM limits, or
- returned a retryable error (`401`, `403`, `429`, `500`, `502`, `503`, `504`), or
- hit a network error or timeout.

### Cascade

1. Request selected from the lowest live priority tier
2. Retryable failure → the router continues the same cascade over the untried credentials
   (crossing provider types), tier by tier, up to `priority: 999`
3. If even the last-resort tier is exhausted → `503 Service Unavailable`

## Debug Logging

Enable debug logging to see detailed request routing:

```yaml
server:
  logging_level: debug
```

```bash
./auto_ai_router -config config.yaml
```
