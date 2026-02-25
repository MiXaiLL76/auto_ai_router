# Troubleshooting

## Rate Limit Behavior

The router uses two-level rate limiting:

- **Credential level** — RPM (requests per minute) and TPM (tokens per minute) per API key
- **Model level** — additional limits for specific (credential + model) pairs

When a limit is reached:

1. Router tries another credential for the same model (round-robin)
2. If no credentials are available, returns `429 Too Many Requests`
3. If fallback proxies are configured, routes to them automatically

### Check Current Usage

```bash
curl http://localhost:8080/health | jq '.credentials'
```

## Common HTTP Errors

### 503 Service Unavailable

- All credentials have exhausted their rate limits
- All fallback proxies are unavailable
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

## Fallback Behavior

Fallback proxies (`is_fallback: true`) activate when:

- Primary credentials exhaust their RPM/TPM limits
- Primary providers return errors (`401`, `403`, `429`, `500`, `502`, `503`, `504`)
- Network errors or timeouts occur

### Fallback Chain

1. Request sent to primary credential
2. Primary fails → try fallback proxy
3. Fallback proxy handles the request with its own credential pool
4. If fallback is also unavailable → `503 Service Unavailable`

## Debug Logging

Enable debug logging to see detailed request routing:

```yaml
server:
  logging_level: debug
```

```bash
./auto_ai_router -config config.yaml
```
