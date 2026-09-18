# Video generation

AIR exposes an asynchronous Runway video API.

## Public API

| Method | Path |
| --- | --- |
| POST | `/v1/videos` |
| GET | `/v1/videos/{id}` |
| DELETE | `/v1/videos/{id}` |
| GET and HEAD | `/v1/videos/{id}/content` |
| POST | `/v1/media/uploads` |
| PUT | `/v1/media/uploads/{id}/object` |
| POST | `/v1/media/uploads/{id}/complete` |

Supported models

- `runway/gen4.5`
- `runway/gen4_turbo`

`runway/gen4.5` supports text and image input.
`runway/gen4_turbo` requires image input.
Duration is an integer from 2 through 10 seconds.
Text mode accepts `1280:720` and `720:1280`.
Image mode also accepts `1104:832`, `832:1104`, `960:960` and `1584:672`.
Uploaded images are converted to data URIs no larger than 5,000,000 bytes.

## Runtime

The public AIR process authenticates every request through the LiteLLM database.
Organization policies control model admission and provide the exact USD price per output second.

PostgreSQL stores jobs, immutable pricing snapshots, leases and uploads.
Every AIR replica runs one lease based worker by default.
The worker submits and polls Runway, copies completed artifacts to S3 compatible storage and commits one idempotent SpendLog row.

The stable video job identifier is also the SpendLog request identifier.
A settlement replay therefore cannot create a second charge.

## Configuration

```yaml
video:
  enabled: true
  runway_api_key: os.environ/AIR_VIDEO_RUNWAY_API_KEY
  runway_base_url: https://api.dev.runwayml.com
  runway_api_version: 2024-11-06
  s3_endpoint: https://s3.twcstorage.ru
  s3_region: ru-1
  s3_bucket: a73def9f-143e-4c0e-a7c8-eb36cae1a4be
  s3_access_key: os.environ/AIR_VIDEO_S3_ACCESS_KEY
  s3_secret_key: os.environ/AIR_VIDEO_S3_SECRET_KEY
  s3_prefix: air-video/
  upload_signing_key: os.environ/AIR_VIDEO_UPLOAD_SIGNING_KEY
  poll_interval: 5s
  lease_ttl: 5m
  upload_ttl: 15m
  max_artifact_bytes: 1073741824
  worker_concurrency: 1
  models:
    - name: runway/gen4.5
      provider_model: gen4.5
    - name: runway/gen4_turbo
      provider_model: gen4_turbo
```

Video requires LiteLLM SpendLogs writes.
Startup fails when video is enabled without PostgreSQL, object storage, Runway credentials or synchronous spend settlement.

Budgeted and rate limited keys are rejected until durable asynchronous reservations are enabled.
The current Cloud.ru video organizations have no such limits.

## Billing

The organization price profile must define `output_cost_per_video_per_second` for every admitted video model.
The quote is fixed when the job is created.
Later price changes do not alter an active job.

The worker writes SpendLogs only after the provider result has been copied to owned storage.
Failed and cancelled jobs have no spend.

## Security

Client supplied organization headers are ignored.
All job and upload reads use the authenticated organization identifier.

Uploads use HMAC signed URLs, strict size checks, SHA-256 validation and container signatures.
Provider artifacts require HTTPS and public network addresses.
Redirects receive the same network validation.
Artifact size is bounded before it is copied to object storage.
