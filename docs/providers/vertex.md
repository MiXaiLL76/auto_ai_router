# Vertex AI

## Configuration

### With Service Account File

```yaml
credentials:
  - name: "vertex_ai"
    type: "vertex-ai"
    project_id: "your-gcp-project"
    location: "global"
    credentials_file: "path/to/service-account.json"
    rpm: 100
    tpm: 50000
```

### With Credentials JSON (environment variable)

```yaml
credentials:
  - name: "vertex_ai"
    type: "vertex-ai"
    project_id: "os.environ/GCP_PROJECT_ID"
    location: "us-central1"
    credentials_json: "os.environ/VERTEX_CREDENTIALS"
    rpm: 100
    tpm: 50000
```

## Required Fields

| Field              | Description                                                |
| ------------------ | ---------------------------------------------------------- |
| `project_id`       | GCP project ID                                             |
| `location`         | GCP region (e.g., `global`, `us-central1`, `europe-west1`) |
| `credentials_file` | Path to service account JSON file                          |
| `credentials_json` | **Or** service account JSON content as a string            |

!!! note
Provide either `credentials_file` or `credentials_json`, not both.

## Authentication

Vertex AI uses OAuth2 tokens obtained from the service account. The router automatically manages token refresh.

## Multiple Credentials

You can configure multiple Vertex AI credentials for load balancing:

```yaml
credentials:
  - name: "vertex_project_a"
    type: "vertex-ai"
    project_id: "project-a"
    location: "global"
    credentials_file: "sa-a.json"
    rpm: 100
    tpm: 50000

  - name: "vertex_project_b"
    type: "vertex-ai"
    project_id: "project-b"
    location: "global"
    credentials_file: "sa-b.json"
    rpm: 100
    tpm: 50000
```

Requests are distributed across credentials using round-robin. See [Load Balancing](../advanced/balancing.md).
