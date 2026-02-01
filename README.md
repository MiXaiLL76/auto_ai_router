# 🤖 Auto AI Router

Высокопроизводительный роутер для проксирования запросов к различным LLM API с автоматической балансировкой нагрузки, контролем лимитов и поддержкой proxy-цепочек.

## ✨ Основные возможности

### Поддерживаемые провайдеры

- **OpenAI** (включая Azure OpenAI)
- **Google Vertex AI**
- **Anthropic Claude**
- **Proxy** - встроенная поддержка цепочек роутеров (автоматическая балансировка между инстансами)

### Контроль и балансировка

- **Round-robin балансировка** с учетом доступности credentials
- **Двухуровневый контроль лимитов**:
  - Credential level: RPM и TPM лимиты
  - Model level: специфичные лимиты для пары (credential + model)
- **Model-aware routing**: автоматический выбор провайдера по доступности модели
- **Fail2ban механизм**: автоматический бан провайдеров при ошибках

### Мониторинг и статистика

- **Prometheus метрики**: детальный мониторинг нагрузки, статуса провайдеров
- **HTTP /health endpoint**: статистика по credentials и models в JSON и HTML форматах
- **Статистика proxy**: автоматическое агрегирование метрик из remote proxy инстансов

### Другое

- **Master key авторизация**: единый ключ для всех клиентов
- **Streaming поддержка**: Server-Sent Events (SSE)
- **Поддержка переменных окружения**: безопасное хранение API ключей
- **Оптимизированное логирование**: сокращение длинных полей (embeddings, base64)

______________________________________________________________________

## 🚀 Быстрый старт

### Требования

- Go 1.21+ или Docker

### Локальная сборка

```bash
# Clone и build
git clone https://github.com/mixaill76/auto_ai_router.git
cd auto_ai_router
go build -o auto_ai_router ./cmd/server/

# Запуск
./auto_ai_router -config config.yaml
```

### Docker

```bash
docker build -t auto-ai-router:latest .
docker run -p 8080:8080 -v $(pwd)/config.yaml:/app/config.yaml auto-ai-router:latest
```

______________________________________________________________________

## 📋 Конфигурация

### Базовый пример (config.yaml)

```yaml
server:
  port: 8080
  master_key: "sk-your-master-key-here"  # Требуется: ключ авторизации
  logging_level: info  # info, debug, error

fail2ban:
  max_attempts: 3
  ban_duration: permanent
  error_codes: [401, 403, 429, 500, 502, 503, 504]

monitoring:
  prometheus_enabled: true

credentials:
  # OpenAI credential
  - name: "openai_main"
    type: "openai"
    api_key: "sk-proj-xxxxxxxxxxxxx"
    base_url: "https://api.openai.com"
    rpm: 100
    tpm: 50000

  # Vertex AI credential
  - name: "vertex_ai"
    type: "vertex-ai"
    project_id: "your-gcp-project"
    location: "global"
    credentials_file: "path/to/service-account.json"
    rpm: 100
    tpm: 50000

  # Proxy credential - fallback при исчерпании основных лимитов
  - name: "proxy_fallback"
    type: "proxy"
    base_url: "http://backup-router.local:8080"  # URL другого auto_ai_router
    api_key: "sk-remote-key"  # Опционально
    is_fallback: true  # Использовать как fallback

# Модели с лимитами (опционально)
models:
  - name: "gpt-4o"
    credential: openai_main
    rpm: 100
    tpm: 50000
  - name: "gemini-2.5-pro"
    credential: vertex_ai
    rpm: 100
    tpm: 50000
```

### Поддерживаемые типы провайдеров

| Провайдер    | Type        | Обязательные поля                                                   |
| ------------ | ----------- | ------------------------------------------------------------------- |
| OpenAI       | `openai`    | `api_key`, `base_url`                                               |
| Anthropic    | `anthropic` | `api_key`, `base_url`                                               |
| Vertex AI    | `vertex-ai` | `project_id`, `location`, `credentials_file` или `credentials_json` |
| Proxy Router | `proxy`     | `base_url`                                                          |

### Proxy как fallback

Proxy credentials позволяют встроить цепочку роутеров:

```yaml
credentials:
  # Основной провайдер
  - name: "openai_main"
    type: "openai"
    api_key: "sk-..."
    base_url: "https://api.openai.com"
    rpm: 100
    tpm: 50000

  # Fallback: другой инстанс auto_ai_router
  - name: "backup_router"
    type: "proxy"
    base_url: "http://10.0.1.50:8080"
    is_fallback: true  # Использовать только когда основные credentials исчерпаны
```

Когда `openai_main` исчерпает свои лимиты, запросы автоматически перенаправляются на `backup_router`.

______________________________________________________________________

## 🔌 API использование

### Запрос к роутеру

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer sk-your-master-key-here" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Health check

```bash
# JSON format
curl http://localhost:8080/health

# HTML dashboard
curl http://localhost:8080/vhealth
```

Health endpoint показывает:

- Статус всех credentials (RPM/TPM usage)
- Статус всех models
- Статистику из подключенных proxy инстансов

______________________________________________________________________

## 📊 Мониторинг

### Prometheus метрики

Доступны на `/metrics`:

- `auto_ai_router_credential_rpm_current` - текущий RPM usage
- `auto_ai_router_credential_tpm_current` - текущий TPM usage
- `auto_ai_router_credential_banned` - статус бана
- `auto_ai_router_requests_total` - всего запросов
- `auto_ai_router_requests_duration_seconds` - время ответа

Примечание: Proxy credentials **не** включаются в Prometheus метрики. Их статистика доступна через `/health` endpoint и синхронизируется из remote `/health` endpoint каждые 30 секунд.

### HTML Dashboard

Откройте http://localhost:8080/vhealth для интерактивного дашбоарда.

______________________________________________________________________

## 🔐 Безопасность

### Переменные окружения

```yaml
credentials:
  - name: "openai"
    type: "openai"
    api_key: "os.environ/OPENAI_API_KEY"  # Читает из env переменной
    base_url: "https://api.openai.com"
```

```bash
export OPENAI_API_KEY="sk-proj-..."
./auto_ai_router -config config.yaml
```

### Master Key

Все запросы требуют Authorization header с master key:

```bash
Authorization: Bearer sk-your-master-key-here
```

______________________________________________________________________

## 📚 Advanced

### Несколько credentials для одной модели

```yaml
models:
  - name: "gpt-4o"
    credential: openai_main
    rpm: 100
  - name: "gpt-4o"
    credential: openai_secondary
    rpm: 100
```

Роутер будет балансировать запросы между обоими credentials.

### Ограничение моделей по провайдерам

По умолчанию все модели доступны для всех credentials. Используйте секцию `models` для привязки.

______________________________________________________________________

## 🛠️ Разработка

### Тестирование

```bash
go test ./...
```

### Логирование

```bash
# Debug mode
./auto_ai_router -config config.yaml  # logging_level: debug
```

______________________________________________________________________

## 📝 Лицензия

MIT License - см. LICENSE файл.

______________________________________________________________________

## 🤝 Контрибьюшены

Приветствуются issue и pull requests!
