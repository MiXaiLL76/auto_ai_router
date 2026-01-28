# Auto AI Router

Высокопроизводительный роутер для проксирования запросов к OpenAI-подобным API с автоматической балансировкой нагрузки, контролем лимитов и умным выбором провайдеров.

## Основные возможности

- **Round-robin балансировка** с учетом доступности credentials
- **Комбинированный контроль лимитов**:
  - **RPM** (Requests Per Minute) - на уровне credentials и моделей
  - **TPM** (Tokens Per Minute) - на уровне credentials и моделей
  - Независимые лимиты для каждой пары (credential, model)
  - Метрики RPM/TPM для каждой пары (credential, model) в Prometheus
- **Model-aware routing**: автоматический выбор провайдера по доступности модели
- **Fail2ban механизм**: автоматический бан неработающих провайдеров
- **Master key авторизация**: единый ключ для всех клиентов
- **Streaming поддержка**: Server-Sent Events (SSE)
- **Prometheus метрики**: мониторинг нагрузки, статуса и использования токенов
- **Статические модели**: модели и лимиты задаются прямо в config.yaml
- **Поддержка разных провайдеров**: OpenAI, Azure OpenAI, Vertex AI
- **Оптимизированное логирование**: автоматическое сокращение длинных полей (embeddings, base64)
- **Переменные окружения**: поддержка `os.environ/VAR_NAME` в config.yaml
- **CI/CD проверки**: автоматические тесты, lint и проверка качества кода

## Быстрый старт

### 1. Сборка

#### Локальная сборка

```bash
make build
# или
go build -o auto_ai_router ./cmd/server/
```

#### Docker

```bash
# Сборка образа
make docker-build

# Или напрямую
docker build -t auto-ai-router:latest .
```

### 2. Настройка конфигурации

**config.yaml:**

```yaml
server:
  port: 8080
  max_body_size_mb: 100              # Максимальный размер тела запроса в MB
  master_key: "sk-your-secret-key"   # Ключ для клиентов
  logging_level: info                 # info, debug, error
  replace_v1_models: false            # Включить model-aware routing
  request_timeout: 90s                # Таймаут запроса (или -1 для отключения)
  default_models_rpm: 50              # RPM по умолчанию для моделей (или -1 для отключения)

credentials:
  - name: "openai_main"
    type: "openai"                    # Тип провайдера: openai, vertex-ai
    api_key: "sk-proj-..."            # Или os.environ/OPENAI_API_KEY
    base_url: "https://api.openai.com"
    rpm: 100                          # Requests per minute для credential (или -1 для отключения)
    tpm: 50000                        # Tokens per minute для credential (или 0/-1 для отключения)

  - name: "vertex_ai"
    type: "vertex-ai"
    project_id: "your-project-id"
    location: "global"
    credentials_file: "path/to/service-account.json"
    rpm: 100
    tpm: 50000

fail2ban:
  max_attempts: 3
  ban_duration: permanent             # или "5m", "1h"
  error_codes: [401, 403, 429, 500, 502, 503, 504]

monitoring:
  prometheus_enabled: true
  health_check_path: "/health"

# Модели с привязкой к конкретным credentials
models:
  - name: gpt-4o-mini
    credential: openai_main           # Обязательно указать credential
    rpm: 60
    tpm: 30000
  - name: gemini-2.5-pro
    credential: vertex_ai
    rpm: 100
    tpm: 50000
```

**Поддерживаемые типы провайдеров:**

- `openai` - OpenAI API и Azure OpenAI
- `vertex-ai` - Google Vertex AI

**Примечания:**

- Все модели задаются в секции `models` в config.yaml
- Каждая модель должна быть привязана к конкретному credential
- При `replace_v1_models: false` используется статический список моделей из конфига

### Логика работы лимитов

**Уровень 1: Credential (общие лимиты)**

- Каждый credential имеет общий RPM/TPM лимит для ВСЕХ моделей вместе
- Например: `openai_main` с `rpm: 100, tpm: 50000` - не более 100 запросов и 50000 токенов в минуту суммарно

**Уровень 2: Model (индивидуальные лимиты)**

- Каждая модель имеет свои RPM/TPM лимиты, привязанные к конкретному credential
- Например: `gpt-4o-mini` с `credential: openai_main, rpm: 60` - не более 60 RPM для этой модели через указанный credential

**Итоговый лимит = MIN(credential_limit, model_limit)**

Пример:

- `openai_main`: rpm=100, tpm=50000
- `gpt-4o-mini` с credential=openai_main: rpm=60, tpm=30000
- Итог для `(openai_main, gpt-4o-mini)`: **60 RPM, 30000 TPM**

**Отключение лимитов:**
Для отключения лимита используйте значение `-1` или `0` (для TPM):

- `request_timeout: -1` - бесконечный таймаут запроса
- `default_models_rpm: -1` - без лимита RPM по умолчанию для моделей
- `rpm: -1` - без лимита RPM для credential или модели
- `tpm: 0` или `tpm: -1` - без лимита TPM для credential или модели

### 3. Запуск

#### Локальный запуск

```bash
./auto_ai_router -config config.yaml
```

#### Docker Compose

```bash
# Запуск в фоне
make docker-run

# Или напрямую
docker-compose up -d

# Просмотр логов
docker-compose logs -f

# Остановка
make docker-stop
```

### 4. Использование

```python
from openai import OpenAI

client = OpenAI(
    api_key="sk-your-secret-key",  # master_key из config.yaml
    base_url="http://localhost:8080/v1",
)

response = client.chat.completions.create(
    model="gpt-4o-mini", messages=[{"role": "user", "content": "Hello!"}]
)
```

## Как это работает

1. **Клиент отправляет запрос** с `master_key` в `Authorization` header
2. **Роутер проверяет master_key** и извлекает модель из запроса
3. **Round-robin выбирает credential** с учетом:
   - Доступности модели у провайдера
   - Credential RPM лимита
   - Credential TPM лимита (проверка текущего использования)
   - Model RPM лимита для пары (credential, model)
   - Model TPM лимита для пары (credential, model)
   - Fail2ban статуса
4. **Подменяет master_key** на реальный API ключ провайдера
5. **Проксирует запрос** к upstream API
6. **Получает ответ** и извлекает `usage.total_tokens`
7. **Обновляет счетчики TPM** для credential и модели
8. **Возвращает ответ** клиенту (с поддержкой streaming)

## Endpoints

- `POST /v1/chat/completions` - Chat completions (с model-aware routing)
- `GET /v1/models` - Список моделей из конфигурации
- `GET /health` - Health check (JSON)
- `GET /vhealth` - Визуальный дашборд здоровья системы с метриками по credentials и моделям (сортировка по TPM по умолчанию)
- `GET /metrics` - Prometheus метрики

## Метрики Prometheus

Доступные метрики:

- `auto_ai_router_credential_rpm_current` - текущий RPM для credential
- `auto_ai_router_credential_tpm_current` - текущий TPM для credential
- `auto_ai_router_credential_banned` - статус бана credential (0/1)
- `auto_ai_router_model_rpm_current` - текущий RPM для пары (credential, model)
- `auto_ai_router_model_tpm_current` - текущий TPM для пары (credential, model)
