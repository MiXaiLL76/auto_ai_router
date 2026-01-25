# Auto AI Router

Высокопроизводительный роутер для проксирования запросов к OpenAI-подобным API с автоматической балансировкой нагрузки, контролем лимитов и умным выбором провайдеров.

## Основные возможности

- **Round-robin балансировка** с учетом доступности credentials
- **Двухуровневый RPM контроль**: на уровне провайдеров и моделей
- **Model-aware routing**: автоматический выбор провайдера по доступности модели
- **Fail2ban механизм**: автоматический бан неработающих провайдеров
- **Master key авторизация**: единый ключ для всех клиентов
- **Streaming поддержка**: Server-Sent Events (SSE)
- **Prometheus метрики**: мониторинг нагрузки и статуса
- **Автоматический сбор моделей**: от всех провайдеров с объединением списков

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
  master_key: "sk-your-secret-key"  # Ключ для клиентов
  logging_level: info                # info, debug, error
  replace_v1_models: true            # Включить model-aware routing
  request_timeout: 5m                # Таймаут запроса (или -1 для отключения)
  default_models_rpm: 50             # RPM по умолчанию для моделей (или -1 для отключения)

credentials:
  - name: "openai_main"
    api_key: "sk-proj-..."
    base_url: "https://api.openai.com"
    rpm: 60                          # Лимит провайдера (или -1 для отключения)

  - name: "openai_backup"
    api_key: "sk-proj-..."
    base_url: "https://api.another.com"
    rpm: -1                          # Без лимитов

fail2ban:
  max_attempts: 3
  ban_duration: permanent            # или "5m", "1h"
  error_codes: [401, 403, 429, 500, 502, 503, 504]
```

**models.yaml** (создается автоматически):
```yaml
models:
  - name: gpt-4o
    rpm: 50                          # Индивидуальный лимит модели (или -1 для отключения)

  - name: gpt-4o-mini
    rpm: 100

  - name: gpt-3.5-turbo
    rpm: -1                          # Без лимитов
```

**Отключение лимитов:**
Для параметров `request_timeout`, `default_models_rpm`, `rpm` (для credentials и models) можно указать значение `-1`, чтобы отключить соответствующий лимит:
- `request_timeout: -1` - бесконечный таймаут запроса
- `default_models_rpm: -1` - без лимита RPM по умолчанию для моделей
- `rpm: -1` - без лимита RPM для конкретного провайдера или модели


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
    api_key="sk-your-secret-key",      # master_key из config.yaml
    base_url="http://localhost:8080/v1"
)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## Как это работает

1. **Клиент отправляет запрос** с `master_key` в `Authorization` header
2. **Роутер проверяет master_key** и извлекает модель из запроса
3. **Round-robin выбирает credential** с учетом:
   - Доступности модели у провайдера
   - Credential RPM лимита
   - Model RPM лимита
   - Fail2ban статуса
4. **Подменяет master_key** на реальный API ключ провайдера
5. **Проксирует запрос** к upstream API
6. **Возвращает ответ** клиенту (с поддержкой streaming)

## Endpoints

- `POST /v1/chat/completions` - Chat completions (с model-aware routing)
- `GET /v1/models` - Объединенный список моделей от всех провайдеров
- `GET /health` - Health check
- `GET /metrics` - Prometheus метрики
