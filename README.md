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
- **Автоматический сбор моделей**: от всех провайдеров с объединением списков
- **Оптимизированное логирование**: автоматическое сокращение длинных полей (embeddings, base64)
- **Переменные окружения**: поддержка `os.environ/VAR_NAME` в config.yaml
- **Статические модели**: возможность задать модели и лимиты прямо в config.yaml
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
  master_key: "sk-your-secret-key"  # Ключ для клиентов
  logging_level: info                # info, debug, error
  replace_v1_models: true            # Включить model-aware routing
  request_timeout: 5m                # Таймаут запроса (или -1 для отключения)
  default_models_rpm: 50             # RPM по умолчанию для моделей (или -1 для отключения)

credentials:
  - name: "openai_main"
    api_key: "sk-proj-..."           # Или os.environ/OPENAI_API_KEY
    base_url: "https://api.openai.com"
    rpm: 100                         # Requests per minute для credential (или -1 для отключения)
    tpm: 50000                       # Tokens per minute для credential (или 0/-1 для отключения)

  - name: "openai_backup"
    api_key: "os.environ/BACKUP_KEY" # Поддержка переменных окружения
    base_url: "https://api.another.com"
    rpm: 50
    tpm: 0                           # Без лимита токенов

fail2ban:
  max_attempts: 3
  ban_duration: permanent            # или "5m", "1h"
  error_codes: [401, 403, 429, 500, 502, 503, 504]

# Опционально: статические модели (объединяются с models.yaml)
models:
  - name: gpt-4o-mini
    credential: openai_backup # Опционально можно указать, что эта модель именно только к этим credentials
    rpm: 60
    tpm: 30000
```

**models.yaml** (создается автоматически или вручную):

```yaml
models:
  - name: gpt-4o-mini
    rpm: 60                          # Дополнительное ограничение для модели (применяется к каждому credential)
    tpm: 30000                       # Ограничение токенов для модели (применяется к каждому credential)

  - name: text-embedding-3-small
    rpm: 100
    tpm: -1                          # Без лимита токенов для embeddings

  - name: gpt-3.5-turbo
    rpm: -1                          # Без лимитов RPM
    tpm: 0                           # Без лимитов TPM
```

**Примечания:**

- Модели из config.yaml имеют приоритет над models.yaml
- При `replace_v1_models: false` модели берутся из models.yaml (если файл существует) или из config.yaml

### Логика работы лимитов (Комбинированный подход)

**Уровень 1: Credential (обязательные общие лимиты)**

- Каждый credential имеет общий RPM/TPM лимит для ВСЕХ моделей вместе
- Например: `openai_main` с `rpm: 100, tpm: 50000` - не более 100 запросов и 50000 токенов в минуту суммарно

**Уровень 2: Model (опциональные ограничения)**

- Лимиты из `models.yaml` применяются к КАЖДОМУ credential независимо
- Например: `gpt-4o-mini` с `rpm: 60` означает не более 60 RPM для этой модели через КАЖДЫЙ credential

**Итоговый лимит = MIN(credential_limit, model_limit)**

Пример:

- `openai_main`: rpm=100, tpm=50000
- `gpt-4o-mini` в models.yaml: rpm=60, tpm=30000
- Итог для `(openai_main, gpt-4o-mini)`: **60 RPM, 30000 TPM**
- Итог для `(openai_main, text-embedding-3-small)` без лимитов в models.yaml: **100 RPM, 50000 TPM**

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
- `GET /v1/models` - Объединенный список моделей от всех провайдеров
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
