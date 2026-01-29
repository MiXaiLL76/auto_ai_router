# Auto AI Router

Высокопроизводительный роутер для проксирования запросов к различным LLM API (OpenAI, Anthropic, Vertex AI) с автоматической балансировкой нагрузки, контролем лимитов и умным выбором провайдеров.

## Основные возможности

- **Поддержка нескольких провайдеров**: OpenAI, Anthropic, Google Vertex AI
- **Round-robin балансировка** с учетом доступности credentials
- **Двухуровневый контроль лимитов**:
  - **Уровень 1**: Лимиты на credential (общие для всех моделей)
  - **Уровень 2**: Лимиты на модель (специфичные для пары credential + model)
  - **RPM** (Requests Per Minute) и **TPM** (Tokens Per Minute)
  - Итоговый лимит = MIN(credential_limit, model_limit)
- **Model-aware routing**: автоматический выбор провайдера по доступности модели
- **Fail2ban механизм**: автоматический бан провайдеров при ошибках
- **Master key авторизация**: единый ключ для всех клиентов
- **Streaming поддержка**: Server-Sent Events (SSE)
- **Prometheus метрики**: детальный мониторинг нагрузки, статуса и использования токенов
- **Статические модели**: модели и лимиты задаются в config.yaml
- **Поддержка переменных окружения**: `os.environ/VAR_NAME` в config.yaml для защиты чувствительных данных
- **Оптимизированное логирование**: автоматическое сокращение длинных полей (embeddings, base64)

## Быстрый старт

### 1. Сборка

#### Требования

- Go 1.21 или выше
- (Опционально) Docker для контейнеризации

#### Локальная сборка

```bash
# Используя Makefile
make build

# Или напрямую
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
  port: 8080                          # Порт сервера
  max_body_size_mb: 100               # Максимальный размер тела запроса в MB
  master_key: "sk-your-secret-key"    # Мастер-ключ для клиентов (требуется)
  logging_level: info                 # Уровень логирования: info, debug, error
  request_timeout: 5m                 # Таймаут запроса (или -1 для отключения)
  default_models_rpm: 50              # RPM по умолчанию для моделей (или -1 без лимита)

# Fail2ban - блокировка провайдеров при ошибках
fail2ban:
  max_attempts: 3                     # Количество ошибок перед баном
  ban_duration: permanent             # Длительность бана: "permanent", "5m", "1h" и т.д.
  error_codes: [401, 403, 429, 500, 502, 503, 504]  # HTTP коды, считаемые ошибками

# Учетные данные провайдеров
credentials:
  # OpenAI / Azure OpenAI
  - name: "openai_main"
    type: "openai"
    api_key: "os.environ/OPENAI_API_KEY"  # Или прямое значение
    base_url: "https://api.openai.com"    # Для Azure: https://<resource>.openai.azure.com/openai
    rpm: 100                               # Requests per minute лимит
    tpm: 50000                             # Tokens per minute лимит (-1 или 0 для без лимита)

  # Anthropic
  - name: "anthropic_main"
    type: "anthropic"
    api_key: "os.environ/ANTHROPIC_API_KEY"
    base_url: "https://api.anthropic.com"
    rpm: 100
    tpm: 50000

  # Google Vertex AI
  - name: "vertex_ai"
    type: "vertex-ai"
    project_id: "your-gcp-project-id"
    location: "us-central1"
    credentials_file: "path/to/service-account.json"  # Или credentials_json для JSON строки
    rpm: 100
    tpm: 50000

# Мониторинг
monitoring:
  prometheus_enabled: true
  health_check_path: "/health"

# Модели с привязкой к credentials
models:
  - name: gpt-4o-mini
    credential: openai_main
    rpm: 60                           # Лимит для этой модели через этот credential
    tpm: 30000

  - name: claude-opus-4-1
    credential: anthropic_main
    rpm: 50
    tpm: 25000

  - name: gemini-2.5-pro
    credential: vertex_ai
    rpm: 100
    tpm: 50000
```

**Поддерживаемые типы провайдеров:**

| Тип       | Поле   | Значение    |
| --------- | ------ | ----------- |
| OpenAI    | `type` | `openai`    |
| Anthropic | `type` | `anthropic` |
| Vertex AI | `type` | `vertex-ai` |

**Vertex AI специфические поля:**

- `project_id` - ID проекта GCP (требуется)
- `location` - регион (требуется), например: `us-central1`, `global`
- `credentials_file` - путь к service account JSON файлу
- `credentials_json` - JSON строка с credentials (альтернатива credentials_file)

**Примечания о конфигурации:**

- Все модели задаются в секции `models` и должны иметь привязку к `credential`
- Используйте `os.environ/VAR_NAME` для защиты чувствительных данных (API ключи)
- Каждый credential требует минимум: `name`, `type`, `api_key`
- Для OpenAI/Anthropic также требуется `base_url`
- Для Vertex AI требуются: `project_id` и `location`

## Advanced Topics

### Маршрутизация с несколькими credentials для одной модели

Вы можете использовать одну модель с несколькими credentials, создав несколько записей в секции `models`:

```yaml
credentials:
  - name: "openai_main"
    rpm: 100
  - name: "openai_backup"
    rpm: 100

models:
  - name: gpt-4o-mini
    credential: openai_main
    rpm: 50
  - name: gpt-4o-mini
    credential: openai_backup
    rpm: 50
```

Роутер будет автоматически балансировать нагрузку между обоими credentials, выбирая доступный с наименьшей нагрузкой.

### Мониторинг в Prometheus + Grafana

Пример dashboard для мониторинга:

```promql
# TPM использование по models
auto_ai_router_model_tpm_current / auto_ai_router_model_tpm_limit * 100

# Процент RPM использования
auto_ai_router_credential_rpm_current / auto_ai_router_credential_rpm_limit * 100

# Alert: Credential будет забанен за 5 минут
auto_ai_router_credential_rpm_current / auto_ai_router_credential_rpm_limit > 0.9

# Список забаненных credentials
auto_ai_router_credential_banned == 1
```

### Azure OpenAI интеграция

Для Azure OpenAI используйте тот же тип `openai` с правильным `base_url`:

```yaml
credentials:
  - name: "azure_openai"
    type: "openai"
    api_key: "os.environ/AZURE_OPENAI_KEY"
    base_url: "https://<your-resource>.openai.azure.com/openai"
    rpm: 100
```

Роутер автоматически добавит `/v1` к base_url, поэтому не нужно включать его явно.

### Custom Vertex AI credentials

Для Vertex AI есть несколько способов аутентификации:

**Способ 1: Service Account файл**

```yaml
credentials:
  - name: "vertex_sa_file"
    type: "vertex-ai"
    project_id: "my-project"
    location: "us-central1"
    credentials_file: "/path/to/service-account.json"
```

**Способ 2: JSON строка (для Docker)**

```yaml
credentials:
  - name: "vertex_sa_json"
    type: "vertex-ai"
    project_id: "os.environ/GCP_PROJECT_ID"
    location: "us-central1"
    credentials_json: "os.environ/GCP_SERVICE_ACCOUNT_JSON"
```

**Способ 3: API ключ (Express Mode)**

```yaml
credentials:
  - name: "vertex_api_key"
    type: "vertex-ai"
    project_id: "my-project"
    location: "us-central1"
    api_key: "os.environ/VERTEX_API_KEY"
```

## Validation и Обработка ошибок

### Валидация конфигурации

Роутер проверяет конфигурацию при запуске:

- ✓ master_key требуется и не может быть пустой
- ✓ Каждый credential имеет обязательные поля для своего типа
- ✓ Каждая модель привязана к существующему credential
- ✓ Port валиден (1-65535)
- ✓ Логирование имеет правильный уровень (info, debug, error)
- ✓ Лимиты корректны (-1 для отключения, положительные для активации)

### Fail2ban система

При обнаружении ошибок credential автоматически банится:

```yaml
fail2ban:
  max_attempts: 3                    # Количество ошибок перед баном
  ban_duration: permanent            # Длительность бана
  error_codes: [401, 403, 429, 500, 502, 503, 504]  # Какие ошибки считать критическими
```

**Поведение:**

- При получении `max_attempts` ошибок из списка `error_codes`, credential помечается как забанен
- Забаненный credential не будет выбран для маршрутизации запросов
- Со временем бан может быть снят (зависит от `ban_duration`)
- Использование `permanent` означает, что credential остается забанен до перезагрузки

## Troubleshooting

### "All credentials are banned or unavailable"

**Возможные причины:**

1. Все credentials получили слишком много ошибок
2. Все RPM/TPM лимиты исчерпаны
3. Запрашиваемая модель не поддерживается ни одним credential

**Решение:**

- Проверьте `/vhealth` endpoint для статуса credentials
- Убедитесь, что API ключи корректны в конфигурации
- Проверьте лимиты (RPM/TPM) - возможно они слишком строгие
- Валидируйте, что модель поддерживается хотя бы одним провайдером

### "Invalid master_key"

**Возможные причины:**

1. master_key в запросе не совпадает с config.yaml
2. Authorization header неправильно отформатирован

**Решение:**

```bash
# Правильный формат
curl -H "Authorization: Bearer sk-your-master-key"
```

### "Model not found"

**Возможные причины:**

1. Модель не задана в секции `models` конфигурации
2. Ошибка в названии модели

**Решение:**

- Проверьте список доступных моделей: `GET /v1/models`
- Убедитесь, что модель правильно задана в config.yaml

### "Timeout exceeded"

**Возможные причины:**

1. `request_timeout` слишком мал
2. Upstream провайдер медленно обрабатывает запрос
3. Проблемы с сетевой связью

**Решение:**

```yaml
server:
  request_timeout: -1    # Отключить таймаут или увеличить значение (например 5m)
```

## Development

### Структура проекта

```
auto_ai_router/
├── cmd/
│   └── server/
│       └── main.go           # Точка входа приложения
├── internal/
│   ├── config/
│   │   └── config.go         # Загрузка и валидация конфигурации
│   ├── router/
│   │   └── *.go              # Логика маршрутизации
│   └── [other packages]      # Другие компоненты
├── config.yaml               # Конфигурационный файл по умолчанию
├── Dockerfile                # Docker конфигурация
├── docker-compose.yml        # Docker Compose конфигурация
├── Makefile                  # Build и запуск скрипты
└── README.md                 # Этот файл
```

### Запуск тестов

```bash
# Все тесты
make test

# Тесты с покрытием
make test-coverage

# Посмотреть coverage отчет
make test-coverage-html
```

### Linting и форматирование

```bash
# golangci-lint проверка
make lint

# Форматирование кода
go fmt ./...

# Mod tidying
go mod tidy
```

### Добавление нового провайдера

1. Добавьте новый `ProviderType` в `internal/config/config.go`:

```go
const (
    ProviderTypeOpenAI    ProviderType = "openai"
    ProviderTypeAnthropic ProviderType = "anthropic"
    ProviderTypeVertexAI  ProviderType = "vertex-ai"
    ProviderTypeYourAPI   ProviderType = "yourapi"  // Новый провайдер
)
```

2. Обновите `IsValid()` метод в config.go

3. Добавьте специфичные для провайдера поля в `CredentialConfig` struct

4. Реализуйте обработчик в router пакете

5. Добавьте примеры в README.md и config.yaml

### Локальная разработка с Docker

```bash
# Сборка разработческого образа
docker build -f Dockerfile.dev -t auto-ai-router:dev .

# Запуск контейнера с горячей перезагрузкой
docker run -it --rm \
  -v $(pwd):/app \
  -p 8080:8080 \
  auto-ai-router:dev
```

## Contributing

Contributions приветствуются! Пожалуйста:

1. Fork репозиторий
2. Создайте feature branch (`git checkout -b feature/amazing-feature`)
3. Commit ваши изменения (`git commit -m 'Add amazing feature'`)
4. Push в branch (`git push origin feature/amazing-feature`)
5. Откройте Pull Request

### Требования к коду

- Пройти все тесты: `make test`
- Пройти linting: `make lint`
- Добавить тесты для новой функциональности
- Обновить README.md если нужны изменения в документации
- Использовать четкие git commit messages

## Лицензия

[Указать лицензию]

## Support

Если у вас есть вопросы или проблемы:

1. Проверьте [раздел Troubleshooting](#troubleshooting)
2. Посмотрите логи сервера: `docker-compose logs auto-ai-router`
3. Откройте Issue на GitHub с описанием проблемы и логами

### Логика работы лимитов

Роутер применяет **двухуровневую систему лимитов** для точного контроля использования API:

**Уровень 1: Credential (общие лимиты)**

Каждый credential имеет общий RPM/TPM лимит для **всех моделей вместе**:

```yaml
credentials:
  - name: "openai_main"
    rpm: 100      # Максимум 100 запросов в минуту для всех моделей на openai_main
    tpm: 50000    # Максимум 50000 токенов в минуту для всех моделей на openai_main
```

**Уровень 2: Model (индивидуальные лимиты)**

Каждая модель может иметь собственные лимиты для пары `(credential, model)`:

```yaml
models:
  - name: gpt-4o-mini
    credential: openai_main
    rpm: 60       # Лимит 60 RPM для gpt-4o-mini через openai_main
    tpm: 30000    # Лимит 30000 TPM для gpt-4o-mini через openai_main
```

**Вычисление итогового лимита:**

```
Итоговый лимит = MIN(credential_limit, model_limit)
```

**Пример:**

```yaml
credentials:
  - name: "openai_main"
    rpm: 100
    tpm: 50000

models:
  - name: gpt-4o-mini
    credential: openai_main
    rpm: 60
    tpm: 30000
```

Результат для пары `(openai_main, gpt-4o-mini)`:

- **RPM**: MIN(100, 60) = **60 запросов/минуту**
- **TPM**: MIN(50000, 30000) = **30000 токенов/минуту**

**Отключение лимитов:**

Используйте специальные значения для отключения лимитов:

| Параметр             | Значение     | Значение   | Описание                                |
| -------------------- | ------------ | ---------- | --------------------------------------- |
| `rpm`                | `-1`         | (no limit) | Отключить RPM лимит                     |
| `tpm`                | `0` или `-1` | (no limit) | Отключить TPM лимит                     |
| `request_timeout`    | `-1`         | (no limit) | Отключить таймаут запроса               |
| `default_models_rpm` | `-1`         | (no limit) | Без лимита RPM для моделей по умолчанию |

**Примеры:**

```yaml
# Неограниченный credential
credentials:
  - name: "unlimited"
    rpm: -1    # Без лимита
    tpm: -1    # Без лимита

# Без лимита для конкретной модели
models:
  - name: gpt-4-turbo
    credential: openai_main
    rpm: -1    # Без лимита
    tpm: -1    # Без лимита
```

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

### 4. Переменные окружения

Для безопасности храните чувствительные данные в переменных окружения:

```bash
export OPENAI_API_KEY="sk-proj-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export MASTER_KEY="sk-your-master-key-here"
```

Затем в `config.yaml` ссылайтесь на них:

```yaml
server:
  master_key: "os.environ/MASTER_KEY"

credentials:
  - name: "openai_main"
    api_key: "os.environ/OPENAI_API_KEY"
  - name: "anthropic_main"
    api_key: "os.environ/ANTHROPIC_API_KEY"
```

### 5. Использование

**Python (с OpenAI клиентом):**

```python
from openai import OpenAI

client = OpenAI(
    api_key="sk-your-master-key",  # master_key из config.yaml
    base_url="http://localhost:8080/v1",
)

# Обычный запрос
response = client.chat.completions.create(
    model="gpt-4o-mini", messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)

# Streaming запрос
stream = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Explain quantum physics"}],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

**cURL:**

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer sk-your-master-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

**Просмотр доступных моделей:**

```bash
curl http://localhost:8080/v1/models \
  -H "Authorization: Bearer sk-your-master-key"
```

**Проверка здоровья:**

```bash
curl http://localhost:8080/health
curl http://localhost:8080/vhealth  # Визуальный дашборд
```

## Как это работает

1. **Клиент отправляет запрос** с `master_key` в `Authorization: Bearer sk-your-master-key` header
2. **Роутер проверяет master_key** и валидирует содержимое запроса
3. **Round-robin выбирает лучший credential** с учетом приоритета:
   - ✓ Модель доступна у провайдера
   - ✓ Credential не забанен (fail2ban)
   - ✓ Credential имеет достаточно RPM лимита
   - ✓ Credential имеет достаточно TPM лимита
   - ✓ Model имеет достаточно RPM лимита для пары (credential, model)
   - ✓ Model имеет достаточно TPM лимита для пары (credential, model)
4. **Подменяет master_key** на реальный API ключ выбранного провайдера
5. **Проксирует запрос** к upstream API провайдера
6. **Обрабатывает ответ**:
   - Для обычных ответов: извлекает `usage.total_tokens` и обновляет счетчики
   - Для streaming: передает события клиенту и подсчитывает токены
7. **Обновляет метрики** для credential и модели (RPM/TPM счетчики)
8. **Возвращает ответ** клиенту

## API Endpoints

### Chat Completions

```
POST /v1/chat/completions
```

Основной endpoint для отправки запросов. Поддерживает:

- Обычные запросы (с возвратом полного ответа)
- Streaming запросы (Server-Sent Events)
- Автоматический выбор провайдера по доступности модели

**Требуется:** `Authorization: Bearer sk-your-master-key` header

### Список моделей

```
GET /v1/models
```

Возвращает список всех доступных моделей из конфигурации в формате OpenAI API.

### Health Check

```
GET /health
```

JSON формат с информацией о здоровье сервиса:

```json
{
  "status": "healthy",
  "credentials": {"openai_main": "available", "anthropic_main": "banned"},
  "timestamp": "2024-01-01T12:00:00Z"
}
```

### Visual Dashboard

```
GET /vhealth
```

Интерактивный HTML дашборд с:

- Статусом каждого credential
- Текущими RPM/TPM счетчиками
- Информацией о баненых провайдерах
- Таблица моделей с доступностью

### Prometheus Метрики

```
GET /metrics
```

Метрики в Prometheus текстовом формате.

## Prometheus Метрики

### Credential метрики

- `auto_ai_router_credential_rpm_current{credential="openai_main"}` - текущий RPM использование
- `auto_ai_router_credential_tpm_current{credential="openai_main"}` - текущее TPM использование
- `auto_ai_router_credential_banned{credential="openai_main"}` - статус бана (0=available, 1=banned)
- `auto_ai_router_credential_rpm_limit{credential="openai_main"}` - RPM лимит (конфигурированный)
- `auto_ai_router_credential_tpm_limit{credential="openai_main"}` - TPM лимит (конфигурированный)

### Model метрики (пары credential + model)

- `auto_ai_router_model_rpm_current{credential="openai_main",model="gpt-4o-mini"}` - текущий RPM
- `auto_ai_router_model_tpm_current{credential="openai_main",model="gpt-4o-mini"}` - текущее TPM использование
- `auto_ai_router_model_rpm_limit{credential="openai_main",model="gpt-4o-mini"}` - RPM лимит
- `auto_ai_router_model_tpm_limit{credential="openai_main",model="gpt-4o-mini"}` - TPM лимит

**Пример Prometheus запроса:**

```
# Top 5 моделей по TPM использованию
topk(5, auto_ai_router_model_tpm_current)

# Все забаненные credentials
auto_ai_router_credential_banned == 1
```
