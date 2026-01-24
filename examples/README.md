# Auto AI Router - Examples

Эта папка содержит примеры использования auto_ai_router с Python и библиотекой OpenAI.

## Установка зависимостей

```bash
cd examples
pip install -r requirements.txt
```

Или с использованием виртуального окружения:

```bash
cd examples
python3 -m venv venv
source venv/bin/activate  # На Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Настройка

1. Убедитесь, что роутер запущен:
```bash
cd ..
./auto_ai_router -config config.yaml
```

2. (Опционально) Скопируйте `.env.example` в `.env` и настройте:
```bash
cp .env.example .env
```

## Примеры

### 1. Проверка здоровья роутера
Проверяет статус роутера и доступность credentials:
```bash
python test_health.py
```

### 2. Базовый запрос (без streaming)
Отправляет обычный запрос к API:
```bash
python basic_request.py
```

### 3. Streaming запрос
Тестирует Server-Sent Events (SSE) streaming:
```bash
python streaming_request.py
```

### 4. Множественные запросы
Тестирует round-robin балансировку и rate limiting:
```bash
python multiple_requests.py
```

### 5. Метрики Prometheus
Получает и отображает метрики роутера:
```bash
python test_metrics.py
```

## Что тестируют примеры

- **test_health.py**: Health check endpoint
- **basic_request.py**: Обычные запросы без streaming
- **streaming_request.py**: SSE streaming запросы
- **multiple_requests.py**: Round-robin балансировка, RPM limiting
- **test_metrics.py**: Prometheus метрики

## Примечания

- API ключ в запросах может быть любым - роутер заменит его на настоящий из `config.yaml`
- Модель по умолчанию: `gpt-4o-mini`
- Все примеры используют `base_url="http://localhost:8080/v1"`

## Проверка метрик

После выполнения запросов можно проверить метрики:

```bash
curl http://localhost:8080/metrics | grep auto_ai_router
```

Или используйте `test_metrics.py` для более читаемого вывода.

## Отладка

Для включения debug режима в роутере, установите в `config.yaml`:
```yaml
server:
  debug: true
```

Затем перезапустите роутер и вы увидите детальные логи всех запросов.
