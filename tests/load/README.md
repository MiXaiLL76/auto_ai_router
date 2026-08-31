# LLM Gateway Load Test (k6)

Нагрузочный тест шлюза LLM-провайдера на k6: смешанный трафик по нескольким эндпоинтам (streaming chat, non-streaming chat, streaming responses, embeddings) с метриками латентности и throughput.

## Что меряем

| Метрика | Описание |
|---|---|
| `llm_ttft_ms` | TTFT ≈ `res.timings.waiting` (TTFB). Для **не-стрим** chat это фактически полная латентность запроса, т.к. тело не стримится. |
| `llm_stream_duration_ms` | Полное время ответа (`res.timings.duration`). |
| `llm_stream_tokens_per_sec` | Токены/сек на **каждый отдельный** ответ (`out_tokens / duration`). |
| `llm_output_tokens` | Counter выходных токенов. Его `rate` в итоговой сводке = суммарный throughput, ток/сек. |
| `llm_input_tokens` | Counter входных (prompt/input) токенов. |
| `llm_bad_body` | Доля ответов, где не распарсился `usage`, либо (для стрима) отсутствует `[DONE]`. |

> **Примечание про парсинг тела:** k6 буферизует весь SSE-ответ целиком, поэтому `usage` вытаскивается регулярным выражением из тела ответа. Мок статичный, тело детерминированное — такой подход надёжен.

## Микс эндпоинтов

По умолчанию распределение **30 / 30 / 30 / 10**:

| Ключ | Эндпоинт | Режим |
|---|---|---|
| `stream` | `/v1/chat/completions` | `stream: true` (SSE) |
| `chat`   | `/v1/chat/completions` | `stream: false` (обычный JSON-ответ) |
| `resp`   | `/v1/responses`        | `stream: true` |
| `embed`  | `/v1/embeddings`       | не стримится |

## Запуск

### Против реального шлюза

```bash
ulimit -n 1048576              # обязательно: 5000 VU = тысячи сокетов
k6 run -e API_KEY=sk-... load-test.js
```

## Переменные окружения

| Переменная | По умолчанию | Назначение |
|---|---|---|
| `BASE_URL` | `https://example.ru` | Базовый URL шлюза |
| `API_KEY` | — | Уходит как `Authorization: Bearer <ключ>` |
| `AUTH_HEADER` | — | Имя заголовка авторизации, если не `Bearer` (напр. `X-Api-Key`) |
| `AUTH_VALUE` | — | Полное значение заголовка авторизации (переопределяет `API_KEY`/`Bearer`) |
| `TARGET_LOW` / `TARGET_HIGH` | `500` / `1000` | Плато нагрузки (VU) |
| `RAMP` / `HOLD` | `1m` / `3m` | Длительность разгона / удержания |
| `W_STREAM` / `W_CHAT` / `W_RESP` / `W_EMBED` | `30` / `30` / `30` / `10` | Вес эндпоинтов в миксе |

### Пример с кастомным заголовком авторизации и другим миксом

```bash
ulimit -n 1048576
k6 run \
  -e BASE_URL=https://gateway.example.com \
  -e AUTH_HEADER=X-Api-Key \
  -e AUTH_VALUE=my-secret-key \
  -e TARGET_LOW=200 -e TARGET_HIGH=500 \
  -e RAMP=30s -e HOLD=2m \
  -e W_STREAM=40 -e W_CHAT=20 -e W_RESP=30 -e W_EMBED=10 \
  load-test.js
```

### Через docker-compose

`docker-compose.yml` поднимает три сервиса:

- **k6** — сам раннер (образ `grafana/k6`), монтирует `load-test.js` и пишет метрики в InfluxDB (`K6_OUT=influxdb`);
- **influxdb** — хранилище временных рядов для метрик k6;
- **grafana** — дашборд для просмотра метрик в реальном времени (доступна на `http://localhost:3000`, анонимный доступ с правами Admin).

`ulimits.nofile` в сервисе `k6` уже выставлен в `1048576`, отдельный `ulimit` на хосте не нужен.

```bash
# .env рядом с docker-compose.yml, или через переменные окружения shell
API_KEY=sk-... BASE_URL=https://example.ru docker compose up
```

Все те же переменные (`TARGET_LOW/HIGH`, `RAMP/HOLD`, `W_STREAM/W_CHAT/W_RESP/W_EMBED`, `AUTH_HEADER/AUTH_VALUE`) пробрасываются в контейнер `k6` через `environment` в compose-файле и берутся из окружения хоста (с дефолтами, если не заданы).
