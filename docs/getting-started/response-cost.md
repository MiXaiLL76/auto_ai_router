# Получение стоимости запроса

Роутер считает стоимость каждого запроса и сообщает её **в теле ответа**, в
`usage.cost` (USD) — рядом со счётчиками токенов. Клиент, который уже читает
`usage`, получает стоимость бесплатно.

```json
{
  "id": "chatcmpl-…",
  "choices": [ … ],
  "usage": {
    "prompt_tokens": 13,
    "completion_tokens": 2,
    "total_tokens": 15,
    "cost": 0.000004095000000000001
  }
}
```

______________________________________________________________________

## Где смотреть

| Эндпоинт                        | Путь к стоимости                      |
| ------------------------------- | ------------------------------------- |
| `POST /v1/chat/completions`     | `response["usage"]["cost"]`           |
| `POST /v1/completions`          | `response["usage"]["cost"]`           |
| `POST /v1/responses`            | `response["usage"]["cost"]`           |
| `POST /v1/messages`             | `response["usage"]["cost"]`           |
| стриминг `/v1/chat/completions` | последний usage-чанк: `usage.cost`    |
| стриминг `/v1/messages`         | событие `message_delta`: `usage.cost` |

______________________________________________________________________

## Код

### OpenAI-совместимые эндпоинты (`openai` SDK)

```python
from openai import OpenAI

client = OpenAI(
    api_key="sk-…",
    base_url="https://api.vsellm.ru/v1",
)

# non-stream
r = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Ответь одним словом"}],
    max_tokens=20,
)
print(r.usage.cost)  # 0.000004095000000000001
```

В стриминге нужен `stream_options` — без него usage-чанка (и стоимости) в
потоке не будет:

```python
cost = None
stream = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Ответь одним словом"}],
    max_tokens=20,
    stream=True,
    stream_options={"include_usage": True},
)
for chunk in stream:
    if chunk.usage is not None and getattr(chunk.usage, "cost", None) is not None:
        cost = chunk.usage.cost
print(cost)
```

> `cost` — расширение роутера, в типах openai SDK его нет. Поэтому
> `getattr(usage, "cost", None)` вместо `usage.cost` — так код не падает на
> апстримах без этого поля.

### Anthropic Messages API (только стандартная библиотека)

```python
import json, os, urllib.request

request = urllib.request.Request(
    "https://api.vsellm.ru/v1/messages",
    data=json.dumps(
        {
            "model": "claude-sonnet-4.6",
            "max_tokens": 20,
            "messages": [{"role": "user", "content": "Ответь одним словом"}],
        }
    ).encode(),
    method="POST",
)
request.add_header("Content-Type", "application/json")
request.add_header("x-api-key", os.environ["AIR_API_KEY"])
request.add_header("anthropic-version", "2023-06-01")

with urllib.request.urlopen(request, timeout=120) as response:
    body = json.loads(response.read())

print(body["usage"]["cost"])
```

______________________________________________________________________

## Готовые примеры в репозитории

| Файл                                          | Что покрывает                                                 |
| --------------------------------------------- | ------------------------------------------------------------- |
| `examples/example_response_cost.py`           | chat/completions, его стриминг и responses — `openai` SDK     |
| `examples/example_response_cost_anthropic.py` | `/v1/messages` и его стриминг — только стандартная библиотека |

```bash
AIR_BASE_URL=https://api.vsellm.ru/v1 \
AIR_API_KEY=sk-… \
python3 examples/example_response_cost.py gpt-4o-mini

AIR_BASE_URL=https://api.vsellm.ru \
AIR_API_KEY=sk-… \
AIR_MODEL=claude-sonnet-4.6 \
python3 examples/example_response_cost_anthropic.py
```

______________________________________________________________________

## Что именно означает эта сумма

`usage.cost` — **себестоимость запроса по тарифам роутера**:

| Компонент           | Тариф                                                                       |
| ------------------- | --------------------------------------------------------------------------- |
| входные токены      | `input_cost_per_token` модели                                               |
| выходные токены     | `output_cost_per_token` модели                                              |
| прочитанные из кеша | `cache_read_input_token_cost`                                               |
| записанные в кеш    | `cache_creation_input_token_cost` (+ отдельный тариф для 12-часовой записи) |
| reasoning-токены    | `output_cost_per_reasoning_token` (иначе — как выходные)                    |

Проверка на живых запросах: при попадании промпта в кеш стоимость корректно
падает — на `gpt-5.5` с 1280 кешированными токенами `0.0125255` → `0.0050375`.

**Это не счёт провайдера.** Роутер перепродаёт по своим ценам, поэтому
сравнивайте со своим прайсом, если именно по нему выставляете счёт клиенту.

______________________________________________________________________

## Кто получает стоимость

Выдача стоимости — настройка на уровне организации
(`organization_policies[].response_cost_mode`):

| Режим       | Что приходит клиенту                      |
| ----------- | ----------------------------------------- |
| `off`       | ничего (значение по умолчанию)            |
| `total`     | `usage.cost`                              |
| `breakdown` | `usage.cost` плюс разбивка по компонентам |

Организация без явного режима стоимость не получает.

______________________________________________________________________

## Особенности

- **Заголовка `X-Total-Cost` сейчас нет** — стоимость берите из тела ответа.
- **Разбивка по компонентам отдельно не отдаётся** в `usage`: reasoning-токены
  идут отдельной строкой `reasoning_cost` в логах, поэтому
  `output_cost` может быть нулевым, хотя `total_cost` — нет. Итог всегда
  сходится: `total = input + output + cached_input + cache_creation + reasoning + …`.
- **При ошибке (4xx/5xx) стоимости нет** — запрос не был выполнен.
- **В стриминге `/v1/chat/completions` нужен `stream_options={"include_usage": true}`.**
  Для `/v1/messages` usage приходит в событии `message_delta` сам.
- **Мультипликатор `rate` из тарифов не применяется** — это заложено в коде
  (`models/manager.go`: «parsed and preserved … but it is not yet applied
  anywhere in cost calculation»). Если вам нужна цена с наценкой, считайте её
  сами.
