# Examples

Эта папка содержит примеры конфигураций и использования Auto AI Router.

## Быстрый старт

1. Установите зависимости:

```bash
cd examples
pip install -r requirements.txt
```

2. Настройте переменные окружения:

```bash
cp .env.example .env
# Отредактируйте .env файл
```

3. Запустите роутер с нужной конфигурацией
4. Запустите примеры

______________________________________________________________________

## Примеры использования API

### Chat Completions

#### `basic_request.py`

Базовый пример использования chat completions API:

- Простой запрос к модели gpt-4o-mini
- Обработка ответа
- Вывод токенов использования

```bash
python examples/basic_request.py
```

#### `streaming_request.py`

Пример стриминга ответов от модели:

- Server-Sent Events (SSE) streaming
- Постепенный вывод ответа
- Работа с чанками данных

```bash
python examples/streaming_request.py
```

### Embeddings

#### `basic_embeddings.py`

Примеры работы с text-embedding-3-small:

- Создание embedding для одного текста
- Batch обработка множественных текстов
- Вычисление косинусного сходства между векторами

```bash
python examples/basic_embeddings.py
```

______________________________________________________________________

## Примеры конфигураций

### 1. Minimal Configuration (`config-minimal.yaml`)

Минимальная конфигурация для быстрого старта:

- Один провайдер (OpenAI)
- Базовые настройки без лимитов
- Простой fail2ban

**Использование:**

```bash
./auto_ai_router -config examples/config-minimal.yaml
```

**Подходит для:**

- Локальной разработки
- Тестирования
- Малых проектов с одним API ключом

______________________________________________________________________

### 2. Multi-Provider Configuration (`config-multi-provider.yaml`)

Конфигурация с несколькими провайдерами:

- OpenAI (основной и резервный)
- Azure OpenAI
- Альтернативный провайдер
- Model-aware routing
- Разные RPM лимиты для каждого провайдера

**Использование:**

```bash
./auto_ai_router -config examples/config-multi-provider.yaml
```

**Подходит для:**

- Распределения нагрузки между несколькими аккаунтами
- Повышения доступности через fallback провайдеры
- Использования разных провайдеров для разных моделей

______________________________________________________________________

### 3. Production Configuration (`config-production.yaml`)

Production-ready конфигурация:

- Несколько tier'ов провайдеров
- Строгие rate limits
- Увеличенные таймауты
- Permanent ban для сбойных провайдеров

**Использование:**

```bash
./auto_ai_router -config examples/config-production.yaml
```

**Подходит для:**

- Production окружения
- Высоконагруженных сервисов
- Критичных приложений с требованиями к SLA
