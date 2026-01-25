# Configuration Examples

Эта папка содержит примеры конфигураций для различных сценариев использования Auto AI Router.

## Доступные примеры

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

---

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

---

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
