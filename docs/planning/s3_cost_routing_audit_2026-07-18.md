# S3 (cost/routing) — аудит состояния и план закрытия, 2026-07-18

> Результат исследования сессии: сверка фактического кода репозитория с планом
> Sprint 3 из `poc-ai-engineer-plan.md` (S3.T1–S3.T5). Цель спринта — production-grade
> концерны: observability, model routing, prompt/response cache, budget caps, дашборд.

## TL;DR

**Компоненты S3 написаны и покрыты юнит-тестами (63 passed), но существуют как
острова: ни router, ни budget, ни cache не подключены к агенту и API.** Агент
по-прежнему работает через один `LLMProvider` с фиксированными моделями.
Работа сессии — интеграция, а не написание новых компонентов.

## Статус по задачам плана

| Задача | Компонент | Статус | Разрыв с AC |
|---|---|---|---|
| S3.T1 Observability | `observability/tracing.py`, `@traced` на узлах графа, `llm.generate.openai` span, `cost_usd` в state | ✅ сделано | Langfuse — OTLP-экспорт при `LANGFUSE_HOST` (+stub без SDK); live-проверка в UI не выполнялась |
| S3.T2 Model router | `llm/router.py` (`RoutedLLM`), `data/router.yaml`, fallback на 429/503, тесты | ⚠️ компонент есть | **Не подключён к агенту**: узлы берут прямой `LLMProvider` + модель из настроек. Escalate при low-confidence judge не реализован |
| S3.T3 Prompt cache | `llm/cache.py` (`RedisResponseCache`, `@cached`, `FakeRedisCache`), тесты | ⚠️ компонент есть | Декоратор ни на что не навешен; `cached_input_tokens`/`prompt_cache_hit_tokens` не пробрасываются в `LLMResponse` |
| S3.T4 Budget caps | `llm/budget.py` (`BudgetGuard` на Redis INCRBYFLOAT, `FakeBudgetGuard`), тесты | ⚠️ компонент есть | **Не подключён в `RoutedLLM`**; `BudgetExceededError` → HTTP 429 маппинга нет; узлы глушат все исключения `except Exception` — budget-ошибка утонет |
| S3.T5 Дашборд | `apps/api/routes/metrics.py` `/metrics/summary` (p95, cost, refusal rate из audit-лога) | ✅ сделано (разрешённая планом альтернатива Grafana) | Нет eval-score trend — приемлемо для POC |

## Ключевые наблюдения

1. **Схема интеграции очевидна из кода**: `RoutedLLM.route(role, …)` — единственная
   точка, через которую по плану должны идти все вызовы («Подключить в RoutedLLM»).
   Но интерфейс узлов — `LLMProvider.complete(model=…)`. Нужен адаптер
   роль→провайдер, чтобы не переписывать все узлы.
2. **Tenant и накопленный cost запроса** нужны budget-guard'у в момент LLM-вызова,
   но граф строится один раз, а tenant приходит с каждым запросом. Решение —
   `contextvars` (аналог ThreadLocal/MDC в Java): `agent.run()` открывает
   request-контекст {tenant, total_usd}, `RoutedLLM` его читает и пополняет.
3. **Узлы глушат исключения** (`except Exception → fallback значение`). Для
   `BudgetExceededError` это неверно: hard-stop по плану должен долететь до API
   как 429. Нужен точечный `except BudgetExceededError: raise` перед generic.
4. **Легаси**: `ModelRouter` (deprecated shim) остаётся для обратной совместимости,
   новый код идёт только через `RoutedLLM`.

## План закрытия (эта сессия)

1. `LLMResponse.cached_input_tokens` + парсинг `prompt_tokens_details.cached_tokens`
   (Gemini/OpenAI-compat) и `prompt_cache_hit_tokens` (DeepSeek) в провайдере.
2. `llm/context.py` — request-scoped контекст (tenant, накопленный cost) на contextvars.
3. `RoutedLLM(cache=…, budget_guard=…)`: route() = cache lookup → вызов
   (primary→fallback) → budget check-and-commit (после вызова, когда cost известен)
   → cache set. Cache hit не тратит бюджет.
4. `RoleScopedLLM` — адаптер `LLMProvider` поверх `(RoutedLLM, role)`; маппинг узлов:
   intent→classifier, plan/cluster→planner, summarize→summarizer,
   judge/groundedness→judge.
5. Escalate: judge при `score < 0.5` повторяет оценку через role=escalate
   (опциональный `escalate_llm` в фабрике узла).
6. `build_graph(router=…)` / `ProductInsightAgent(router=…)`; API: opt-in через
   `POC_USE_ROUTER=1` (default off — текущий однопровайдерный путь не ломаем).
7. `BudgetExceededError` → 429 в error_handlers; узлы пропускают её наверх.
8. Тесты на каждый шов; полный прогон suite.

## Definition of Done (спринт S3 закрыт, когда)

- [x] Компоненты router/budget/cache существуют и покрыты юнит-тестами
- [x] `POST /ask` c `POC_USE_ROUTER=1` идёт через RoutedLLM (роли по router.yaml)
- [x] Фейковый 429 от primary → фоллбек без падения запроса (тест)
- [x] Низкий score judge → повторная оценка через escalate (тест)
- [x] Повторный идентичный вызов → из кэша, бюджет не тратится (тест)
- [x] Лимит `per_request_usd=0.001` → `BudgetExceededError` → API 429 (тест)
- [x] `cached_input_tokens` виден в `LLMResponse` при cache-hit провайдера

## Итог сессии 2026-07-18 (закрытие выполнено)

Интеграция реализована и покрыта тестами — **304 passed, 1 skipped**, ruff чистый:

| Шов | Реализация |
|---|---|
| Cache/budget в роутере | `RoutedLLM(cache=…, budget_guard=…)`: lookup → вызов (primary→fallback) → charge → cache set; hit не тратит бюджет |
| Request-контекст | `llm/context.py` — contextvars (аналог ThreadLocal/MDC); `agent.run()` открывает scope {tenant, total_usd} |
| Адаптер узлов | `llm/role_provider.py` `RoleScopedLLM`; маппинг ролей в `graph._NODE_ROLES` (wiring-слой) |
| Escalate | judge при score < 0.5 повторяет оценку через role=escalate; fail-soft, кроме бюджета |
| Hard-stop | `BudgetExceededError` re-raise во всех LLM-узлах; API-handler → HTTP 429 `budget_exceeded` |
| API opt-in | `POC_USE_ROUTER=1` (+ `POC_REDIS_URL`; без него — in-memory fakes, dev only) |
| Cache-метрики | `LLMResponse.cached_input_tokens` ← `prompt_tokens_details.cached_tokens` / `prompt_cache_hit_tokens` |

Новые тесты: `test_router_integration.py` (cache/budget/context, 10),
`test_role_provider.py` (4), `test_judge_escalate.py` (7), `test_cached_tokens.py` (6),
budget→429 в `test_error_handlers.py`.

### Осталось за пределами кода (live-проверки, следующая сессия)

- [ ] Live smoke: `POC_USE_ROUTER=1` + реальные ключи + Redis (docker-compose) —
  прогон `/ask`, проверить trace в Langfuse UI (S3.T1 AC) и cost < 30% на повторе (S3.T3 AC)
- [ ] Обновить статус S3 в `github_portfolio_plan.md` / контексте job_fable
