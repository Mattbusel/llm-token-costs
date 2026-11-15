# Changelog

## 0.1.0 (2026-11-15)

First release.

- `llm costs`: what the responses in logs.db cost, grouped by model, day, month or conversation, with `--since`, `--until`, `-m`, `-n`, `--json` and `-d`.
- `llm costs estimate`: what a prompt will cost on one or more models before you send it.
- `llm costs price`: which llm-prices row a model id resolves to.
- `llm costs set-price`: your own price for a model.
- Cached input, Anthropic cache writes and long-context tiers are billed at their own rates.
- Reads both the `turns` table (llm 0.32 and later) and the older `responses` table.
- Python API: `cost_of(response)`, `price_for(model_id)`, `usage_from_tokens(...)`.
