# llm-token-costs

[![PyPI](https://img.shields.io/pypi/v/llm-token-costs.svg)](https://pypi.org/project/llm-token-costs/)
[![Tests](https://github.com/Mattbusel/llm-token-costs/actions/workflows/test.yml/badge.svg)](https://github.com/Mattbusel/llm-token-costs/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/Mattbusel/llm-token-costs/blob/main/LICENSE)

A plugin for [LLM](https://llm.datasette.io/) that tells you what your prompts cost.

`llm` already logs the input and output tokens of every response to `logs.db`. This plugin prices those rows, so `llm costs` answers "what did I spend this week, and on which model?" without leaving the terminal. It also estimates a prompt before you send it.

Prices come from the [llm-prices](https://github.com/Mattbusel/llm-prices) Python package: a table of USD per 1M token rates for OpenAI, Anthropic, Google, Mistral, DeepSeek, xAI, Groq, Together, Fireworks and Cohere, with a source URL and a checked date on every row. (That package is a separate project from Simon Willison's [llm-prices.com](https://www.llm-prices.com/) calculator.)

## Installation

```bash
llm install llm-token-costs
```

## What did my prompts cost?

```
$ llm costs
model                        responses    input     cached   output  cost (USD)
---------------------------  ---------  -------  ---------  -------  ----------
anthropic/claude-sonnet-4-5         23  498,180  1,249,981   58,880     $2.8187
gemini/gemini-2.5-flash             31  232,996          0   19,778     $0.1193
gpt-4o-mini                         64   86,400     96,000   41,280     $0.0449
llama3.2:latest                      9   11,997          0    4,095    no price
total                              127  829,573  1,345,981  124,033     $2.9830

* No price for: llama3.2:latest (9). These are left out of the total.
  Set one with: llm costs set-price llama3.2:latest --input 0 --output 0
```

Options:

- `--since` / `--until`: a date (`2026-11-01`), an ISO time, `today`, or a span back from now (`30m`, `12h`, `7d`, `4w`).
- `--by model|day|month|conversation`: how to group the rows.
- `-m MODEL`: only this model (repeatable; aliases work).
- `-n 10`: only the first 10 groups.
- `--json`: machine readable output.
- `-d path/to/logs.db`: a different database.

```bash
llm costs --since 7d --by day
llm costs --by conversation -n 10
llm costs -m gpt-4o-mini --since 2026-11-01 --json
```

## What will this prompt cost?

```
$ cat notes.md | llm costs estimate 'Summarize this' -m gpt-4o-mini -m claude-sonnet-4.5 -m gemini-2.5-flash
```

The prompt is read from the argument and standard input, like `llm prompt`. Input tokens are counted with `tiktoken` when it is installed (`llm install 'llm-token-costs[tiktoken]'`) and estimated at about 4 characters per token otherwise. Use `--input-tokens N` to skip counting. The output length is the median output of that model in your own logs (once there are 3 or more responses), else 500, or whatever you pass to `--output-tokens`.

## Which price does a model get?

```
$ llm costs price gpt-4o-mini claude-opus-4.5 llama3.2:latest
model            price row                  input/1M  output/1M  cached/1M  status
---------------  -------------------------  --------  ---------  ---------  ------
gpt-4o-mini      openai/gpt-4o-mini             0.15        0.6      0.075  listed
claude-opus-4.5  anthropic/claude-opus-4-5         5         25        0.5  listed
llama3.2:latest  no price
```

With no arguments it lists every model you have installed. Matching tries the exact model the API reported (`resolved_model`, for example `gpt-4o-mini-2024-07-18`) before the llm model id, and understands plugin ids such as `anthropic/claude-sonnet-4-5`, `gemini/gemini-2.5-pro` and `openrouter/openai/gpt-4o`, plus `claude-opus-4.5` style aliases.

## Your own prices

For local models, private endpoints or negotiated rates:

```bash
llm costs set-price llama3.2:latest --input 0 --output 0
llm costs set-price my-azure-gpt --input 2.5 --output 10 --cache-read 1.25
llm costs set-price my-azure-gpt --remove
```

These are stored in `llm-token-costs.json` in the llm user directory and win over llm-prices.

## How tokens are billed

- **Cached input.** OpenAI and Gemini count cached tokens inside `input_tokens`; Anthropic counts them outside. The plugin reads `token_details` to split them out and bills cached tokens at the cache-read price.
- **Anthropic cache writes** bill at the 5 minute or 1 hour write price, from `cache_creation`.
- **Long context.** Models with a long-context tier (Gemini 2.5 Pro, for example) bill at the higher rate when the prompt is over the threshold.
- **Responses with no token counts** (some plugins do not report usage) are counted and listed, never guessed.
- **Prices are today's prices.** `logs.db` does not record the price at call time, so an old response is priced at the current rate. Retired models keep their last listed price.

Works with logs written by llm 0.27 and later, including the `turns` table that llm 0.32 introduced and the older `responses` table.

## Python API

```python
import llm
from llm_token_costs import cost_of, price_for

response = llm.get_model("gpt-4o-mini").prompt("Three names for a pet pelican")
print(response.text())
print(cost_of(response))          # USD, or None when the model has no price

print(price_for("claude-opus-4.5").output)  # USD per 1M output tokens
```

## Development

```bash
python -m venv venv && source venv/bin/activate
pip install -e '.[test]'
pytest
```

## License

MIT. Maintained by Matthew Busel, who also maintains [llm-prices](https://github.com/Mattbusel/llm-prices).
