"""LLM plugin that prices the responses in logs.db with the llm-prices package.

Commands added to ``llm``:

- ``llm costs``: what your logged prompts cost, grouped by model, day, month or
  conversation.
- ``llm costs estimate``: what a prompt would cost before you run it.
- ``llm costs price``: which price row a model resolves to.
- ``llm costs set-price``: your own price for a model llm-prices does not list
  (local models, private endpoints, negotiated rates).

Prices come from the ``llm-prices`` Python package (github.com/Mattbusel/llm-prices),
a table of USD-per-1M-token rates with a source URL and checked date on every row.
It is a different project from Simon Willison's llm-prices.com calculator.
"""

from __future__ import annotations

import datetime
import json
import math
import pathlib
import re
import statistics
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union

import click
import llm
import llm_prices
import sqlite_utils

__version__ = "0.1.0"

__all__ = [
    "LoggedResponse",
    "Override",
    "cost_of",
    "iter_logged_responses",
    "load_overrides",
    "price_for",
    "usage_from_tokens",
]

OVERRIDES_FILENAME = "llm-token-costs.json"
DEFAULT_OUTPUT_TOKENS = 500
_DOTTED_VERSION = re.compile(r"(?<=\d)\.(?=\d)")
_RELATIVE = re.compile(r"^(\d+)\s*([mhdw])$")


# -- prices -------------------------------------------------------------------


@dataclass(frozen=True)
class Override:
    """A price you set with ``llm costs set-price``, USD per 1M tokens."""

    model: str
    input: float
    output: float
    cache_read: Optional[float] = None

    provider = "custom"
    source_url = None
    checked = None
    status = "custom"

    def cost_usage(self, usage: llm_prices.Usage) -> "_Total":
        cache_rate = self.cache_read if self.cache_read is not None else self.input
        total = (
            usage.input * self.input
            + usage.cache_read * cache_rate
            + (usage.cache_write_5m + usage.cache_write_1h) * self.input
            + usage.output * self.output
        ) / 1_000_000
        return _Total(total)


@dataclass(frozen=True)
class _Total:
    total: float


PriceLike = Union[llm_prices.Price, Override]


def overrides_path() -> pathlib.Path:
    return llm.user_dir() / OVERRIDES_FILENAME


def load_overrides(path: Optional[pathlib.Path] = None) -> Dict[str, Override]:
    """Prices set with ``llm costs set-price``, keyed by model id."""
    path = path or overrides_path()
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    out = {}
    for model, row in raw.items():
        try:
            out[model] = Override(
                model=model,
                input=float(row["input"]),
                output=float(row["output"]),
                cache_read=(
                    float(row["cache_read"])
                    if row.get("cache_read") is not None
                    else None
                ),
            )
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _save_overrides(overrides: Dict[str, Override], path: pathlib.Path) -> None:
    data = {}
    for model, o in sorted(overrides.items()):
        row: Dict[str, float] = {"input": o.input, "output": o.output}
        if o.cache_read is not None:
            row["cache_read"] = o.cache_read
        data[model] = row
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _candidates(name: str) -> Iterator[Tuple[Optional[str], str]]:
    """(provider hint, model name) pairs to try for one model id, best first."""
    seen = set()

    def emit(provider, model):
        key = (provider, model)
        if model and key not in seen:
            seen.add(key)
            return True
        return False

    names = [name]
    if "/" in name:
        prefix, rest = name.split("/", 1)
        if emit(prefix, rest):
            yield prefix, rest
        names.append(name.rsplit("/", 1)[1])
    if name.startswith("models/"):
        names.append(name[len("models/") :])
    for n in list(names):
        dashed = _DOTTED_VERSION.sub("-", n)
        if dashed != n:
            names.append(dashed)
    for n in names:
        if emit(None, n):
            yield None, n


def price_for(
    model_id: Optional[str],
    resolved_model: Optional[str] = None,
    overrides: Optional[Dict[str, Override]] = None,
) -> Optional[PriceLike]:
    """The price that applies to a model, or ``None`` when no table lists it.

    ``resolved_model`` (the exact model the API reported) is tried before the
    llm model id. Your own ``set-price`` values win over llm-prices.
    """
    overrides = load_overrides() if overrides is None else overrides
    names = [n for n in (resolved_model, model_id) if n]
    for n in names:
        if n in overrides:
            return overrides[n]
    for n in names:
        for provider, candidate in _candidates(n):
            try:
                if provider:
                    price = llm_prices.lookup_provider(provider, candidate)
                else:
                    price = llm_prices.lookup(candidate)
            except (ValueError, KeyError, TypeError):
                price = None
            if price is not None:
                return price
    return None


# -- token usage --------------------------------------------------------------


def _int(value: Any) -> int:
    try:
        return max(int(value or 0), 0)
    except (TypeError, ValueError):
        return 0


def usage_from_tokens(
    input_tokens: Optional[int],
    output_tokens: Optional[int],
    token_details: Union[None, str, Dict[str, Any]] = None,
) -> llm_prices.Usage:
    """Split llm's logged counts into the token kinds that bill differently.

    llm stores ``input_tokens``, ``output_tokens`` and the provider's extra usage
    fields in ``token_details``. Cached tokens are counted inside ``input_tokens``
    for OpenAI and Gemini and outside it for Anthropic.
    """
    details: Dict[str, Any] = {}
    if isinstance(token_details, str) and token_details:
        try:
            details = json.loads(token_details) or {}
        except ValueError:
            details = {}
    elif isinstance(token_details, dict):
        details = token_details
    if not isinstance(details, dict):
        details = {}

    inp = _int(input_tokens)
    out = _int(output_tokens)
    cache_read = cw5 = cw1 = 0

    anthropic_keys = (
        "cache_read_input_tokens",
        "cache_creation_input_tokens",
        "cache_creation",
    )
    if any(k in details for k in anthropic_keys):
        cache_read = _int(details.get("cache_read_input_tokens"))
        creation = details.get("cache_creation")
        if isinstance(creation, dict):
            cw5 = _int(creation.get("ephemeral_5m_input_tokens"))
            cw1 = _int(creation.get("ephemeral_1h_input_tokens"))
        if not (cw5 or cw1):
            cw5 = _int(details.get("cache_creation_input_tokens"))
    else:
        cached = 0
        for key in ("prompt_tokens_details", "input_tokens_details"):
            sub = details.get(key)
            if isinstance(sub, dict):
                cached = cached or _int(sub.get("cached_tokens"))
        cached = cached or _int(details.get("cachedContentTokenCount"))
        cached = min(cached, inp)
        inp -= cached
        cache_read = cached

    return llm_prices.Usage(
        input=inp,
        output=out,
        cache_read=cache_read,
        cache_write_5m=cw5,
        cache_write_1h=cw1,
    )


def _usage_cost(price: PriceLike, usage: llm_prices.Usage) -> Optional[float]:
    result = price.cost_usage(usage)
    if result is None:
        # A cache-write price the table does not list: bill those as input.
        if usage.cache_write_5m or usage.cache_write_1h:
            fallback = llm_prices.Usage(
                input=usage.input + usage.cache_write_5m + usage.cache_write_1h,
                output=usage.output,
                cache_read=usage.cache_read,
            )
            result = price.cost_usage(fallback)
    return None if result is None else float(result.total)


def cost_of(response: Any, overrides: Optional[Dict[str, Override]] = None):
    """USD cost of an ``llm`` Response, or ``None`` when it cannot be priced.

    >>> response = llm.get_model("gpt-4o-mini").prompt("hi")
    >>> cost_of(response)  # doctest: +SKIP
    1.5e-05
    """
    usage = response.usage()
    if usage.input is None and usage.output is None:
        return None
    price = price_for(
        response.model.model_id,
        getattr(response, "resolved_model", None),
        overrides,
    )
    if price is None:
        return None
    return _usage_cost(
        price, usage_from_tokens(usage.input, usage.output, usage.details)
    )


# -- reading logs.db ----------------------------------------------------------


@dataclass
class LoggedResponse:
    id: str
    model: str
    resolved_model: Optional[str]
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    token_details: Optional[str]
    datetime_utc: str
    conversation_id: Optional[str]
    conversation_name: Optional[str]


_TURNS_SQL = """
select turns.id, turns.model, turns.resolved_model, turns.input_tokens,
    turns.output_tokens, turns.token_details, turns.datetime_utc,
    turns.thread_id as conversation_id, threads.name as conversation_name
from turns left join threads on turns.thread_id = threads.id
"""

_RESPONSES_SQL = """
select responses.id, responses.model, {resolved} as resolved_model,
    {inp} as input_tokens, {out} as output_tokens, {details} as token_details,
    responses.datetime_utc, responses.conversation_id,
    {conv_name} as conversation_name
from responses{join}
{where}
"""


def iter_logged_responses(db: sqlite_utils.Database) -> Iterator[LoggedResponse]:
    """Every model call in an llm logs database, old and new schema.

    llm 0.32 and later log to ``turns``. Older versions logged to ``responses``.
    Rows in both (written during the switch) are read once, from ``turns``.
    """
    tables = set(db.table_names())
    if "turns" in tables:
        for row in db.query(_TURNS_SQL):
            yield LoggedResponse(**row)
    if "responses" in tables:
        cols = {c.name for c in db["responses"].columns}

        def col(name):
            return f"responses.{name}" if name in cols else "null"

        has_conversations = "conversations" in tables
        sql = _RESPONSES_SQL.format(
            resolved=col("resolved_model"),
            inp=col("input_tokens"),
            out=col("output_tokens"),
            details=col("token_details"),
            conv_name="conversations.name" if has_conversations else "null",
            join=(
                " left join conversations"
                " on responses.conversation_id = conversations.id"
                if has_conversations
                else ""
            ),
            where=(
                "where responses.id not in (select id from turns)"
                if "turns" in tables
                else ""
            ),
        )
        for row in db.query(sql):
            yield LoggedResponse(**row)


def _parse_when(
    value: str, now: Optional[datetime.datetime] = None
) -> datetime.datetime:
    now = now or datetime.datetime.now(datetime.timezone.utc)
    value = value.strip()
    m = _RELATIVE.match(value)
    if m:
        n, unit = int(m.group(1)), m.group(2)
        delta = {
            "m": datetime.timedelta(minutes=n),
            "h": datetime.timedelta(hours=n),
            "d": datetime.timedelta(days=n),
            "w": datetime.timedelta(weeks=n),
        }[unit]
        return now - delta
    if value == "today":
        return now.replace(hour=0, minute=0, second=0, microsecond=0)
    try:
        parsed = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise click.BadParameter(
            f"{value!r}: use a date like 2026-11-01, an ISO timestamp, 'today' "
            "or a span like 30m, 12h, 7d, 4w"
        ) from None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=datetime.timezone.utc)
    return parsed


def _row_time(value: Optional[str]) -> Optional[datetime.datetime]:
    if not value:
        return None
    try:
        parsed = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=datetime.timezone.utc)
    return parsed


# -- summarising --------------------------------------------------------------


@dataclass
class _Group:
    key: str
    label: str
    responses: int = 0
    input: int = 0
    cache_read: int = 0
    cache_write: int = 0
    output: int = 0
    cost: float = 0.0
    priced: int = 0
    unpriced: int = 0
    no_usage: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "responses": self.responses,
            "input_tokens": self.input,
            "cached_input_tokens": self.cache_read,
            "cache_write_tokens": self.cache_write,
            "output_tokens": self.output,
            "cost_usd": round(self.cost, 8),
            "priced_responses": self.priced,
            "unpriced_responses": self.unpriced,
            "responses_without_token_counts": self.no_usage,
        }


def summarize(
    rows: Iterable[LoggedResponse],
    by: str = "model",
    models: Tuple[str, ...] = (),
    since: Optional[datetime.datetime] = None,
    until: Optional[datetime.datetime] = None,
    overrides: Optional[Dict[str, Override]] = None,
) -> Tuple[List[_Group], Dict[str, int], Dict[str, int]]:
    """Group logged responses and price them.

    Returns the groups (most expensive first), the unpriced models with their
    response counts, and the models whose responses had no token counts.
    """
    overrides = load_overrides() if overrides is None else overrides
    price_cache: Dict[Tuple[Optional[str], Optional[str]], Optional[PriceLike]] = {}
    groups: Dict[str, _Group] = {}
    unpriced: Dict[str, int] = {}
    no_usage: Dict[str, int] = {}
    model_filter = set(models)

    for row in rows:
        if model_filter and not (
            row.model in model_filter or row.resolved_model in model_filter
        ):
            continue
        if since or until:
            when = _row_time(row.datetime_utc)
            if when is None:
                continue
            if since and when < since:
                continue
            if until and when >= until:
                continue

        if by == "day":
            key = label = (row.datetime_utc or "")[:10] or "unknown"
        elif by == "month":
            key = label = (row.datetime_utc or "")[:7] or "unknown"
        elif by == "conversation":
            key = row.conversation_id or row.id
            label = (row.conversation_name or "").strip().replace("\n", " ")[:40] or key
        else:
            key = label = row.model or "unknown"

        g = groups.get(key)
        if g is None:
            g = groups[key] = _Group(key=key, label=label)
        g.responses += 1

        if row.input_tokens is None and row.output_tokens is None:
            g.no_usage += 1
            no_usage[row.model] = no_usage.get(row.model, 0) + 1
            continue

        usage = usage_from_tokens(
            row.input_tokens, row.output_tokens, row.token_details
        )
        g.input += usage.input
        g.cache_read += usage.cache_read
        g.cache_write += usage.cache_write_5m + usage.cache_write_1h
        g.output += usage.output

        ck = (row.model, row.resolved_model)
        if ck not in price_cache:
            price_cache[ck] = price_for(row.model, row.resolved_model, overrides)
        price = price_cache[ck]
        cost = _usage_cost(price, usage) if price is not None else None
        if cost is None:
            g.unpriced += 1
            unpriced[row.model] = unpriced.get(row.model, 0) + 1
        else:
            g.priced += 1
            g.cost += cost

    if by in ("day", "month"):
        ordered = sorted(groups.values(), key=lambda g: g.key, reverse=True)
    else:
        ordered = sorted(groups.values(), key=lambda g: (-g.cost, -g.responses, g.key))
    return ordered, unpriced, no_usage


def _money(value: float) -> str:
    if value == 0:
        return "$0.00"
    if value < 0.01:
        return f"${value:.6f}".rstrip("0")
    return f"${value:,.4f}"


def _table(headers: List[str], rows: List[List[str]], align: str) -> str:
    widths = [len(h) for h in headers]
    for r in rows:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(cell))

    def fmt(cells):
        out = []
        for i, cell in enumerate(cells):
            out.append(
                cell.ljust(widths[i]) if align[i] == "l" else cell.rjust(widths[i])
            )
        return "  ".join(out).rstrip()

    lines = [fmt(headers), fmt(["-" * w for w in widths])]
    lines.extend(fmt(r) for r in rows)
    return "\n".join(lines)


def _prices_footer(logged: bool = True) -> str:
    checked = max((p.checked for p in llm_prices.all() if p.checked), default="")
    version = getattr(llm_prices, "__version__", "")
    text = (
        f"Prices: llm-prices {version} (github.com/Mattbusel/llm-prices), checked "
        f"{checked}."
    )
    if logged:
        text += (
            " logs.db does not store the price at call time, so every response is"
            " priced at today's rate."
        )
    return text


# -- tokens for estimates ------------------------------------------------------


def _count_tokens(text: str) -> Tuple[int, str]:
    try:
        import tiktoken  # type: ignore

        enc = tiktoken.get_encoding("o200k_base")
        return len(enc.encode(text)), "tiktoken o200k_base"
    except Exception:
        return (
            max(1, math.ceil(len(text) / 4)) if text else 0
        ), "about 4 characters per token"


def _log_stats(
    db_path: pathlib.Path, model_id: str
) -> Tuple[Optional[int], Optional[str]]:
    """Median output tokens (with 3+ logged responses) and the latest resolved model."""
    if not db_path.exists():
        return None, None
    db = sqlite_utils.Database(db_path)
    rows = [
        r
        for r in iter_logged_responses(db)
        if r.model == model_id or r.resolved_model == model_id
    ]
    values = [r.output_tokens for r in rows if r.output_tokens is not None]
    resolved = None
    dated = sorted(
        (r for r in rows if r.resolved_model), key=lambda r: r.datetime_utc or ""
    )
    if dated:
        resolved = dated[-1].resolved_model
    median = int(statistics.median(values)) if len(values) >= 3 else None
    return median, resolved


# -- CLI ------------------------------------------------------------------------


def _logs_db_path() -> pathlib.Path:
    return llm.user_dir() / "logs.db"


def _resolve_model_id(name: str) -> str:
    try:
        return llm.get_model(name).model_id
    except Exception:
        return name


@llm.hookimpl
def register_commands(cli):
    @cli.group(name="costs", invoke_without_command=True)
    @click.option(
        "-d",
        "--database",
        type=click.Path(dir_okay=False, path_type=pathlib.Path),
        help="Path to logs.db (default: the one llm logs to)",
    )
    @click.option(
        "-m", "--model", "models", multiple=True, help="Only this model (repeatable)"
    )
    @click.option(
        "--since", help="Start: 2026-11-01, an ISO time, 'today', or 7d / 12h / 4w ago"
    )
    @click.option("--until", help="End (exclusive), same formats as --since")
    @click.option(
        "--by",
        type=click.Choice(["model", "day", "month", "conversation"]),
        default="model",
        show_default=True,
        help="How to group the rows",
    )
    @click.option(
        "-n", "--limit", type=int, default=0, help="Show only the first N groups"
    )
    @click.option("json_", "--json", is_flag=True, help="Output JSON")
    @click.pass_context
    def costs(ctx, database, models, since, until, by, limit, json_):
        """What the prompts in your llm logs cost, priced with llm-prices.

        \b
        llm costs                      # all time, by model
        llm costs --since 7d --by day  # last week, per day
        llm costs -m gpt-4o-mini --since 2026-11-01
        llm costs --by conversation -n 10

        Subcommands: estimate, price, set-price.
        """
        if ctx.invoked_subcommand is not None:
            return
        db_path = database or _logs_db_path()
        if not db_path.exists():
            raise click.ClickException(f"No logs database at {db_path}")
        since_dt = _parse_when(since) if since else None
        until_dt = _parse_when(until) if until else None
        resolved_models = tuple(_resolve_model_id(m) for m in models) + tuple(models)
        db = sqlite_utils.Database(db_path)
        groups, unpriced, no_usage = summarize(
            iter_logged_responses(db),
            by=by,
            models=resolved_models if models else (),
            since=since_dt,
            until=until_dt,
        )
        total_cost = sum(g.cost for g in groups)
        shown = groups[:limit] if limit and limit > 0 else groups

        if json_:
            click.echo(
                json.dumps(
                    {
                        "by": by,
                        "groups": [g.as_dict() for g in shown],
                        "total_cost_usd": round(total_cost, 8),
                        "total_responses": sum(g.responses for g in groups),
                        "unpriced_models": unpriced,
                        "models_without_token_counts": no_usage,
                        "prices": {
                            "package": "llm-prices",
                            "version": getattr(llm_prices, "__version__", None),
                        },
                    },
                    indent=2,
                )
            )
            return

        if not groups:
            click.echo("No logged responses match.")
            return

        headers = [by, "responses", "input", "cached", "output", "cost (USD)"]
        rows = []
        for g in shown:
            cost = _money(g.cost) if g.priced else "no price"
            if g.priced and (g.unpriced or g.no_usage):
                cost += " *"
            rows.append(
                [
                    g.label,
                    f"{g.responses:,}",
                    f"{g.input + g.cache_write:,}",
                    f"{g.cache_read:,}",
                    f"{g.output:,}",
                    cost,
                ]
            )
        rows.append(
            [
                "total",
                f"{sum(g.responses for g in groups):,}",
                f"{sum(g.input + g.cache_write for g in groups):,}",
                f"{sum(g.cache_read for g in groups):,}",
                f"{sum(g.output for g in groups):,}",
                _money(total_cost),
            ]
        )
        click.echo(_table(headers, rows, "lrrrrr"))
        if limit and len(groups) > limit:
            click.echo(
                f"({len(groups) - limit} more groups not shown; total covers all)"
            )
        if unpriced or no_usage:
            click.echo("")
        if unpriced:
            names = ", ".join(f"{m} ({n})" for m, n in sorted(unpriced.items()))
            click.echo(f"* No price for: {names}. These are left out of the total.")
            first = sorted(unpriced)[0]
            click.echo(
                f"  Set one with: llm costs set-price {first} --input 0 --output 0"
            )
        if no_usage:
            names = ", ".join(f"{m} ({n})" for m, n in sorted(no_usage.items()))
            click.echo(f"* No token counts logged for: {names}.")
        click.echo("")
        click.echo(_prices_footer())

    @costs.command(name="estimate")
    @click.argument("prompt", required=False)
    @click.option(
        "-m", "--model", "models", multiple=True, help="Model to price (repeatable)"
    )
    @click.option("-s", "--system", help="System prompt, counted as input")
    @click.option(
        "--input-tokens",
        type=int,
        help="Use this input token count instead of counting",
    )
    @click.option("--output-tokens", type=int, help="Expected output tokens")
    @click.option(
        "-d",
        "--database",
        type=click.Path(dir_okay=False, path_type=pathlib.Path),
        help="logs.db used for the typical output length",
    )
    @click.option("json_", "--json", is_flag=True, help="Output JSON")
    def estimate(prompt, models, system, input_tokens, output_tokens, database, json_):
        """Estimate what a prompt will cost before you send it.

        The prompt comes from the argument and/or standard input, the same way
        `llm prompt` reads it. Output length defaults to the median output of
        that model in your logs (with at least 3 logged responses), else 500.

        \b
        llm costs estimate 'Summarize this' -m gpt-4o-mini -m claude-haiku-4.5 < notes.txt
        cat big.py | llm costs estimate -m gpt-5 --output-tokens 2000
        """
        text = ""
        if input_tokens is None and not sys.stdin.isatty():
            text = sys.stdin.read()
        if prompt:
            text = f"{text}\n{prompt}" if text else prompt
        if system:
            text = f"{system}\n{text}"
        if input_tokens is None:
            if not text:
                raise click.UsageError(
                    "Give a prompt, pipe text in, or pass --input-tokens"
                )
            n_in, method = _count_tokens(text)
        else:
            n_in, method = input_tokens, "given"

        if not models:
            models = (llm.get_default_model(),)
        db_path = database or _logs_db_path()
        overrides = load_overrides()
        results = []
        for name in models:
            model_id = _resolve_model_id(name)
            median, resolved = _log_stats(db_path, model_id)
            if output_tokens is not None:
                n_out, basis = output_tokens, "given"
            else:
                if median is not None:
                    n_out, basis = median, "median in your logs"
                else:
                    n_out, basis = DEFAULT_OUTPUT_TOKENS, "default"
            price = price_for(model_id, resolved, overrides)
            cost = None
            if price is not None:
                cost = _usage_cost(price, llm_prices.Usage(input=n_in, output=n_out))
            results.append(
                {
                    "model": model_id,
                    "input_tokens": n_in,
                    "output_tokens": n_out,
                    "output_basis": basis,
                    "price_row": (
                        None if price is None else f"{price.provider}/{price.model}"
                    ),
                    "cost_usd": cost,
                }
            )

        if json_:
            click.echo(
                json.dumps({"token_count": method, "estimates": results}, indent=2)
            )
            return
        rows = [
            [
                r["model"],
                f"{r['input_tokens']:,}",
                f"{r['output_tokens']:,} ({r['output_basis']})",
                "no price" if r["cost_usd"] is None else _money(r["cost_usd"]),
            ]
            for r in results
        ]
        click.echo(_table(["model", "input", "output", "cost (USD)"], rows, "lrrr"))
        click.echo("")
        click.echo(f"Input tokens: {method}. " + _prices_footer(logged=False))

    @costs.command(name="price")
    @click.argument("models", nargs=-1)
    @click.option("json_", "--json", is_flag=True, help="Output JSON")
    def price_cmd(models, json_):
        """Show the price row each model resolves to.

        With no arguments, lists every model llm has installed.

        \b
        llm costs price gpt-4o-mini claude-sonnet-4.5
        """
        if models:
            ids = [_resolve_model_id(m) for m in models]
        else:
            ids = [m.model_id for m in llm.get_models()]
        overrides = load_overrides()
        out = []
        for model_id in ids:
            p = price_for(model_id, None, overrides)
            out.append(
                {
                    "model": model_id,
                    "matched": None if p is None else f"{p.provider}/{p.model}",
                    "input": None if p is None else p.input,
                    "output": None if p is None else p.output,
                    "cache_read": None if p is None else p.cache_read,
                    "status": None if p is None else p.status,
                    "checked": None if p is None else p.checked,
                    "source_url": None if p is None else p.source_url,
                }
            )
        if json_:
            click.echo(json.dumps(out, indent=2))
            return

        def rate(v):
            return "" if v is None else f"{v:g}"

        rows = [
            [
                r["model"],
                r["matched"] or "no price",
                rate(r["input"]),
                rate(r["output"]),
                rate(r["cache_read"]),
                r["status"] or "",
            ]
            for r in out
        ]
        click.echo(
            _table(
                ["model", "price row", "input/1M", "output/1M", "cached/1M", "status"],
                rows,
                "llrrrl",
            )
        )

    @costs.command(name="set-price")
    @click.argument("model")
    @click.option("--input", "input_", type=float, help="USD per 1M input tokens")
    @click.option("--output", "output_", type=float, help="USD per 1M output tokens")
    @click.option("--cache-read", type=float, help="USD per 1M cached input tokens")
    @click.option("--remove", is_flag=True, help="Delete your price for this model")
    def set_price(model, input_, output_, cache_read, remove):
        """Set your own price for a model, in USD per 1M tokens.

        Useful for local models (price 0), private endpoints and negotiated rates.
        Your price wins over llm-prices for that model id.

        \b
        llm costs set-price llama3.2:latest --input 0 --output 0
        llm costs set-price my-azure-gpt --input 2.5 --output 10 --cache-read 1.25
        """
        path = overrides_path()
        overrides = load_overrides(path)
        if remove:
            if overrides.pop(model, None) is None:
                raise click.ClickException(f"No price set for {model}")
            _save_overrides(overrides, path)
            click.echo(f"Removed your price for {model}")
            return
        if input_ is None or output_ is None:
            raise click.UsageError("Pass both --input and --output (or --remove)")
        if min(input_, output_) < 0 or (cache_read is not None and cache_read < 0):
            raise click.UsageError("Prices cannot be negative")
        overrides[model] = Override(model, input_, output_, cache_read)
        _save_overrides(overrides, path)
        click.echo(f"{model}: ${input_:g} input, ${output_:g} output per 1M tokens")
