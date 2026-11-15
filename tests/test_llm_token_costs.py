import datetime
import json

import llm
import llm_prices
import pytest
import sqlite_utils
from click.testing import CliRunner
from llm.cli import cli
from llm.plugins import pm

import llm_token_costs
from llm_token_costs import Override, price_for, usage_from_tokens


class FakeModel(llm.Model):
    """Logs fixed usage the way the OpenAI plugin does, with no network call."""

    model_id = "fake-gpt"
    can_stream = False

    def __init__(self):
        self.usage = (1_000, 200, {"prompt_tokens_details": {"cached_tokens": 400}})

    def execute(self, prompt, stream, response, conversation):
        inp, out, details = self.usage
        response.set_usage(input=inp, output=out, details=details)
        response.set_resolved_model("gpt-4o-mini-2024-07-18")
        return ["ok"]


class NoPriceModel(llm.Model):
    model_id = "local-thing"
    can_stream = False

    def execute(self, prompt, stream, response, conversation):
        response.set_usage(input=50, output=10)
        return ["ok"]


@pytest.fixture(autouse=True)
def user_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_USER_PATH", str(tmp_path))
    return tmp_path


@pytest.fixture
def fake_models():
    class Plugin:
        __name__ = "FakeModels"

        @llm.hookimpl
        def register_models(self, register):
            register(FakeModel())
            register(NoPriceModel())

    pm.register(Plugin(), name="test-fake-models")
    try:
        yield
    finally:
        pm.unregister(name="test-fake-models")


def run(*args, input=None):
    result = CliRunner().invoke(cli, list(args), input=input, catch_exceptions=False)
    assert result.exit_code == 0, result.output
    return result.output


def expected_gpt4o_mini():
    p = llm_prices.lookup("gpt-4o-mini")
    return (600 * p.input + 400 * p.cache_read + 200 * p.output) / 1_000_000


# -- usage parsing ---------------------------------------------------------------


def test_usage_openai_chat_cached_tokens_are_inside_input():
    u = usage_from_tokens(1000, 200, {"prompt_tokens_details": {"cached_tokens": 400}})
    assert (u.input, u.cache_read, u.output) == (600, 400, 200)


def test_usage_openai_responses_api():
    u = usage_from_tokens(
        1000, 50, json.dumps({"input_tokens_details": {"cached_tokens": 1000}})
    )
    assert (u.input, u.cache_read) == (0, 1000)


def test_usage_gemini_cached_content():
    u = usage_from_tokens(5000, 100, {"cachedContentTokenCount": 4000})
    assert (u.input, u.cache_read) == (1000, 4000)


def test_usage_anthropic_cache_is_outside_input():
    u = usage_from_tokens(
        20,
        300,
        {
            "cache_read_input_tokens": 9000,
            "cache_creation_input_tokens": 1200,
            "cache_creation": {
                "ephemeral_5m_input_tokens": 200,
                "ephemeral_1h_input_tokens": 1000,
            },
        },
    )
    assert (u.input, u.cache_read, u.cache_write_5m, u.cache_write_1h) == (
        20,
        9000,
        200,
        1000,
    )


def test_usage_anthropic_total_cache_creation_is_5m():
    u = usage_from_tokens(20, 300, {"cache_creation_input_tokens": 1200})
    assert (u.cache_write_5m, u.cache_write_1h) == (1200, 0)


def test_usage_tolerates_bad_details():
    assert usage_from_tokens(None, 5, "not json").output == 5
    assert usage_from_tokens(10, 5, "[1, 2]").input == 10


# -- price resolution ---------------------------------------------------------------


@pytest.mark.parametrize(
    "model_id,provider,expected",
    [
        ("gpt-4o-mini", "openai", "gpt-4o-mini"),
        ("anthropic/claude-sonnet-4-5", "anthropic", "claude-sonnet-4-5"),
        ("claude-opus-4.5", "anthropic", "claude-opus-4-5"),
        ("gemini/gemini-2.5-pro", "google", "gemini-2.5-pro"),
        ("openrouter/openai/gpt-4o", "openai", "gpt-4o"),
    ],
)
def test_price_for_llm_model_ids(model_id, provider, expected):
    p = price_for(model_id, overrides={})
    assert p is not None
    assert p.provider == provider
    assert p.model.startswith(expected)


def test_resolved_model_is_tried_first():
    p = price_for("my-alias", "gpt-4o-mini-2024-07-18", overrides={})
    assert p.model == "gpt-4o-mini"


def test_unknown_model_has_no_price():
    assert price_for("llama3.2:latest", overrides={}) is None


def test_override_wins():
    o = Override("gpt-4o-mini", 1.0, 2.0)
    assert price_for("gpt-4o-mini", overrides={"gpt-4o-mini": o}) is o


# -- llm costs against a real logs.db ------------------------------------------------


def test_costs_end_to_end(fake_models):
    run("prompt", "-m", "fake-gpt", "hello")
    run("prompt", "-m", "fake-gpt", "again")
    run("prompt", "-m", "local-thing", "hi")
    out = run("costs")
    assert "fake-gpt" in out
    assert "No price for: local-thing (1)" in out
    assert "llm-prices" in out

    data = json.loads(run("costs", "--json"))
    group = {g["key"]: g for g in data["groups"]}["fake-gpt"]
    assert group["responses"] == 2
    assert group["input_tokens"] == 1200
    assert group["cached_input_tokens"] == 800
    assert group["output_tokens"] == 400
    assert group["cost_usd"] == pytest.approx(2 * expected_gpt4o_mini())
    assert data["unpriced_models"] == {"local-thing": 1}
    assert data["total_cost_usd"] == pytest.approx(2 * expected_gpt4o_mini())


def test_costs_filters_and_grouping(fake_models):
    run("prompt", "-m", "fake-gpt", "hello")
    run("prompt", "-m", "local-thing", "hi")
    data = json.loads(run("costs", "--json", "-m", "fake-gpt"))
    assert [g["key"] for g in data["groups"]] == ["fake-gpt"]

    today = datetime.datetime.now(datetime.timezone.utc).date().isoformat()
    data = json.loads(run("costs", "--json", "--by", "day", "--since", "1d"))
    assert data["groups"][0]["key"] == today
    assert data["groups"][0]["responses"] == 2

    data = json.loads(run("costs", "--json", "--until", "2000-01-01"))
    assert data["groups"] == []


def test_set_price_prices_local_model(fake_models):
    run("prompt", "-m", "local-thing", "hi")
    run("costs", "set-price", "local-thing", "--input", "1", "--output", "10")
    data = json.loads(run("costs", "--json"))
    assert data["unpriced_models"] == {}
    assert data["total_cost_usd"] == pytest.approx((50 * 1 + 10 * 10) / 1_000_000)
    assert "local-thing" in run("costs", "price", "local-thing")
    run("costs", "set-price", "local-thing", "--remove")
    assert json.loads(run("costs", "--json"))["unpriced_models"] == {"local-thing": 1}


def test_cost_of_response(fake_models):
    response = llm.get_model("fake-gpt").prompt("hi")
    response.text()
    assert llm_token_costs.cost_of(response) == pytest.approx(expected_gpt4o_mini())


# -- legacy logs.db (llm before 0.32) ---------------------------------------------


def make_legacy_db(path, with_turns_duplicate=False):
    db = sqlite_utils.Database(path)
    db["conversations"].insert(
        {"id": "c1", "name": "old chat", "model": "gpt-4o"}, pk="id"
    )
    db["responses"].insert_all(
        [
            {
                "id": "r1",
                "model": "gpt-4o",
                "prompt": "x",
                "response": "y",
                "conversation_id": "c1",
                "datetime_utc": "2025-03-01T10:00:00.000000",
                "input_tokens": 1_000_000,
                "output_tokens": 0,
                "token_details": None,
            },
            {
                "id": "r2",
                "model": "gpt-4o",
                "prompt": "x",
                "response": "y",
                "conversation_id": "c1",
                "datetime_utc": "2025-03-02T10:00:00.000000",
                "input_tokens": None,
                "output_tokens": None,
                "token_details": None,
            },
        ],
        pk="id",
    )
    if with_turns_duplicate:
        db["threads"].insert({"id": "c1", "name": "old chat"}, pk="id")
        db["turns"].insert(
            {
                "id": "r1",
                "thread_id": "c1",
                "model": "gpt-4o",
                "resolved_model": None,
                "input_tokens": 1_000_000,
                "output_tokens": 0,
                "token_details": None,
                "datetime_utc": "2025-03-01T10:00:00+00:00",
            },
            pk="id",
        )
    return db


@pytest.mark.parametrize("dual", [False, True])
def test_legacy_responses_table(tmp_path, dual):
    path = tmp_path / "old.db"
    make_legacy_db(path, with_turns_duplicate=dual)
    data = json.loads(run("costs", "-d", str(path), "--json", "--by", "conversation"))
    (group,) = data["groups"]
    assert group["label"] == "old chat"
    assert group["responses"] == 2
    assert group["responses_without_token_counts"] == 1
    assert group["cost_usd"] == pytest.approx(llm_prices.lookup("gpt-4o").input)
    assert data["models_without_token_counts"] == {"gpt-4o": 1}


def test_missing_database(tmp_path):
    result = CliRunner().invoke(cli, ["costs", "-d", str(tmp_path / "nope.db")])
    assert result.exit_code == 1
    assert "No logs database" in result.output


# -- estimate --------------------------------------------------------------------------


def test_estimate_given_counts():
    data = json.loads(
        run(
            "costs",
            "estimate",
            "-m",
            "gpt-4o-mini",
            "-m",
            "claude-opus-4.5",
            "--input-tokens",
            "10000",
            "--output-tokens",
            "1000",
            "--json",
        )
    )
    by_model = {e["model"]: e for e in data["estimates"]}
    p = llm_prices.lookup("gpt-4o-mini")
    assert by_model["gpt-4o-mini"]["cost_usd"] == pytest.approx(
        (10_000 * p.input + 1_000 * p.output) / 1_000_000
    )
    assert by_model["claude-opus-4.5"]["price_row"].startswith(
        "anthropic/claude-opus-4-5"
    )


def test_estimate_counts_stdin_and_uses_log_median(fake_models):
    for _ in range(3):
        run("prompt", "-m", "fake-gpt", "hi")
    data = json.loads(
        run("costs", "estimate", "-m", "fake-gpt", "--json", input="a" * 400)
    )
    (e,) = data["estimates"]
    assert e["input_tokens"] > 0
    assert e["output_tokens"] == 200
    assert e["output_basis"] == "median in your logs"
    assert e["price_row"] == "openai/gpt-4o-mini"


def test_price_lists_installed_models():
    out = run("costs", "price")
    assert "gpt-4o-mini" in out
    assert "openai/gpt-4o-mini" in out
