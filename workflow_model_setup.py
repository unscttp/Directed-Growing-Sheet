"""Choosing the model: the one question the one-step setup asks.

``python install.py`` (or ``python quickstart.py --setup``) runs ``setup_model``:
the user picks a service from the list in ``presets/models.json`` (Gemini 4
Hydrogen is the default there; edit the file to change the offer),
enters the API key if the service needs one, and picks a model. The choice is
checked against the models the key can use and with one small test request,
then saved to ``.dgs/config.json`` next to this file (git-ignored, readable only
by the user). ``quickstart.py`` reads it from there; the environment variables
``DGS_API_KEY``, ``DGS_MODEL`` and ``DGS_BASE_URL`` override it.
"""

from __future__ import annotations

import getpass
import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from workflow_schema import validate_document
from workflow_types import JsonDict

CONFIG_PATH = Path(__file__).with_name(".dgs") / "config.json"
CHECK_PROMPT = "Reply with the single word OK."
MAX_LISTED = 15


MODELS_PRESET = Path(__file__).with_name("presets") / "models.json"


@dataclass(frozen=True)
class Service:
    """One entry of ``presets/models.json``."""

    id: str
    label: str
    base_url: str | None  # None: ask
    model: str | None  # suggested model ID; None: choose from the list
    needs_key: bool
    key_hint: str = ""
    # "chat": Chat Completions (openai_compatible); "responses": OpenAI's Responses
    # API (openai_responses), which OpenAI's newest models need for tool calls.
    api: str = "chat"


def load_services(path: str | Path | None = None) -> tuple[list[Service], int]:
    """The services the setup offers, and the index of the default one."""
    document = json.loads(Path(path or MODELS_PRESET).read_text("utf-8"))
    validate_document(document, "model_services")
    services = [Service(**entry) for entry in document["services"]]
    ids = [service.id for service in services]
    if len(set(ids)) != len(ids):
        raise ValueError("Service ids in the models preset must be unique.")
    if document["default"] not in ids:
        raise ValueError(f"The default service {document['default']!r} is not listed.")
    return services, ids.index(document["default"])


# -- the saved choice --------------------------------------------------------


def load_config(path: Path | None = None) -> JsonDict:
    """The saved choice, with ``DGS_*`` environment variables taking precedence."""
    path = path or CONFIG_PATH
    config: JsonDict = {}
    try:
        config = json.loads(path.read_text("utf-8"))
    except (OSError, ValueError):
        config = {}
    for key, variable in (
        ("api_key", "DGS_API_KEY"),
        ("model", "DGS_MODEL"),
        ("base_url", "DGS_BASE_URL"),
    ):
        if os.environ.get(variable):
            config[key] = os.environ[variable]
    return config


def save_config(config: JsonDict, path: Path | None = None) -> Path:
    path = path or CONFIG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2), "utf-8")
    try:
        path.chmod(0o600)  # it holds the API key
    except OSError:
        pass
    return path


# -- talking to the service --------------------------------------------------


def _request(url: str, api_key: str | None, body: JsonDict | None = None, timeout: float = 30):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    data = json.dumps(body).encode("utf-8") if body is not None else None
    request = urllib.request.Request(
        url, data=data, headers=headers, method="POST" if body else "GET"
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _message(text: str) -> str:
    """The provider's own message from an error body, on one line."""
    try:
        data = json.loads(text)
    except ValueError:
        return " ".join(text.split())[:200]
    if isinstance(data, list) and data:
        data = data[0]
    error = data.get("error", data) if isinstance(data, dict) else data
    if isinstance(error, dict):
        return str(error.get("message") or error)[:200]
    return str(error)[:200]


def key_problem(problem: str) -> bool:
    """Whether a failed check points at the key rather than the model."""
    lowered = problem.lower()
    return "http 401" in lowered or "http 403" in lowered or "api key" in lowered


def _explain(exc: Exception, api_key: str | None) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        text = _message(exc.read().decode("utf-8", "replace"))
        if api_key:
            text = text.replace(api_key, "[key]")
        return f"HTTP {exc.code}: {text}"
    if isinstance(exc, urllib.error.URLError):
        return f"could not connect ({exc.reason})"
    return f"{type(exc).__name__}: {exc}"


def list_models(base_url: str, api_key: str | None) -> list[str]:
    """Model IDs the key can use, as the service reports them (empty if it cannot)."""
    try:
        data = _request(base_url.rstrip("/") + "/models", api_key)
    except Exception:
        return []
    ids = [str(item.get("id", "")).removeprefix("models/") for item in data.get("data") or []]
    return sorted({model_id for model_id in ids if model_id})


def check_model(base_url: str, api_key: str | None, model: str, api: str = "chat") -> str | None:
    """Send one tiny request; return None if it worked, else why not."""
    try:
        if api == "responses":
            body = {"model": model, "input": CHECK_PROMPT, "store": False}
            reply = _request(base_url.rstrip("/") + "/responses", api_key, body, timeout=60)
            reply["output"]
        else:
            body = {"model": model, "messages": [{"role": "user", "content": CHECK_PROMPT}]}
            reply = _request(base_url.rstrip("/") + "/chat/completions", api_key, body, timeout=60)
            reply["choices"][0]["message"]
    except Exception as exc:
        return _explain(exc, api_key)
    return None


# -- the questions -----------------------------------------------------------


def _pick_model(service: Service, available: list[str], ask, say) -> str:
    suggested = service.model
    if suggested and (not available or suggested in available):
        answer = ask(f"Model [{suggested}]: ").strip()
        return answer or suggested
    if suggested:
        say(f"{suggested} is not among the models this key can use.")
    if available:
        family = (suggested or "").rsplit("-", 1)[0]
        shown = [m for m in available if family and m.startswith(family)] or available
        shown = shown[:MAX_LISTED]
        say("Models this key can use:")
        for number, model_id in enumerate(shown, 1):
            say(f"  {number}) {model_id}")
        while True:
            answer = ask("Choose a number or type a model ID: ").strip()
            if answer.isdigit() and 1 <= int(answer) <= len(shown):
                return shown[int(answer) - 1]
            if answer:
                return answer
    while True:
        answer = ask("Model ID: ").strip()
        if answer:
            return answer


_ENTER_KEY_AGAIN = object()


def _check_a_model(service, base_url, api_key, ask, say, lister, checker):
    """Pick a model and check it: the model, _ENTER_KEY_AGAIN, or None to stop."""
    while True:
        model = _pick_model(service, lister(base_url, api_key), ask, say)
        say(f"Checking {model} ...")
        problem = checker(base_url, api_key, model, service.api)
        if problem is None:
            return model
        say(f"That did not work: {problem}")
        if service.needs_key and key_problem(problem):
            if ask("The key was not accepted. Enter it again? [Y/n] ").strip().lower() == "n":
                return None
            return _ENTER_KEY_AGAIN
        if ask("Try another model? [Y/n] ").strip().lower() == "n":
            return None
        service = Service(
            service.id, service.label, base_url, None, service.needs_key, api=service.api
        )


def setup_model(
    ask: Callable[[str], str] = input,
    secret: Callable[[str], str] = getpass.getpass,
    say: Callable[[str], None] = print,
    *,
    path: Path | None = None,
    lister: Callable = list_models,
    checker: Callable = check_model,
    services_path: str | Path | None = None,
) -> JsonDict | None:
    """Ask which model to use, check it, save it. Returns the saved choice, or None."""
    services, default_index = load_services(services_path)
    previous = load_config(path)
    say("\nWhich model should the assistant use?")
    for number, service in enumerate(services, 1):
        default = "   (default, press Enter)" if number - 1 == default_index else ""
        say(f"  {number}) {service.label}{default}")
    while True:
        answer = ask(f"Choose 1-{len(services)} [{default_index + 1}]: ").strip()
        if not answer:
            service = services[default_index]
            break
        if answer.isdigit() and 1 <= int(answer) <= len(services):
            service = services[int(answer) - 1]
            break
        say("Please type one of the numbers.")

    base_url = service.base_url
    while not base_url:
        base_url = ask("Base URL of the API (for example https://host/v1): ").strip()

    kept = previous.get("api_key") if previous.get("base_url") == base_url else None
    while True:
        api_key = None
        if service.needs_key:
            hint = " (Enter keeps the saved one)" if kept else ""
            api_key = secret(f"{service.key_hint or 'API key'}{hint}: ").strip() or kept
            if not api_key and service.base_url is not None:
                say("This service needs a key. Run the setup again when you have one.")
                return None
        model = _check_a_model(service, base_url, api_key, ask, say, lister, checker)
        if model is None:
            return None
        if model is not _ENTER_KEY_AGAIN:
            break
        kept = None

    config = {
        "service": service.id,
        "label": service.label,
        "base_url": base_url,
        "model": model,
        "api": service.api,
    }
    if api_key:
        config["api_key"] = api_key
    saved = save_config(config, path)
    say(f"Saved: {model} at {base_url} ({saved}).")
    return config
