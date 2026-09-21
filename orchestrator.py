"""Optional, standalone LLM refinement for one visual prompt.

Importing this module does not read configuration, create sockets, or make HTTP
requests. The live transcriber intentionally does not import or invoke it.
"""

import argparse
import time
from dataclasses import dataclass

import requests

from osc_output import NullOutputPublisher, OscOutputPublisher
from runtime_config import (
    ConfigError,
    RuntimeConfig,
    format_config_error,
    load_env_file,
)
from runtime_logging import RuntimeLogSession


SUPPORTED_PROVIDERS = {"capriole", "ollama"}


class PromptProviderError(RuntimeError):
    """A provider returned an unusable response or could not be reached."""


@dataclass(frozen=True)
class PromptRoute:
    provider: str
    model: str

    @classmethod
    def parse(cls, value):
        provider, separator, model = str(value).partition(":")
        provider = provider.strip().lower()
        model = model.strip()
        if not separator or not provider or not model:
            raise ValueError("prompt routes must use provider:model")
        if provider not in SUPPORTED_PROVIDERS:
            choices = ", ".join(sorted(SUPPORTED_PROVIDERS))
            raise ValueError(f"prompt provider must be one of: {choices}")
        return cls(provider=provider, model=model)


class OllamaPromptProvider:
    name = "ollama"

    def __init__(self, session, endpoint, timeout):
        self.session = session
        self.endpoint = endpoint
        self.timeout = timeout

    def generate(self, model, prompt):
        try:
            response = self.session.post(
                self.endpoint,
                json={"model": model, "prompt": prompt, "stream": False},
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise PromptProviderError("Ollama request failed") from exc

        if not isinstance(payload, dict):
            raise PromptProviderError("Ollama response must be a JSON object")
        result = payload.get("response")
        if not isinstance(result, str) or not result.strip():
            raise PromptProviderError("Ollama response did not contain prompt text")
        return result


class CapriolePromptProvider:
    name = "capriole"

    def __init__(self, session, endpoint, api_key, timeout):
        if not RuntimeConfig.is_secret_configured(api_key):
            raise ValueError("Capriole requires a configured API key")
        self.session = session
        self.endpoint = endpoint
        self.api_key = api_key
        self.timeout = timeout

    def generate(self, model, prompt):
        try:
            response = self.session.post(
                self.endpoint,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={"model": model, "input": prompt},
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise PromptProviderError("Capriole request failed") from exc

        result = self._extract_text(payload)
        if not isinstance(result, str) or not result.strip():
            raise PromptProviderError(
                "Capriole response did not contain prompt text"
            )
        return result

    @staticmethod
    def _extract_text(payload):
        if not isinstance(payload, dict):
            return None
        for name in ("output", "response"):
            value = payload.get(name)
            if isinstance(value, str) and value.strip():
                return value
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            return None
        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            return None
        message = first_choice.get("message")
        if not isinstance(message, dict):
            return None
        return message.get("content")


class MultiLLMClient:
    def __init__(
        self,
        routes,
        providers,
        *,
        target_tokens=70,
        max_output_characters=1000,
        logger=None,
        owned_session=None,
        clock=None,
    ):
        self.routes = tuple(routes)
        self.providers = dict(providers)
        self.target_tokens = target_tokens
        self.max_output_characters = max_output_characters
        self.logger = logger
        self.owned_session = owned_session
        self.clock = clock or time.monotonic
        self.closed = False

        if not self.routes:
            raise ValueError("at least one prompt route is required")
        if target_tokens <= 0 or max_output_characters <= 0:
            raise ValueError("prompt refinement limits must be positive")
        missing = {
            route.provider
            for route in self.routes
            if route.provider not in self.providers
        }
        if missing:
            raise ValueError(
                "missing prompt providers: " + ", ".join(sorted(missing))
            )

    def generate_prompt(self, raw_speech, scene_context=""):
        raw_speech = " ".join(str(raw_speech).split())
        scene_context = " ".join(str(scene_context).split())
        if not raw_speech:
            return None

        prompt = self._build_input(raw_speech, scene_context)
        for attempt, route in enumerate(self.routes, start=1):
            started_at = self.clock()
            self._log(
                "info",
                "Trying prompt refinement provider",
                event="prompt_refinement_attempt",
                provider=route.provider,
                model=route.model,
                attempt=attempt,
            )
            try:
                result = self.providers[route.provider].generate(
                    route.model,
                    prompt,
                )
                result = self._normalize_output(result)
            except Exception as exc:
                self._log(
                    "warning",
                    "Prompt refinement provider failed",
                    event="prompt_refinement_failed",
                    provider=route.provider,
                    model=route.model,
                    attempt=attempt,
                    error_type=type(exc).__name__,
                )
                continue

            self._log(
                "info",
                "Prompt refinement completed",
                event="prompt_refinement_completed",
                provider=route.provider,
                model=route.model,
                attempt=attempt,
                latency_seconds=round(self.clock() - started_at, 4),
                output_characters=len(result),
            )
            return result

        self._log(
            "warning",
            "All prompt refinement providers failed",
            event="prompt_refinement_exhausted",
            attempts=len(self.routes),
        )
        return None

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.owned_session is not None:
            self.owned_session.close()

    def _build_input(self, raw_speech, scene_context):
        context = scene_context or "none"
        return (
            "You are a visual prompt engineer for SDXL. "
            f"Use at most {self.target_tokens} tokens. "
            "Return only one descriptive prompt with no commentary.\n"
            f"Current scene context: {context}\n"
            f"New speech input: {raw_speech}"
        )

    def _normalize_output(self, value):
        if not isinstance(value, str):
            raise PromptProviderError("provider output must be text")
        result = " ".join(value.split())
        if not result:
            raise PromptProviderError("provider output was empty")
        if len(result) > self.max_output_characters:
            raise PromptProviderError("provider output exceeded the safety limit")
        return result

    def _log(self, level, message, **extra):
        if self.logger is not None:
            getattr(self.logger, level)(message, extra=extra)


def create_prompt_client(config, *, session=None, logger=None):
    routes = tuple(
        PromptRoute.parse(value) for value in config.prompt_refinement_chain
    )
    active_providers = {route.provider for route in routes}
    owned_session = None
    if session is None:
        session = requests.Session()
        owned_session = session

    try:
        providers = {}
        if "ollama" in active_providers:
            providers["ollama"] = OllamaPromptProvider(
                session,
                config.prompt_refinement_ollama_endpoint,
                config.prompt_refinement_request_timeout,
            )
        if "capriole" in active_providers:
            providers["capriole"] = CapriolePromptProvider(
                session,
                config.prompt_refinement_capriole_endpoint,
                config.capriole_api_key,
                config.prompt_refinement_request_timeout,
            )

        return MultiLLMClient(
            routes,
            providers,
            target_tokens=config.prompt_refinement_target_tokens,
            max_output_characters=(
                config.prompt_refinement_max_output_characters
            ),
            logger=logger,
            owned_session=owned_session,
        )
    except Exception:
        if owned_session is not None:
            owned_session.close()
        raise


class PromptOrchestrator:
    def __init__(self, llm_client, output_publisher, *, logger=None):
        self.llm_client = llm_client
        self.output_publisher = output_publisher
        self.logger = logger
        self.last_prompt = ""
        self.closed = False

    def refine_and_send(self, raw_text, scene_context=""):
        raw_text = " ".join(str(raw_text).split())
        if not raw_text:
            return None

        context = " ".join(str(scene_context).split()) or self.last_prompt
        refined_prompt = self.llm_client.generate_prompt(raw_text, context)
        if not refined_prompt:
            refined_prompt = f"{raw_text}, cinematic, 8k"
            self._log(
                "warning",
                "Using deterministic prompt fallback",
                event="prompt_refinement_fallback",
                input_characters=len(raw_text),
            )

        if refined_prompt == self.last_prompt:
            self._log(
                "debug",
                "Duplicate refined prompt suppressed",
                event="prompt_refinement_duplicate",
            )
            return refined_prompt

        try:
            delivered = self.output_publisher.send("/prompt", refined_prompt)
        except Exception as exc:
            delivered = False
            self._log(
                "warning",
                "Prompt output failed",
                event="prompt_output_failed",
                error_type=type(exc).__name__,
            )

        if delivered:
            self.last_prompt = refined_prompt
            self._log(
                "info",
                "Refined prompt published",
                event="prompt_refinement_published",
                output_characters=len(refined_prompt),
            )
        else:
            self._log(
                "warning",
                "Refined prompt was not delivered",
                event="prompt_output_degraded",
            )
        return refined_prompt

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            self.output_publisher.close()
        finally:
            self.llm_client.close()

    def _log(self, level, message, **extra):
        if self.logger is not None:
            getattr(self.logger, level)(message, extra=extra)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Refine one text prompt through an opt-in LLM fallback chain."
    )
    parser.add_argument("text", help="Speech text to convert into an SDXL prompt.")
    parser.add_argument(
        "--context",
        default="",
        help="Optional scene context supplied to the refinement provider.",
    )
    parser.add_argument(
        "--no-osc",
        action="store_true",
        help="Print the refined prompt without sending it to TouchDesigner.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    load_env_file()
    try:
        config = RuntimeConfig.from_environment()
    except ConfigError as exc:
        print(format_config_error(exc))
        return 2

    try:
        log_session = RuntimeLogSession(config)
    except RuntimeError as exc:
        print(f"Could not initialize runtime logging: {exc}")
        return 2

    logger = log_session.logger("orchestrator")
    publisher = None
    orchestrator = None
    try:
        publisher = (
            NullOutputPublisher()
            if args.no_osc
            else OscOutputPublisher(
                config.osc_ip,
                config.osc_port,
                status_interval=config.osc_status_interval,
                error_log_interval=config.osc_output_error_log_interval,
                prompt_retry_base_seconds=config.osc_prompt_retry_base_seconds,
                prompt_retry_max_seconds=config.osc_prompt_retry_max_seconds,
                logger=log_session.logger("osc"),
            )
        )
        llm_client = create_prompt_client(config, logger=logger)
        orchestrator = PromptOrchestrator(
            llm_client,
            publisher,
            logger=logger,
        )
        result = orchestrator.refine_and_send(args.text, args.context)
        if result is None:
            print("Prompt text must not be empty.")
            return 2
        print(f"[REFINED PROMPT] {result}")
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Could not initialize prompt refinement: {exc}")
        return 2
    finally:
        if orchestrator is not None:
            orchestrator.close()
        elif publisher is not None:
            publisher.close()
        log_session.close()


if __name__ == "__main__":
    raise SystemExit(main())
