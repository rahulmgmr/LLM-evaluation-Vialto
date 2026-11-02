"""NeMo Guardrails runtime for input and output moderation."""

from pathlib import Path
from typing import Any

from nemoguardrails import LLMRails, RailsConfig


GUARDRAILS_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config" / "nemo_guardrails"


class NeMoGuardrailsRunner:
    """Generate answers through NeMo so both input and output rails run."""

    def __init__(self, model_name: str, temperature: float = 0.0, max_tokens: int = 512):
        if not model_name:
            raise ValueError("A SUT model_name is required to initialize NeMo Guardrails.")

        self.config = RailsConfig.from_path(str(GUARDRAILS_CONFIG_DIR))
        main_models = [model for model in self.config.models if model.type == "main"]
        if not main_models:
            raise ValueError("NeMo Guardrails config must define a model with type 'main'.")

        # Keep the SUT model and generation settings in config/config.json as
        # the single source of truth; config.yml supplies provider and rail policy.
        for model in main_models:
            model.model = model_name
            model.parameters["temperature"] = temperature
            model.parameters["max_tokens"] = max_tokens

        self.rails = LLMRails(self.config)

    async def generate(self, user_input: str) -> dict[str, Any]:
        """Return the assistant response, all activated rails, and violations."""
        result = await self.rails.generate_async(
            messages=[{"role": "user", "content": user_input}],
            options={"log": {"activated_rails": True}},
        )

        response = getattr(result, "response", result)
        if isinstance(response, str):
            generated = response.strip()
        elif isinstance(response, list):
            assistant_messages = [
                message
                for message in response
                if isinstance(message, dict) and message.get("role") == "assistant"
            ]
            generated = str(assistant_messages[-1].get("content", "")).strip() if assistant_messages else ""
        elif isinstance(response, dict):
            generated = str(response.get("content", "")).strip()
        else:
            generated = ""

        if generated.casefold() in {
            "i'm sorry, an internal error has occurred.",
            "i am sorry, an internal error has occurred.",
        }:
            raise RuntimeError(
                "NeMo Guardrails returned its internal-error response; "
                "the response was not scored as an LLM answer. Check NeMo's "
                "runtime logs for the failing rail or model call."
            )

        log = getattr(result, "log", None)
        activated_rails = getattr(log, "activated_rails", []) or []
        activities = []
        failures = []
        for rail in activated_rails:
            decisions = list(getattr(rail, "decisions", []) or [])
            stopped = bool(getattr(rail, "stop", False))
            name = str(getattr(rail, "name", "unknown rail"))
            rail_type = str(getattr(rail, "type", "unknown"))
            # NeMo logs a generic 'stop' when a rail flow aborts, which can
            # also indicate an action/runtime error. Only count explicit policy
            # outcomes as triggers; keep the stop flag for troubleshooting.
            triggered = any(
                any(marker in decision.lower() for marker in ("refuse", "block", "exception", "disallow"))
                for decision in decisions
            )
            activity = {
                "type": rail_type,
                "name": name,
                "decisions": decisions,
                "stopped": stopped,
                "triggered": triggered,
            }
            activities.append(activity)
            if triggered:
                detail = ", ".join(decisions) if decisions else "rail stopped generation"
                failures.append(f"NeMo {rail_type} rail '{name}' triggered: {detail}")

        # An input/output rail may intentionally stop generation without
        # returning an assistant message. Keep that as an observable result so
        # the evaluator can distinguish an expected block from a runtime error.
        if not generated and not any(a["triggered"] and a["stopped"] for a in activities):
            raise RuntimeError("NeMo Guardrails returned no assistant response and no blocking rail was recorded.")

        return {
            "generated": generated,
            "rail_activity": activities,
            "guardrails": failures,
        }
