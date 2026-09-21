"""Validate and negotiate realtime session configuration."""

import copy
from dataclasses import asdict, dataclass
from typing import cast

from pydantic import ValidationError

from sglang_omni.serve.realtime.schema import (
    GrantedCapabilities,
    JsonObject,
    SessionConfiguration,
    SessionType,
    SessionUpdateRequest,
)
from sglang_omni.serve.realtime.types import Capabilities, ProtocolError, RuntimeLimits


def merge_config(current: JsonObject, patch: JsonObject) -> JsonObject:
    result = copy.deepcopy(current)
    for key, value in patch.items():
        current_value = result.get(key)
        if isinstance(value, dict) and isinstance(current_value, dict):
            result[key] = merge_config(current_value, value)
        else:
            result[key] = copy.deepcopy(value)
    return result


@dataclass(kw_only=True)
class SessionNegotiation:
    model: str
    capabilities: Capabilities
    limits: RuntimeLimits

    def negotiate(
        self, current: SessionConfiguration, state: str, patch: object
    ) -> tuple[SessionConfiguration, GrantedCapabilities]:
        try:
            candidate = SessionUpdateRequest.model_validate(
                {"session": merge_config(cast(JsonObject, current), patch)}
                if isinstance(patch, dict)
                else {"session": patch}
            ).session
        except ValidationError as exc:
            location = ".".join(str(part) for part in exc.errors()[0]["loc"])
            raise ProtocolError(
                "invalid_request", "invalid session configuration", location
            ) from exc
        if state == "OPEN":

            def frozen(config: SessionConfiguration) -> SessionConfiguration:
                result = copy.deepcopy(config)
                result.pop("output_modalities", None)
                audio = result.get("audio", {})
                audio.get("input", {}).pop("turn_detection", None)
                return result

            if frozen(candidate) != frozen(current):
                raise ProtocolError("invalid_state", "session field is frozen")
        if "instructions" in candidate and (
            len(candidate["instructions"]) > self.limits.max_history_chars
        ):
            raise ProtocolError(
                "invalid_request", "instructions exceed context or have invalid type"
            )
        if candidate.get("model", self.model) != self.model:
            raise ProtocolError("invalid_request", "model differs from deployment")
        typ = candidate.get(
            "type",
            (
                "transcription"
                if self.capabilities.interaction == "transcription"
                else "realtime"
            ),
        )
        if typ != (
            "transcription"
            if self.capabilities.interaction == "transcription"
            else "realtime"
        ):
            raise ProtocolError("invalid_request", "session type is unavailable")
        self.validate_audio(candidate)
        requested = candidate.get(
            "output_modalities", list(self.capabilities.output_modalities)
        )
        outputs = [x for x in requested if x in self.capabilities.output_modalities][:1]
        if not outputs:
            raise ProtocolError("invalid_request", "no supported output combination")
        micro = self.validate_extension(candidate)
        return self.grant(candidate, typ, requested, outputs, micro)

    def validate_audio(self, candidate: SessionConfiguration) -> None:
        audio = candidate.get("audio", {})
        for direction, fmt, rate in (
            (
                "input",
                audio.get("input", {}).get("format"),
                self.capabilities.input_rate,
            ),
            (
                "output",
                audio.get("output", {}).get("format"),
                self.capabilities.output_rate,
            ),
        ):
            if fmt is not None and fmt != dict(type="audio/pcm", rate=rate):
                raise ProtocolError(
                    "invalid_request",
                    "unsupported PCM format or sample rate",
                    f"session.audio.{direction}.format",
                )
        if audio.get("input", {}).get("turn_detection") is not None:
            raise ProtocolError(
                "not_applicable", "VAD is only available for turn-based sessions"
            )

    def validate_extension(self, candidate: SessionConfiguration) -> float | None:
        extension = candidate.get("sglang", {})
        if (
            extension.get("tail_policy", self.capabilities.tail_policy)
            != self.capabilities.tail_policy
        ):
            raise ProtocolError("invalid_request", "tail policy is unavailable")
        timebase = extension.get("timebase", {})
        if (
            timebase.get("native_unit_ms", self.capabilities.native_unit_ms)
            != self.capabilities.native_unit_ms
        ):
            raise ProtocolError("invalid_request", "native cadence is fixed")
        micro = timebase.get("microturn_ms")
        return micro

    def grant(
        self,
        candidate: SessionConfiguration,
        typ: SessionType,
        requested: list[str],
        outputs: list[str],
        micro: float | None,
    ) -> tuple[SessionConfiguration, GrantedCapabilities]:
        grant = self.capabilities.describe()
        grant.update(
            {
                "output_modalities": outputs,
                "limits": asdict(self.limits),
                "rejections": [],
            }
        )
        if micro is not None:
            grant["rejections"].append(
                dict(
                    field="sglang.timebase.microturn_ms",
                    requested=micro,
                    reason="external chunks are variable length; no fixed external cadence",
                    granted=None,
                )
            )
        grant["microturn_ms"] = None
        if outputs != requested:
            grant["rejections"].append(
                dict(
                    field="output_modalities",
                    requested=requested,
                    reason="deployment modalities",
                    granted=outputs,
                )
            )
        for direction, rate in (
            ("input", self.capabilities.input_rate),
            ("output", self.capabilities.output_rate),
        ):
            cast(dict[str, JsonObject], candidate.setdefault("audio", {})).setdefault(
                direction, {}
            ).setdefault("format", dict(type="audio/pcm", rate=rate))
        candidate.update(
            {"model": self.model, "type": typ, "output_modalities": outputs}
        )
        return candidate, grant
