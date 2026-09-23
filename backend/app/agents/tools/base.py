"""Uniform tool interface and result envelope for the agent layer (brief §17, §18).

Every capability the router can dispatch — land-cover classification, change detection, SAR
analysis, optical/SAR fusion, region grounding, VQA, captioning — is wrapped as a :class:`Tool`
exposing the same five-method contract: :meth:`~Tool.validate_input`, :meth:`~Tool.preprocess`,
:meth:`~Tool.predict`, :meth:`~Tool.postprocess`, :meth:`~Tool.explain`. The orchestrator runs any
tool through :meth:`~Tool.run` without knowing what is inside it, and the registry (§18) can swap
one implementation for another behind the same task.

Two honesty commitments shape this layer:

1. **The tier ladder is real, and nothing claims a model it does not have (§6, §18).** Each tool
   declares a :class:`ToolTier`. The tools shipped in this prototype are deterministic classical
   computer-vision / remote-sensing analysers, so they register at :attr:`ToolTier.CLASSICAL`. The
   ``PREFERRED`` / ``FALLBACK`` rungs exist so a trained model can be slotted in later *without
   changing any caller* — but until real weights are present, no tool advertises itself as learned
   or fine-tuned. A trace that says "classical" is telling the truth about what ran.

2. **A tool measures and describes; it never decides to hide a weak result.**
   :meth:`~Tool.postprocess` attaches an evidence-based :class:`~app.services.confidence.ConfidenceReport`
   (§27) computed from the analysis, and phrases its ``answer`` from those measurements. Whether an
   ``INSUFFICIENT`` result is converted into an explicit decline (§28) is the orchestrator's
   evidence-validator step, kept in one place rather than reimplemented in every tool. A tool with a
   weak result still returns it honestly, flagged ``WARNING``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING, Any, ClassVar

from ...core.errors import ErrorCode, ToolContractError, ValidationError
from ...core.logging import get_logger
from ...core.types import ConfidenceLevel, InputMode, Modality, QueryTask, StepStatus

if TYPE_CHECKING:  # annotations only — keeps this base module import-light
    from ...agents.modes import NormalizedRequest
    from ...geospatial.raster import RasterData
    from ...geospatial.validate import InputQuality
    from ...services.confidence import ConfidenceReport

logger = get_logger(__name__)


class ToolTier(IntEnum):
    """Where a tool sits in the fallback chain (§18); lower values are tried first.

    The integer order *is* the try-order, so the orchestrator can sort a chain by tier and run
    the most capable available implementation. Serialised by :attr:`label`.
    """

    PREFERRED = 0
    """A task-specific trained/fine-tuned model. None ship in this prototype (§6)."""

    FALLBACK = 1
    """A lighter learned model used when the preferred one is unavailable."""

    CLASSICAL = 2
    """A deterministic classical-CV / remote-sensing analyser. What this prototype ships."""

    @property
    def label(self) -> str:
        return self.name.lower()


@dataclass
class ToolContext:
    """Inputs available to a tool for one analysis: the query text and the loaded rasters.

    Rasters are already read into memory by the caller (the API upload layer, or a test), so a tool
    never touches the filesystem or trusts a path. Order is meaningful for the pair tasks: for
    bi-temporal change it is ``(before, after)``. Establishing that order is the caller's job, since
    only it knows the acquisition dates — the tools and the router treat the sequence as given.
    """

    query: str
    rasters: list[RasterData] = field(default_factory=list)

    def expect_count(self, n: int, task: QueryTask) -> None:
        """Guard that exactly ``n`` rasters are present, with a task-specific error otherwise.

        The router validates image counts before dispatch, but a tool must still fail loudly when
        run directly rather than index off the end of a short list. Too few and too many are
        distinct error codes because the fix differs: supply another image versus remove one.
        """
        got = len(self.rasters)
        if got == n:
            return
        if got < n:
            raise ValidationError(
                f"This analysis needs {n} image(s) but received {got}.",
                code=ErrorCode.MISSING_SECOND_IMAGE,
                context={"task": task.value, "expected": n, "received": got},
            )
        raise ValidationError(
            f"This analysis works on {n} image(s) but received {got}.",
            code=ErrorCode.TOO_MANY_IMAGES,
            context={"task": task.value, "expected": n, "received": got},
        )


@dataclass
class Prepared:
    """Inputs a tool has validated and readied for prediction.

    A deliberately small shared container so single-image tools need no bespoke type. Pair tools
    stash the aligned second raster, the compatibility verdict and the alignment report in
    :attr:`context` rather than widening this shape for everyone.
    """

    rasters: list[RasterData]
    quality: InputQuality | None = None
    context: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolResult:
    """One tool's output: a concise answer, the structured data behind it, and its confidence.

    Attributes:
        task: The :class:`QueryTask` this result answers.
        tool_name / tier: Which implementation produced it, and at what rung of the fallback
            chain — surfaced in the trace so "classical" is visible rather than implied.
        answer: A concise, deterministic, evidence-grounded statement of the finding. Never a
            language-model generation; every number in it came from the analysis.
        data: The full serialisable result (the service's ``to_dict()``), for the result page.
        confidence: The evidence-based :class:`~app.services.confidence.ConfidenceReport` (§27).
        evidence: Human-readable lines from :meth:`Tool.explain`, shown as the "why".
        warnings: Analysis warnings carried up for the user.
        status: :attr:`StepStatus.OK`, or :attr:`StepStatus.WARNING` when confidence is
            insufficient or warnings were raised. A raising tool is recorded ``FAILED`` by the
            orchestrator, not here.
        raw: The domain result object (may hold numpy arrays, e.g. class maps for rendering).
            Kept off :meth:`to_dict` so serialisation stays clean and small.
    """

    task: QueryTask
    tool_name: str
    tier: ToolTier
    answer: str
    data: dict[str, Any]
    confidence: ConfidenceReport
    evidence: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    status: StepStatus = StepStatus.OK
    raw: Any = None

    @property
    def is_sufficient(self) -> bool:
        return self.confidence.is_sufficient

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task.value,
            "tool": self.tool_name,
            "tier": self.tier.label,
            "status": self.status.value,
            "answer": self.answer,
            "confidence": self.confidence.to_dict(),
            "evidence": list(self.evidence),
            "warnings": list(self.warnings),
            "data": self.data,
        }


# How a contract refusal names each task in user-facing text. Keyed by task rather than by tool
# because the user asked for a capability, not for a particular implementation of it.
_TASK_PHRASES: dict[QueryTask, str] = {
    QueryTask.LAND_COVER_ANALYSIS: "land-cover analysis",
    QueryTask.VQA: "question-answering",
    QueryTask.CAPTIONING: "scene captioning",
    QueryTask.GROUNDING: "region grounding",
    QueryTask.CHANGE_DETECTION: "change detection",
    QueryTask.CHANGE_VQA: "change question-answering",
    QueryTask.OPTICAL_SAR_ANALYSIS: "optical/SAR analysis",
}


def _status_for(confidence: ConfidenceReport, warnings: list[str]) -> StepStatus:
    """A tool's own step status: WARNING on insufficient evidence or any warning, else OK."""
    if confidence.level is ConfidenceLevel.INSUFFICIENT or warnings:
        return StepStatus.WARNING
    return StepStatus.OK


class Tool(ABC):
    """The uniform interface every dispatchable capability implements (§17).

    Subclasses set the three class attributes and the five methods. :meth:`run` is the template
    that composes them in order; the orchestrator calls only :meth:`run` (and reads the class
    attributes via :meth:`describe`). Keeping the five steps separate is what lets a caller, or a
    test, exercise validation without prediction, or swap a preprocessing step, without the method
    contract changing.
    """

    #: The task this tool implements. Set by every subclass.
    task: ClassVar[QueryTask]
    #: Stable identifier for logs, the trace, and ``/api/models``.
    name: ClassVar[str]
    #: Rung in the fallback chain. Classical for everything shipped here.
    tier: ClassVar[ToolTier] = ToolTier.CLASSICAL
    #: One-line human description for the model registry endpoint.
    summary: ClassVar[str] = ""

    # --- declared input contract (§6) -------------------------------------------------
    # These four attributes are what makes "2 images sent to a 1-image analysis" a rejected
    # *request* rather than a zero-confidence *result*. They are read by
    # :meth:`validate_request`, which the registry runs before any tool is executed.
    #
    # The defaults are permissive so that adding the contract does not silently change the
    # behaviour of a tool that has not declared one; every tool shipped here declares all four.

    #: Canonical modes this tool accepts. Empty means "no mode restriction".
    supported_modes: ClassVar[tuple[InputMode, ...]] = ()
    #: Fewest inputs the analysis is defined for.
    min_images: ClassVar[int] = 1
    #: Most inputs the analysis is defined for.
    max_images: ClassVar[int] = 2
    #: Modalities that must all be present. Empty means "any modality".
    required_modalities: ClassVar[frozenset[Modality]] = frozenset()

    def validate_request(self, request: NormalizedRequest) -> None:
        """Check a :class:`~app.agents.modes.NormalizedRequest` against this tool's contract.

        Runs **before** :meth:`run`, so a mismatch is reported as
        :class:`~app.core.errors.ToolContractError` (``TOOL_INPUT_CONTRACT_ERROR``) and the
        analysis never starts. This is the distinction the brief insists on: an application-level
        contract failure is not weak evidence, and must not be reported as
        ``INSUFFICIENT_EVIDENCE``.

        The three checks are ordered cheapest-and-most-specific first — count, then mode, then
        modality — so the message names the one thing the user should change.

        Raises:
            ToolContractError: the request does not satisfy the declared contract.
        """
        count = request.input_count

        if count < self.min_images:
            raise ToolContractError(
                f"{self._contract_subject()} needs {self._count_phrase()}, but received {count}."
                + (" Upload a second image to compare." if self.min_images == 2 else ""),
                context=self._contract_context(request, "min_images"),
            )
        if count > self.max_images:
            raise ToolContractError(
                f"{self._contract_subject()} works on {self._count_phrase()}, but received {count}."
                + (
                    " Select a single image, or ask for a comparison instead."
                    if self.max_images == 1
                    else ""
                ),
                context=self._contract_context(request, "max_images"),
            )
        if self.supported_modes and request.mode not in self.supported_modes:
            accepted = ", ".join(m.value for m in self.supported_modes)
            raise ToolContractError(
                f"{self._contract_subject()} cannot run on a '{request.mode.value}' input "
                f"configuration. It accepts: {accepted}.",
                context=self._contract_context(request, "supported_modes"),
            )
        missing = self.required_modalities - request.modality_set()
        if missing:
            need = ", ".join(sorted(m.value for m in missing))
            raise ToolContractError(
                f"{self._contract_subject()} requires {need} imagery, which is not among the "
                f"inputs provided ({', '.join(m.value for m in request.modalities)}).",
                context=self._contract_context(request, "required_modalities"),
            )

    def _contract_subject(self) -> str:
        """How the contract messages name this tool to the user.

        Derived from the task rather than the tool id, because the user asked for an analysis, not
        for ``sar-backscatter-cv``. The suffix guard stops "optical_sar_analysis" from rendering as
        "the optical sar analysis analysis".
        """
        phrase = _TASK_PHRASES.get(self.task, self.task.value.replace("_", " "))
        if phrase.endswith("analysis"):
            return f"The {phrase}"
        return f"The {phrase} analysis"

    def _count_phrase(self) -> str:
        if self.min_images == self.max_images:
            return f"exactly {self.min_images} image(s)"
        return f"between {self.min_images} and {self.max_images} images"

    def _contract_context(self, request: NormalizedRequest, violated: str) -> dict[str, Any]:
        """Log-only technical detail: what was declared, what arrived, and which rule failed."""
        return {
            "tool": self.name,
            "task": self.task.value,
            "violated": violated,
            "contract": self.describe_contract(),
            "received": {
                "mode": request.mode.value,
                "input_count": request.input_count,
                "modalities": [m.value for m in request.modalities],
            },
        }

    def describe_contract(self) -> dict[str, Any]:
        """The declared input contract, serialised for ``/api/models`` and the trace.

        Kept separate from :meth:`describe` on purpose: ``describe()`` is a fixed four-key shape
        that callers and tests compare by equality, so contract metadata is additive here rather
        than a breaking change there.
        """
        return {
            "supported_modes": [m.value for m in self.supported_modes],
            "min_images": self.min_images,
            "max_images": self.max_images,
            "required_modalities": sorted(m.value for m in self.required_modalities),
        }

    @abstractmethod
    def validate_input(self, ctx: ToolContext) -> None:
        """Raise an :class:`~app.core.errors.AppError` if ``ctx`` is not usable by this tool.

        Modality, image count and any hard precondition are checked here, before any work, so a
        refusal is specific and cheap. Returns ``None`` when the inputs are acceptable.
        """
        ...

    @abstractmethod
    def preprocess(self, ctx: ToolContext) -> Prepared:
        """Ready the validated inputs: assess quality, align a pair, select bands."""
        ...

    @abstractmethod
    def predict(self, prepared: Prepared) -> Any:
        """Run the underlying analysis and return its raw domain result."""
        ...

    @abstractmethod
    def postprocess(self, raw: Any, prepared: Prepared) -> ToolResult:
        """Wrap a raw result in a :class:`ToolResult`: compute confidence, summarise, explain."""
        ...

    @abstractmethod
    def explain(self, raw: Any) -> list[str]:
        """Human-readable evidence lines for the raw result — the user-facing "why"."""
        ...

    def available(self) -> bool:
        """Whether this tool can run right now — the rung's presence check for the chain (§18).

        Classical tools are always available: they carry no weights. A learned tool at the
        ``PREFERRED`` or ``FALLBACK`` rung overrides this to report that its model is not loaded, and
        the registry skips it and falls through to the next tier. Defaulting to ``True`` means a
        model can be slotted in later without the base contract, or any existing tool, changing.
        """
        return True

    def describe_runtime(self) -> dict[str, Any] | None:
        """Live provenance for a tool backed by a real artefact — checkpoint path, mode, metrics.

        ``None`` for the deterministic analysers: they have no checkpoint, and inventing an empty
        one would put a "model" field on a tool that is pure arithmetic. A learned tool overrides
        this and *probes* the artefact on every call, so ``/api/models`` reports what is on disk now
        rather than what was true at import time (§26). Additive, like :meth:`describe_contract`.
        """
        return None

    def run(self, ctx: ToolContext) -> ToolResult:
        """Execute the full pipeline: validate → preprocess → predict → postprocess.

        Exceptions propagate unchanged; the orchestrator's fallback chain decides whether to try
        the next tier or surface a refusal. This method never fabricates a result to avoid raising.
        """
        self.validate_input(ctx)
        prepared = self.preprocess(ctx)
        raw = self.predict(prepared)
        result = self.postprocess(raw, prepared)
        logger.debug(
            "tool completed",
            extra={
                "tool": self.name,
                "task": self.task.value,
                "confidence": result.confidence.level.value,
                "status": result.status.value,
            },
        )
        return result

    def build_result(
        self,
        *,
        answer: str,
        data: dict[str, Any],
        confidence: ConfidenceReport,
        evidence: list[str],
        warnings: list[str] | None = None,
        raw: Any = None,
    ) -> ToolResult:
        """Assemble a :class:`ToolResult`, deriving its status from confidence and warnings.

        A convenience so every :meth:`postprocess` reports status consistently instead of each
        tool re-deriving the OK/WARNING rule.
        """
        warns = list(warnings or [])
        return ToolResult(
            task=self.task,
            tool_name=self.name,
            tier=self.tier,
            answer=answer,
            data=data,
            confidence=confidence,
            evidence=list(evidence),
            warnings=warns,
            status=_status_for(confidence, warns),
            raw=raw,
        )

    def describe(self) -> dict[str, Any]:
        """Registry/`/api/models` metadata for this tool."""
        return {
            "task": self.task.value,
            "name": self.name,
            "tier": self.tier.label,
            "summary": self.summary,
        }
