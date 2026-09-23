"""Tool registry and fallback chain: each :class:`QueryTask` -> the tool(s) that answer it (§18).

The registry is the single place that knows *which* tool runs for a task, and what to do when a
task is recognised but cannot be honestly served. It sits between the router (which decides the task
from the query and the inputs) and the tools (which do the analysis), and holds three commitments the
rest of the system leans on:

1. **A task maps to an ordered fallback chain, tried most-capable first (§18).** Tools declare a
   :class:`~app.agents.tools.base.ToolTier`; the registry sorts a task's tools by tier so a trained
   model (``PREFERRED``) would be tried ahead of the classical analyser that ships today, and falls
   through to the next rung when a higher one is unavailable — either because its weights are absent
   (:meth:`~app.agents.tools.base.Tool.available` is ``False``) or because it reports a model outage
   at run time. Nothing else in the system changes when a model is slotted in later.

2. **One task can be served by several tools that differ by input, not tier.** Optical/SAR analysis
   is one task answered by two classical tools — :class:`~app.agents.tools.sar.SarTool` for a single
   radar image, :class:`~app.agents.tools.fusion.FusionTool` for an optical+SAR pair. The registry
   picks the tool whose :meth:`~app.agents.tools.base.Tool.validate_input` accepts the inputs, so a
   caller never has to know which of the two applies.

3. **Recognising a task is not the same as being able to do it (§9, §18).** ``OBJECT_DETECTION`` and
   ``SEGMENTATION`` are legitimate questions a user can ask, and the classifier names them honestly,
   but no tool implements them. The registry turns them into an explicit, specific refusal that
   redirects to the capabilities that do exist — it never fabricates a bounding box or a pixel mask
   to avoid saying no.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, NoReturn

from ..core.errors import (
    AppError,
    ModelUnavailableError,
    ToolContractError,
    UnsupportedTaskError,
)
from ..core.logging import get_logger
from ..core.types import QueryTask
from .tools import (
    AdaptedSceneCaptioningTool,
    CaptioningTool,
    ChangeTool,
    ChangeVqaTool,
    FusionTool,
    GroundingTool,
    LandCoverTool,
    SarTool,
    Tool,
    ToolContext,
    ToolResult,
    VqaTool,
)

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from .modes import NormalizedRequest

logger = get_logger(__name__)

# The tools this prototype ships, in a stable order. Order is meaningful twice over: it breaks ties
# when two tools share a task at the same tier (so the single-image SAR tool is tried before the pair
# fusion tool for optical/SAR analysis), and it is the order ``/api/models`` lists them.
#
# ``AdaptedSceneCaptioningTool`` is the one PREFERRED rung: the learned EuroSAT-adapted ResNet-18. It
# sits ahead of ``CaptioningTool`` — which answers the same task deterministically — so the tier sort
# tries it first and falls through to the classical analyser whenever its checkpoint, torch or the
# ``enable_learned_models`` setting is missing. The deterministic tool is not replaced or altered.
_TOOL_TYPES: tuple[type[Tool], ...] = (
    LandCoverTool,
    GroundingTool,
    VqaTool,
    AdaptedSceneCaptioningTool,
    CaptioningTool,
    ChangeTool,
    ChangeVqaTool,
    SarTool,
    FusionTool,
)

# Tasks a user can legitimately ask for, and the classifier recognises, but that no tool implements.
# Each maps to the lead sentence of an honest refusal; the available capabilities are appended at
# dispatch time so the refusal always points somewhere useful, never fabricating a result (§18).
_UNSUPPORTED: dict[QueryTask, str] = {
    QueryTask.OBJECT_DETECTION: (
        "Detecting or counting discrete objects — ships, vehicles, aircraft, individual buildings "
        "— with bounding boxes is not implemented in this prototype. It ships deterministic "
        "remote-sensing analysers only, and drawing a box for an object it cannot actually detect "
        "would be a fabrication."
    ),
    QueryTask.SEGMENTATION: (
        "Segmenting the image into arbitrary pixel masks is not implemented in this prototype. The "
        "closest measured capability is per-pixel land-cover classification, which produces masks "
        "for a fixed water / vegetation / built-up / bare-soil vocabulary rather than an arbitrary "
        "target."
    ),
}

# Human-readable capability phrases, in the order a refusal lists them. Every dispatchable task must
# have one (enforced by test), so a redirect can never silently omit a capability that exists.
_CAPABILITY_PHRASES: dict[QueryTask, str] = {
    QueryTask.LAND_COVER_ANALYSIS: "classify land cover in an optical scene",
    QueryTask.VQA: "answer a question about an optical scene",
    QueryTask.CAPTIONING: "describe an optical scene",
    QueryTask.GROUNDING: "locate a land-cover type as regions",
    QueryTask.CHANGE_DETECTION: "detect and describe change between two dates",
    QueryTask.CHANGE_VQA: "answer a question about what changed between two dates",
    QueryTask.OPTICAL_SAR_ANALYSIS: "analyse a SAR image, or fuse an optical and SAR pair",
}


def _humanize(items: list[str]) -> str:
    """Join phrases into a readable clause: ``"a"`` / ``"a and b"`` / ``"a, b and c"``."""
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return f"{', '.join(items[:-1])} and {items[-1]}"


# Contract rules, in the order a refusal should prefer to report them. A tool that accepted the
# image *count* and failed on mode or modality is describing the request more accurately than a
# sibling tool that rejected the count outright: for an optical+optical pair sent to optical/SAR
# analysis, "this needs a radar image too" is the useful sentence, not the single-image tool's
# "works on exactly 1 image". Ties fall back to chain order (highest tier, first registered).
_VIOLATION_PRIORITY: dict[str, int] = {
    "required_modalities": 0,
    "supported_modes": 1,
    "min_images": 2,
    "max_images": 2,
}


def _primary_rejection(rejections: list[ToolContractError]) -> ToolContractError:
    """The single most informative contract refusal to surface to the user."""
    return min(
        rejections,
        key=lambda exc: _VIOLATION_PRIORITY.get(str(exc.context.get("violated")), 9),
    )


class ToolRegistry:
    """Maps each :class:`QueryTask` to its fallback chain and dispatches an analysis to it.

    Construct once and share it: the shipped tools are stateless deterministic analysers, so a single
    registry serves every request. :func:`default_registry` returns a lazily-built shared instance;
    a bespoke set of tools (for tests, or a future model rung) can be passed to the constructor.
    """

    def __init__(self, tools: Iterable[Tool] | None = None) -> None:
        self._tools: tuple[Tool, ...] = (
            tuple(tools) if tools is not None else tuple(cls() for cls in _TOOL_TYPES)
        )
        self._chains: dict[QueryTask, tuple[Tool, ...]] = self._build_chains(self._tools)

    @staticmethod
    def _build_chains(tools: tuple[Tool, ...]) -> dict[QueryTask, tuple[Tool, ...]]:
        grouped: dict[QueryTask, list[Tool]] = {}
        for tool in tools:
            grouped.setdefault(tool.task, []).append(tool)
        # Sort each chain by tier; the sort is stable, so insertion order breaks ties among equal
        # tiers (keeping the single-image SAR tool ahead of the pair fusion tool).
        chains: dict[QueryTask, tuple[Tool, ...]] = {
            task: tuple(sorted(members, key=lambda t: t.tier))
            for task, members in grouped.items()
        }
        # Change detection and change VQA were once one chain, aliased here. They are now separate
        # tools (see ChangeVqaTool) because they answer different questions from the same
        # measurement, so no alias is needed — and the alias would have silently masked a missing
        # change-VQA registration. Fall back to the change chain only if no VQA tool is registered,
        # so a bespoke registry built from a subset of tools still serves the task rather than
        # reporting it undispatchable.
        if QueryTask.CHANGE_DETECTION in chains:
            chains.setdefault(QueryTask.CHANGE_VQA, chains[QueryTask.CHANGE_DETECTION])
        return chains

    # -- introspection -----------------------------------------------------
    def tools_for(self, task: QueryTask) -> tuple[Tool, ...]:
        """The fallback chain for a task, most-capable first; empty when the task is not dispatchable."""
        return self._chains.get(task, ())

    def is_dispatchable(self, task: QueryTask) -> bool:
        """Whether some tool implements this task."""
        return task in self._chains

    def is_unsupported(self, task: QueryTask) -> bool:
        """Whether the task is recognised but deliberately unimplemented (an honest refusal)."""
        return task in _UNSUPPORTED

    def supported_tasks(self) -> tuple[QueryTask, ...]:
        """Dispatchable tasks that carry a capability phrase, in presentation order."""
        return tuple(t for t in _CAPABILITY_PHRASES if t in self._chains)

    def unsupported_tasks(self) -> tuple[QueryTask, ...]:
        """Recognised-but-unimplemented tasks the registry refuses with a redirect."""
        return tuple(_UNSUPPORTED)

    def describe_all(self) -> list[dict[str, Any]]:
        """Metadata for every shipped tool, in registration order (for ``/api/models``, §30)."""
        return [tool.describe() for tool in self._tools]

    def describe_all_with_contracts(self) -> list[dict[str, Any]]:
        """:meth:`describe_all` plus each tool's declared input contract and live availability.

        Kept separate from :meth:`describe_all` because ``describe()`` is a fixed four-key shape
        that callers compare by equality. ``available`` is *probed*, not asserted: a tool appears
        as ready only if it can actually run right now (§26). ``runtime`` is present only for a tool
        backed by a real artefact, and carries the probed checkpoint path and mode.
        """
        out: list[dict[str, Any]] = []
        for tool in self._tools:
            entry = dict(tool.describe())
            entry["contract"] = tool.describe_contract()
            entry["available"] = bool(tool.available())
            runtime = tool.describe_runtime()
            if runtime is not None:
                entry["runtime"] = runtime
            out.append(entry)
        return out

    # -- pre-execution contract gate (§6, §38) -----------------------------
    def check_contracts(
        self, task: QueryTask, request: NormalizedRequest
    ) -> tuple[Tool, ...]:
        """Tools in ``task``'s chain whose declared contract accepts ``request``.

        This is the gate the brief requires *before* execution: it answers "can this request even
        be run?" using declared metadata only, with no imagery touched and no analysis started.

        Returns:
            The accepting tools, most-capable first. Never empty — if no tool accepts, this raises.

        Raises:
            UnsupportedTaskError: the task is recognised but deliberately unimplemented.
            ToolContractError: no tool for this task accepts this input configuration. The error
                raised is the primary (highest-tier, first-registered) tool's own, so the message
                names the specific thing the user should change.
            ModelUnavailableError: no rung of the chain can run at all.
            ValueError: the task is not dispatchable (an internal routing error, not a user fault).
        """
        if task in _UNSUPPORTED:
            self._refuse_unsupported(task)  # raises
        chain = self._chains.get(task)
        if chain is None:
            raise ValueError(f"Task {task.value!r} is not dispatchable through the tool registry.")

        runnable = [tool for tool in chain if tool.available()]
        if not runnable:
            raise ModelUnavailableError(
                "No implementation is currently available for this analysis.",
                context={"task": task.value, "chain": [t.name for t in chain]},
            )

        accepted: list[Tool] = []
        rejections: list[ToolContractError] = []
        for tool in runnable:
            try:
                tool.validate_request(request)
            except ToolContractError as exc:
                rejections.append(exc)
            else:
                accepted.append(tool)

        if not accepted:
            primary = _primary_rejection(rejections)
            logger.info(
                "request rejected by tool input contract before execution",
                extra={
                    "task": task.value,
                    "mode": request.mode.value,
                    "input_count": request.input_count,
                    "violated": primary.context.get("violated"),
                },
            )
            raise primary
        return tuple(accepted)

    # -- dispatch ----------------------------------------------------------
    def dispatch(
        self,
        task: QueryTask,
        ctx: ToolContext,
        *,
        request: NormalizedRequest | None = None,
    ) -> ToolResult:
        """Run the analysis for ``task`` on ``ctx`` and return its :class:`ToolResult`.

        When ``request`` is supplied, :meth:`check_contracts` runs first, so an input
        configuration the task cannot serve is rejected as ``TOOL_INPUT_CONTRACT_ERROR`` before
        any raster is read — rather than surfacing later as a zero-confidence result. Callers that
        already hold a :class:`~app.agents.modes.NormalizedRequest` (the API does) should always
        pass it; the parameter is optional only so direct tool-level tests can call ``dispatch``
        with a bare context.

        Raises :class:`~app.core.errors.UnsupportedTaskError` for a recognised-but-unimplemented
        task (with a redirect to what exists), and :class:`~app.core.errors.ModelUnavailableError`
        when no rung of the chain can run. When the inputs fit no tool in the chain, that tool's own
        specific, recoverable error is surfaced unchanged. This never returns a fabricated result to
        avoid raising — the refusal *is* the correct answer (§28).
        """
        if request is not None:
            chain_after_contract = self.check_contracts(task, request)
        else:
            chain_after_contract = None
        if task in _UNSUPPORTED:
            self._refuse_unsupported(task)  # raises
        chain = self._chains.get(task)
        if chain is None:
            # UNKNOWN is resolved by the router and REPORT_GENERATION by the report path, both before
            # dispatch; reaching here with either is an internal routing error, not a user refusal.
            raise ValueError(f"Task {task.value!r} is not dispatchable through the tool registry.")

        available = (
            list(chain_after_contract)
            if chain_after_contract is not None
            else [tool for tool in chain if tool.available()]
        )
        if not available:
            raise ModelUnavailableError(
                "No implementation is currently available for this analysis.",
                context={"task": task.value, "chain": [t.name for t in chain]},
            )

        # Applicability: keep the tools whose validate_input accepts these inputs. For a single-tool
        # task that is just that tool; for optical/SAR analysis it selects SAR vs fusion by the inputs.
        applicable: list[Tool] = []
        primary_reason: AppError | None = None
        for tool in available:
            try:
                tool.validate_input(ctx)
            except AppError as exc:
                if primary_reason is None:
                    primary_reason = exc
            else:
                applicable.append(tool)

        if not applicable:
            # The inputs fit no implementation for this task. Surface the primary (highest-tier,
            # first-registered) tool's reason — the most specific, user-actionable refusal.
            if primary_reason is not None:
                raise primary_reason
            raise ModelUnavailableError(  # unreachable: available is non-empty
                "No implementation could accept these inputs.",
                context={"task": task.value, "chain": [t.name for t in chain]},
            )

        # Tier fallback: run the most-capable applicable tool; drop to the next only on a model
        # outage (§18). Any other failure is an honest downstream refusal and propagates unchanged.
        outage: ModelUnavailableError | None = None
        for tool in applicable:
            try:
                return tool.run(ctx)
            except ModelUnavailableError as exc:
                logger.warning(
                    "tool unavailable at run time; falling back to next tier",
                    extra={"tool": tool.name, "task": task.value},
                )
                outage = exc
        if outage is not None:
            raise outage
        raise ModelUnavailableError(  # unreachable: applicable is non-empty
            "No implementation in the fallback chain could run.",
            context={"task": task.value, "chain": [t.name for t in applicable]},
        )

    def _refuse_unsupported(self, task: QueryTask) -> NoReturn:
        """Decline a recognised-but-unimplemented task, redirecting to what the system can do (§18)."""
        lead = _UNSUPPORTED[task]
        capabilities = _humanize([_CAPABILITY_PHRASES[t] for t in self.supported_tasks()])
        raise UnsupportedTaskError(
            f"{lead} What this system can do instead: {capabilities}.",
            context={"task": task.value, "supported": [t.value for t in self.supported_tasks()]},
        )


_default: ToolRegistry | None = None


def default_registry() -> ToolRegistry:
    """The shared registry of the tools this prototype ships. Built once on first use, then reused."""
    global _default
    if _default is None:
        _default = ToolRegistry()
    return _default
