"""Structured error hierarchy and the user-facing error envelope.

Two audiences, two levels of detail (brief §29):

* **Users** get ``error_code`` + a plain-language ``message`` + whether it is recoverable.
* **Logs** get the exception, its context dict, and the traceback.

A Python traceback must never reach the client, so route handlers convert ``AppError`` into
:class:`ErrorEnvelope` and everything else into a generic ``INTERNAL_ERROR``.
"""
from __future__ import annotations

from typing import Any


class ErrorCode:
    """Stable machine-readable error codes.

    Plain string constants rather than an Enum so they serialise trivially and can be
    asserted against in tests without imports.
    """

    # --- input / upload ---
    INVALID_FILE = "INVALID_FILE"
    INVALID_GEOTIFF = "INVALID_GEOTIFF"
    EMPTY_FILE = "EMPTY_FILE"
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"
    CORRUPT_RASTER = "CORRUPT_RASTER"
    UNSAFE_FILENAME = "UNSAFE_FILENAME"
    NO_VALID_PIXELS = "NO_VALID_PIXELS"
    """The file opened and parsed correctly but is almost entirely nodata.

    Distinct from ``CORRUPT_RASTER``: nothing is malformed, there is simply no data to
    measure. The distinction matters to the caller, because a corrupt file should be
    re-exported while an empty one should be re-cropped.
    """

    # --- query ---
    EMPTY_QUERY = "EMPTY_QUERY"
    QUERY_TOO_LONG = "QUERY_TOO_LONG"
    UNSUPPORTED_TASK = "UNSUPPORTED_TASK"

    # --- modality / semantics ---
    WRONG_MODALITY = "WRONG_MODALITY"
    MISSING_BAND = "MISSING_BAND"
    MISSING_SECOND_IMAGE = "MISSING_SECOND_IMAGE"
    TOO_MANY_IMAGES = "TOO_MANY_IMAGES"

    # --- geospatial ---
    CRS_MISMATCH = "CRS_MISMATCH"
    NO_SPATIAL_OVERLAP = "NO_SPATIAL_OVERLAP"
    INSUFFICIENT_OVERLAP = "INSUFFICIENT_OVERLAP"
    SIZE_MISMATCH = "SIZE_MISMATCH"
    MISSING_GEOREFERENCE = "MISSING_GEOREFERENCE"
    REPROJECTION_FAILED = "REPROJECTION_FAILED"
    REGISTRATION_FAILED = "REGISTRATION_FAILED"
    PAIR_INCOMPATIBLE = "PAIR_INCOMPATIBLE"
    """The two inputs cannot be compared at all — different sensors for a temporal task, no
    spatial overlap, or irreconcilable grids. Distinct from ``INSUFFICIENT_OVERLAP``, which
    means they *do* overlap but not enough."""

    MISSING_REQUIRED_METADATA = "MISSING_REQUIRED_METADATA"
    """A field the requested analysis cannot proceed without is absent (e.g. no georeference
    when the answer is expressed in ground units)."""

    # --- execution ---
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    TOOL_FAILED = "TOOL_FAILED"
    TOOL_INPUT_CONTRACT_ERROR = "TOOL_INPUT_CONTRACT_ERROR"
    """The request does not satisfy the selected tool's declared input contract — wrong image
    count, wrong mode, or missing modality.

    Raised **before** execution by :meth:`app.agents.tools.base.Tool.validate_request`. This is
    the code that must surface for "2 images sent to a single-image analysis": it is an
    application-level contract failure, categorically different from
    ``INSUFFICIENT_EVIDENCE`` (the analysis ran and the evidence was weak). Conflating the two
    is explicitly forbidden by the brief.
    """

    GPU_UNAVAILABLE = "GPU_UNAVAILABLE"
    """A learned model required an accelerator that is not present."""

    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    TIMEOUT = "TIMEOUT"
    PROCESSING_TIMEOUT = "PROCESSING_TIMEOUT"
    """Analysis exceeded its wall-clock budget. Separate from ``TIMEOUT`` (a transport/read
    timeout) because the remedy differs: shrink the input versus retry the request."""


    # --- infrastructure ---
    NOT_FOUND = "NOT_FOUND"
    STORAGE_ERROR = "STORAGE_ERROR"
    DATABASE_ERROR = "DATABASE_ERROR"
    RATE_LIMITED = "RATE_LIMITED"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class AppError(Exception):
    """Base class for every *expected* failure.

    Args:
        code: One of :class:`ErrorCode`.
        message: User-facing, plain language, no internals.
        recoverable: ``True`` if the user can fix it by changing their input.
        status_code: HTTP status to emit.
        context: Technical detail for logs only — never serialised to the client.
    """

    status_code: int = 400
    code: str = ErrorCode.INTERNAL_ERROR
    recoverable: bool = False

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        recoverable: bool | None = None,
        status_code: int | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code
        if recoverable is not None:
            self.recoverable = recoverable
        if status_code is not None:
            self.status_code = status_code
        self.context: dict[str, Any] = context or {}

    def to_envelope(self, request_id: str) -> dict[str, Any]:
        """Render the client-safe error body (see brief §29)."""
        return {
            "success": False,
            "error_code": self.code,
            "message": self.message,
            "request_id": request_id,
            "recoverable": self.recoverable,
        }


class ValidationError(AppError):
    """Input failed validation; the user can usually fix it and retry."""

    status_code = 422
    code = ErrorCode.INVALID_FILE
    recoverable = True


class GeospatialError(AppError):
    """Raster geometry, CRS, or alignment problem."""

    status_code = 422
    code = ErrorCode.MISSING_GEOREFERENCE
    recoverable = True


class UnsupportedTaskError(AppError):
    """The query does not map to any implemented capability.

    Deliberately its own type: returning this is *correct behaviour*, and is strictly
    preferable to routing an ambiguous query into an arbitrary workflow.
    """

    status_code = 422
    code = ErrorCode.UNSUPPORTED_TASK
    recoverable = True


class ToolContractError(AppError):
    """The request violates the selected tool's declared input contract (brief §6).

    Raised by :meth:`app.agents.tools.base.Tool.validate_request` **before any analysis runs**,
    so the caller is told what is wrong with the *request* rather than being handed a
    zero-confidence result. The alternative — letting the tool start and then reporting
    "insufficient evidence" — misattributes an application error to the imagery, and is
    prohibited.

    Carries the contract it failed against so the message can name the fix.
    """

    status_code = 422
    code = ErrorCode.TOOL_INPUT_CONTRACT_ERROR
    recoverable = True


class PairIncompatibleError(AppError):
    """Two inputs cannot be compared for the requested analysis.

    Separate from :class:`ToolContractError`: the *count* is right and the tool is the right
    one, but the two scenes themselves do not go together — mismatched sensors for a temporal
    comparison, or no usable spatial overlap.
    """

    status_code = 422
    code = ErrorCode.PAIR_INCOMPATIBLE
    recoverable = True


class ProcessingTimeoutError(AppError):
    """Analysis exceeded its wall-clock budget."""

    status_code = 504
    code = ErrorCode.PROCESSING_TIMEOUT
    recoverable = True



class ModelUnavailableError(AppError):
    """No implementation in the fallback chain could run."""

    status_code = 503
    code = ErrorCode.MODEL_UNAVAILABLE
    recoverable = False


class InsufficientEvidenceError(AppError):
    """Analysis ran but the result cannot support a defensible claim.

    Raised instead of emitting a low-quality answer, per the anti-hallucination rules.
    """

    status_code = 200  # a legitimate analytical outcome, not a transport failure
    code = ErrorCode.INSUFFICIENT_EVIDENCE
    recoverable = True


class NotFoundError(AppError):
    """Requested resource does not exist."""

    status_code = 404
    code = ErrorCode.NOT_FOUND
    recoverable = False


class StorageError(AppError):
    """Artifact storage read/write failure."""

    status_code = 500
    code = ErrorCode.STORAGE_ERROR
    recoverable = False


class RateLimitError(AppError):
    """Caller exceeded the allowance for an expensive endpoint."""

    status_code = 429
    code = ErrorCode.RATE_LIMITED
    recoverable = True
