from typing import Union

from sarathi.model_executor.attention.no_op_attention_wrapper import (
    NoOpAttentionWrapper,
)
from sarathi.types import AttentionBackend

try:
    from sarathi.model_executor.attention.flashinfer_attention_wrapper import (
        FlashinferAttentionWrapper,
    )
    _FLASHINFER_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - depends on local runtime
    FlashinferAttentionWrapper = None
    _FLASHINFER_IMPORT_ERROR = exc

ATTENTION_BACKEND = AttentionBackend.NO_OP


def set_attention_backend(backend: Union[str, AttentionBackend]):
    if isinstance(backend, str):
        backend = backend.upper()
        if backend not in AttentionBackend.__members__:
            raise ValueError(f"Unsupported attention backend: {backend}")
        backend = AttentionBackend[backend]
    elif not isinstance(backend, AttentionBackend):
        raise ValueError(f"Unsupported attention backend: {backend}")

    if (
        backend == AttentionBackend.FLASHINFER
        and FlashinferAttentionWrapper is None
    ):
        raise RuntimeError(
            "FlashInfer is unavailable. Install flashinfer-python in a Python 3.10+ "
            "environment to use the FLASHINFER attention backend."
        ) from _FLASHINFER_IMPORT_ERROR

    global ATTENTION_BACKEND
    ATTENTION_BACKEND = backend


def get_attention_wrapper():
    if ATTENTION_BACKEND == AttentionBackend.FLASHINFER:
        if FlashinferAttentionWrapper is None:
            raise RuntimeError(
                "FlashInfer is unavailable. Install flashinfer-python in a Python 3.10+ "
                "environment to use the FLASHINFER attention backend."
            ) from _FLASHINFER_IMPORT_ERROR
        return FlashinferAttentionWrapper.get_instance()
    if ATTENTION_BACKEND == AttentionBackend.NO_OP:
        return NoOpAttentionWrapper.get_instance()

    raise ValueError(f"Unsupported attention backend: {ATTENTION_BACKEND}")
