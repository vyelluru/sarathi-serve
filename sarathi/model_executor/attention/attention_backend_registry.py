from sarathi.model_executor.attention.no_op_attention_wrapper import (
    NoOpAttentionWrapper,
)
from sarathi.types import AttentionBackend
from sarathi.utils.base_registry import BaseRegistry

try:
    from sarathi.model_executor.attention.flashinfer_attention_wrapper import (
        FlashinferAttentionWrapper,
    )
    _FLASHINFER_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - depends on local runtime
    FlashinferAttentionWrapper = None
    _FLASHINFER_IMPORT_ERROR = exc


class AttentionBackendRegistry(BaseRegistry):

    @classmethod
    def get(cls, key, *args, **kwargs):
        if key == AttentionBackend.FLASHINFER and FlashinferAttentionWrapper is None:
            raise RuntimeError(
                "FlashInfer is unavailable. Install flashinfer-python in a Python 3.10+ "
                "environment to use the FLASHINFER attention backend."
            ) from _FLASHINFER_IMPORT_ERROR
        return super().get(key, *args, **kwargs)


AttentionBackendRegistry.register(AttentionBackend.NO_OP, NoOpAttentionWrapper)

if FlashinferAttentionWrapper is not None:
    AttentionBackendRegistry.register(
        AttentionBackend.FLASHINFER, FlashinferAttentionWrapper
    )
