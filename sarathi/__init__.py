"""Sarathi: a high-throughput and memory-efficient inference engine for LLMs"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Import for type checkers only. Importing the engine stack here can pull in
    # heavyweight runtime dependencies (e.g. torch/transformers).
    from sarathi.core.datatypes.request_output import RequestOutput as RequestOutput
    from sarathi.core.datatypes.sampling_params import SamplingParams as SamplingParams
    from sarathi.engine.llm_engine import LLMEngine as LLMEngine

__version__ = "0.1.8"

__all__ = [
    "SamplingParams",
    "RequestOutput",
    "LLMEngine",
]


def __getattr__(name: str):
    # Lazily import to avoid importing heavy deps on `import sarathi`.
    if name == "SamplingParams":
        from sarathi.core.datatypes.sampling_params import SamplingParams

        return SamplingParams
    if name == "RequestOutput":
        from sarathi.core.datatypes.request_output import RequestOutput

        return RequestOutput
    if name == "LLMEngine":
        from sarathi.engine.llm_engine import LLMEngine

        return LLMEngine
    raise AttributeError(name)
