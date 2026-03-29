from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sarathi.model_executor.model_loader import get_model as get_model
    from sarathi.model_executor.utils import set_random_seed as set_random_seed

__all__ = [
    "get_model",
    "set_random_seed",
]


def __getattr__(name: str):
    if name == "get_model":
        from sarathi.model_executor.model_loader import get_model

        return get_model
    if name == "set_random_seed":
        from sarathi.model_executor.utils import set_random_seed

        return set_random_seed
    raise AttributeError(name)
