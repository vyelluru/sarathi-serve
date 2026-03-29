from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Optional, get_args, get_origin
import torch

from sarathi.config.base_poly_config import BasePolyConfig
from sarathi.config.flat_dataclass import create_flat_dataclass
from sarathi.config.utils import get_all_subclasses, get_inner_type, is_optional, is_subclass
from sarathi.logger import init_logger
from sarathi.transformers_utils.config import get_config
from sarathi.types import AttentionBackend, ControllerType, ResourceMapping, SchedulerType
from sarathi.utils.hf_utils import get_and_verify_dtype, get_and_verify_max_len

logger = init_logger(__name__)


def _coerce_enum_value(enum_cls: type[Enum], value: Any) -> Enum:
    if isinstance(value, enum_cls):
        return value

    if isinstance(value, str):
        normalized_value = value.upper()
        if normalized_value in enum_cls.__members__:
            return enum_cls[normalized_value]

        for member in enum_cls:
            if isinstance(member.value, str) and member.value.upper() == normalized_value:
                return member

    return enum_cls(value)


def _resolve_poly_config_subclass(
    base_cls: type[BasePolyConfig], type_value: Any
) -> type[BasePolyConfig]:
    for subclass in get_all_subclasses(base_cls):
        subclass_type = subclass.get_type()
        if type_value == subclass_type:
            return subclass

        if isinstance(type_value, str):
            normalized_value = type_value.upper()
            if isinstance(subclass_type, Enum):
                if subclass_type.name == normalized_value:
                    return subclass
                if (
                    isinstance(subclass_type.value, str)
                    and subclass_type.value.upper() == normalized_value
                ):
                    return subclass
            elif (
                isinstance(subclass_type, str)
                and subclass_type.upper() == normalized_value
            ):
                return subclass

    raise ValueError(f"Invalid type '{type_value}' for {base_cls.__name__}.")


def _load_poly_config_from_dict(
    base_cls: type[BasePolyConfig], config_dict: dict[str, Any]
) -> BasePolyConfig:
    if not isinstance(config_dict, dict):
        raise ValueError(
            f"Expected a mapping for {base_cls.__name__}, got {type(config_dict).__name__}."
        )
    if "type" not in config_dict:
        raise ValueError(f"Missing 'type' for {base_cls.__name__} config.")

    subclass = _resolve_poly_config_subclass(base_cls, config_dict["type"])
    subclass_config = {k: v for k, v in config_dict.items() if k != "type"}
    return _load_dataclass_from_dict(subclass, subclass_config)


def _coerce_config_value(field_type: Any, value: Any) -> Any:
    if value is None:
        return None

    if is_optional(field_type):
        return _coerce_config_value(get_inner_type(field_type), value)

    origin = get_origin(field_type)
    if origin is list:
        (inner_type,) = get_args(field_type)
        return [_coerce_config_value(inner_type, item) for item in value]
    if origin is tuple:
        inner_types = get_args(field_type)
        if len(inner_types) == 2 and inner_types[1] is Ellipsis:
            return tuple(_coerce_config_value(inner_types[0], item) for item in value)
        return tuple(
            _coerce_config_value(inner_type, item)
            for inner_type, item in zip(inner_types, value)
        )
    if origin is dict:
        key_type, value_type = get_args(field_type)
        return {
            _coerce_config_value(key_type, key): _coerce_config_value(value_type, item)
            for key, item in value.items()
        }

    if isinstance(field_type, type) and issubclass(field_type, Enum):
        return _coerce_enum_value(field_type, value)

    if is_subclass(field_type, BasePolyConfig):
        return _load_poly_config_from_dict(field_type, value)

    if hasattr(field_type, "__dataclass_fields__"):
        return _load_dataclass_from_dict(field_type, value)

    return value


def _load_dataclass_from_dict(cls: type[Any], config_dict: dict[str, Any]) -> Any:
    if isinstance(config_dict, cls):
        return config_dict
    if not isinstance(config_dict, dict):
        raise ValueError(
            f"Expected a mapping for {cls.__name__}, got {type(config_dict).__name__}."
        )

    field_map = {field.name: field for field in fields(cls)}
    unknown_fields = sorted(set(config_dict) - set(field_map))
    if unknown_fields:
        unknown_fields_str = ", ".join(unknown_fields)
        raise ValueError(f"Unknown fields for {cls.__name__}: {unknown_fields_str}")

    kwargs = {}
    for field_info in fields(cls):
        if field_info.name not in config_dict:
            continue
        kwargs[field_info.name] = _coerce_config_value(
            field_info.type, config_dict[field_info.name]
        )

    return cls(**kwargs)


def _serialize_config_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, torch.dtype):
        return str(value)
    if is_dataclass(value):
        serialized_fields = {
            field_info.name: _serialize_config_value(getattr(value, field_info.name))
            for field_info in fields(value)
        }
        if isinstance(value, BasePolyConfig):
            type_value = value.get_type()
            serialized_type = type_value.value if isinstance(type_value, Enum) else type_value
            return {"type": serialized_type, **serialized_fields}
        return serialized_fields
    if isinstance(value, list):
        return [_serialize_config_value(item) for item in value]
    if isinstance(value, tuple):
        return [_serialize_config_value(item) for item in value]
    if isinstance(value, dict):
        return {
            _serialize_config_value(key): _serialize_config_value(item)
            for key, item in value.items()
        }
    return value


@dataclass
class ModelConfig:
    model: str = field(
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        metadata={"help": "Name or path of the huggingface model to use."},
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={
            "help": "Trust remote code (e.g., from HuggingFace) when downloading the model and tokenizer."
        },
    )
    download_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Directory to download and load the weights, default to the default cache directory of huggingface."
        },
    )
    load_format: str = field(
        default="auto",
        metadata={
            "help": "The format of the model weights to load: 'auto', 'pt', 'safetensors', 'npcache', or 'dummy'."
        },
    )
    dtype: str = field(
        default="float16",
        metadata={
            "help": "Data type for model weights and activations. 'auto' will use FP16 for FP32 and FP16 models, and BF16 for BF16 models."
        },
    )
    seed: int = field(default=0, metadata={"help": "Random seed for reproducibility."})
    revision: Optional[str] = field(
        default=None,
        metadata={
            "help": "The specific model version to use. Can be a branch name, tag name, or commit id."
        },
    )
    max_model_len: Optional[int] = field(
        default=None,
        metadata={
            "help": "Maximum length of a sequence (including prompt and output). If None, will be derived from the model."
        },
    )

    def __post_init__(self):
        self.hf_config = get_config(self.model, self.trust_remote_code, self.revision)
        self.dtype = get_and_verify_dtype(self.hf_config, self.dtype)
        self.hf_config.dtype = self.dtype
        self.max_model_len = get_and_verify_max_len(self.hf_config, self.max_model_len)
        self._verify_load_format()

    def _verify_load_format(self) -> None:
        load_format = self.load_format.lower()
        if load_format not in ["auto", "pt", "safetensors", "npcache", "dummy"]:
            raise ValueError(
                f"Unknown load format: {self.load_format}. Must be one of "
                "'auto', 'pt', 'safetensors', 'npcache', or 'dummy'."
            )
        self.load_format = load_format

    def verify_with_parallel_config(
        self,
        parallel_config: "ParallelConfig",
    ) -> None:
        total_num_attention_heads = self.hf_config.num_attention_heads
        tensor_parallel_size = parallel_config.tensor_parallel_size
        if total_num_attention_heads % tensor_parallel_size != 0:
            raise ValueError(
                f"Total number of attention heads ({total_num_attention_heads})"
                " must be divisible by tensor parallel size "
                f"({tensor_parallel_size})."
            )

        total_num_hidden_layers = self.hf_config.num_hidden_layers
        pipeline_parallel_size = parallel_config.pipeline_parallel_size
        if total_num_hidden_layers % pipeline_parallel_size != 0:
            raise ValueError(
                f"Total number of hidden layers ({total_num_hidden_layers}) "
                "must be divisible by pipeline parallel size "
                f"({pipeline_parallel_size})."
            )

    def get_hidden_size(self) -> int:
        return self.hf_config.hidden_size

    def get_head_size(self) -> int:
        # FIXME(woosuk): This may not be true for all models.
        return self.hf_config.hidden_size // self.hf_config.num_attention_heads

    def get_num_kv_heads(self, parallel_config: "ParallelConfig") -> int:
        # For GPTBigCode & Falcon:
        # Note: for falcon, when new_decoder_architecture is True, the
        # multi_query flag is ignored and we use n_head_kv for the number of
        # KV heads.
        falcon_model_types = ["falcon", "RefinedWeb", "RefinedWebModel"]
        new_decoder_arch_falcon = (
            self.hf_config.model_type in falcon_model_types
            and getattr(self.hf_config, "new_decoder_architecture", False)
        )
        if not new_decoder_arch_falcon and getattr(
            self.hf_config, "multi_query", False
        ):
            # Multi-query attention, only one KV head.
            return 1
        # For Falcon:
        if getattr(self.hf_config, "n_head_kv", None) is not None:
            return self.hf_config.n_head_kv // parallel_config.tensor_parallel_size
        # For Falcon-40b/Falcon-180b:
        if getattr(self.hf_config, "num_kv_heads", None) is not None:
            return self.hf_config.num_kv_heads // parallel_config.tensor_parallel_size
        # For LLaMA-2:
        if getattr(self.hf_config, "num_key_value_heads", None) is not None:
            return (
                self.hf_config.num_key_value_heads
                // parallel_config.tensor_parallel_size
            )
        total_num_attention_heads = self.hf_config.num_attention_heads
        return total_num_attention_heads // parallel_config.tensor_parallel_size

    def get_num_q_heads(self, parallel_config: "ParallelConfig") -> int:
        if getattr(self.hf_config, "num_attention_heads", None) is not None:
            return (
                self.hf_config.num_attention_heads
                // parallel_config.tensor_parallel_size
            )
        raise ValueError("num_attention_heads is not defined in the model config")

    def get_num_layers(self, parallel_config: "ParallelConfig") -> int:
        total_num_hidden_layers = self.hf_config.num_hidden_layers
        return total_num_hidden_layers // parallel_config.pipeline_parallel_size

    def get_total_num_layers(self) -> int:
        return self.hf_config.num_hidden_layers


@dataclass
class CacheConfig:
    block_size: int = field(
        default=16, metadata={"help": "Size of a cache block in number of tokens."}
    )
    num_gpu_blocks: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of GPU blocks for caching. This gets set after profiling."
        },
    )


@dataclass
class ParallelConfig:
    pipeline_parallel_size: int = field(
        default=2, metadata={"help": "Number of pipeline parallel groups."}
    )
    tensor_parallel_size: int = field(
        default=1, metadata={"help": "Number of tensor parallel groups."}
    )

    def __post_init__(self):
        self.world_size = self.pipeline_parallel_size * self.tensor_parallel_size


@dataclass
class BaseSchedulerConfig(BasePolyConfig):
    max_num_seqs: int = field(
        default=128,
        metadata={
            "help": "Maximum number of sequences to be processed in a single iteration (batch size)."
        },
    )

    @abstractmethod
    def get_max_num_batched_tokens(self, max_model_len: int):
        pass


@dataclass
class VllmSchedulerConfig(BaseSchedulerConfig):
    max_batched_tokens: Optional[int] = field(
        default=None, metadata={"help": "Maximum number of batched tokens."}
    )

    def get_max_num_batched_tokens(self, max_model_len: int):
        if self.max_batched_tokens:
            return min(self.max_batched_tokens, max_model_len)
        return max_model_len

    @staticmethod
    def get_type():
        return SchedulerType.VLLM


@dataclass
class SimpleChunkingSchedulerConfig(BaseSchedulerConfig):
    chunk_size: int = field(
        default=512,
        metadata={"help": "Size of each chunk for simple chunking scheduler."},
    )

    def get_max_num_batched_tokens(self, max_model_len: int):
        #HERE - CHUNK SIZE SHOULD BE COMING FROM THE CONTROLLER
        return self.chunk_size

    @staticmethod
    def get_type():
        return SchedulerType.SIMPLE_CHUNKING


@dataclass
class OrcaSchedulerConfig(BaseSchedulerConfig):

    def get_max_num_batched_tokens(self, max_model_len: int):
        return self.max_num_seqs * max_model_len

    @staticmethod
    def get_type():
        return SchedulerType.ORCA


@dataclass
class FasterTransformerSchedulerConfig(BaseSchedulerConfig):

    def get_max_num_batched_tokens(self, max_model_len: int):
        return self.max_num_seqs * max_model_len

    @staticmethod
    def get_type():
        return SchedulerType.FASTER_TRANSFORMER


@dataclass
class SarathiSchedulerConfig(BaseSchedulerConfig):
    chunk_size: int = field(
        default=512, metadata={"help": "Size of each chunk for Sarathi scheduler."}
    )
    controller_type: ControllerType = field(
        default=ControllerType.AIMD,
        metadata={"help": "Chunk size controller strategy: NONE (static), AIMD, or PID."},
    )
    enable_dynamic_chunking_schedule: bool = field(
        default=False, metadata={"help": "Enable dynamic chunking schedule."}
    )
    low_chunk_size: Optional[int] = field(
        default=None, metadata={"help": "Minimum chunk size for dynamic chunking."}
    )
    high_chunk_size: Optional[int] = field(
        default=None, metadata={"help": "Maximum chunk size for dynamic chunking."}
    )
    chunk_schedule_max_tokens: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum number of tokens for chunk scheduling."},
    )
    chunk_schedule_stages: Optional[int] = field(
        default=None, metadata={"help": "Number of stages for chunk scheduling."}
    )

    def get_max_num_batched_tokens(self, max_model_len: int):
        # Sarathi never schedules more than chunk_size tokens in one iteration.
        if self.enable_dynamic_chunking_schedule:
            return self.high_chunk_size
        else:
            #HERE - CHUNK SIZE SHOULD BE COMING FROM THE CONTROLLER
            return self.chunk_size

    @staticmethod
    def get_type():
        return SchedulerType.SARATHI


@dataclass
class MetricsConfig:
    """Metric configuration."""

    write_metrics: bool = field(
        default=True, metadata={"help": "Whether to write metrics."}
    )
    wandb_project: Optional[str] = field(
        default=None, metadata={"help": "Weights & Biases project name."}
    )
    wandb_group: Optional[str] = field(
        default=None, metadata={"help": "Weights & Biases group name."}
    )
    wandb_run_name: Optional[str] = field(
        default=None, metadata={"help": "Weights & Biases run name."}
    )
    wandb_sweep_id: Optional[str] = field(
        default=None, metadata={"help": "Weights & Biases sweep ID."}
    )
    wandb_run_id: Optional[str] = field(
        default=None, metadata={"help": "Weights & Biases run ID."}
    )
    enable_op_level_metrics: bool = field(
        default=False, metadata={"help": "Enable operation-level metrics."}
    )
    enable_cpu_op_level_metrics: bool = field(
        default=False, metadata={"help": "Enable CPU operation-level metrics."}
    )
    enable_chrome_trace: bool = field(
        default=True, metadata={"help": "Enable Chrome tracing."}
    )
    enable_request_outputs: bool = field(
        default=False, metadata={"help": "Enable request outputs."}
    )
    keep_individual_batch_metrics: bool = field(
        default=False, metadata={"help": "Keep individual batch metrics."}
    )


@dataclass
class ReplicaConfig:
    replica_id: int = field(default=0, metadata={"help": "ID of the replica."})
    output_dir: str = field(
        default=".", metadata={"help": "Output directory for the replica."}
    )
    resource_mapping: Optional[ResourceMapping] = field(
        default=None, metadata={"help": "Resource mapping for the replica."}
    )

    def __post_init__(self):
        self.output_dir = f"{self.output_dir}/replica_{self.replica_id}"

    def get_resource_mapping(self, world_size: int):
        if not self.resource_mapping:
            self.resource_mapping = [
                (None, i) for i in range(world_size)  # List of (node_ip, gpu_id)
            ]
        return self.resource_mapping


@dataclass
class WorkerConfig:
    gpu_memory_utilization: float = field(
        default=0.8, metadata={"help": "GPU memory utilization fraction (0.0 to 1.0)."}
    )
    attention_backend: AttentionBackend = field(
        default=AttentionBackend.FLASHINFER,
        metadata={"help": "Backend to use for attention computation."},
    )

    def __post_init__(self):
        self._verify_args()

    def _verify_args(self) -> None:
        if self.gpu_memory_utilization > 1.0:
            raise ValueError(
                "GPU memory utilization must be less than 1.0. Got "
                f"{self.gpu_memory_utilization}."
            )


@dataclass
class SystemConfig:
    replica_config: ReplicaConfig = field(default_factory=ReplicaConfig)
    model_config: ModelConfig = field(default_factory=ModelConfig)
    worker_config: WorkerConfig = field(default_factory=WorkerConfig)
    cache_config: CacheConfig = field(default_factory=CacheConfig)
    parallel_config: ParallelConfig = field(default_factory=ParallelConfig)
    scheduler_config: BaseSchedulerConfig = field(
        default_factory=SarathiSchedulerConfig
    )
    metrics_config: MetricsConfig = field(default_factory=MetricsConfig)


@dataclass
class BaseEndpointConfig(ABC):
    log_level: str = field(default="info", metadata={"help": "Logging level."})
    output_dir: str = field(default="output", metadata={"help": "Output directory."})
    model_config: ModelConfig = field(default_factory=ModelConfig)
    worker_config: WorkerConfig = field(default_factory=WorkerConfig)
    cache_config: CacheConfig = field(default_factory=CacheConfig)
    parallel_config: ParallelConfig = field(default_factory=ParallelConfig)
    scheduler_config: BaseSchedulerConfig = field(
        default_factory=SarathiSchedulerConfig
    )
    metrics_config: MetricsConfig = field(default_factory=MetricsConfig)

    def __post_init__(self):
        self.output_dir = (
            f"{self.output_dir}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')}"
        )

    @classmethod
    def create_from_cli_args(cls, args=None):
        flat_config = create_flat_dataclass(cls).create_from_cli_args(args=args)
        instance = flat_config.reconstruct_original_dataclass()
        instance.__flat_config__ = flat_config
        return instance

    @classmethod
    def create_from_dict(cls, config_dict: dict[str, Any]):
        return _load_dataclass_from_dict(cls, config_dict)

    def to_dict(self):
        return _serialize_config_value(self)

    def create_system_config(self, replica_config: ReplicaConfig) -> SystemConfig:
        system_config = SystemConfig(
            replica_config=replica_config,
            model_config=self.model_config,
            cache_config=self.cache_config,
            parallel_config=self.parallel_config,
            scheduler_config=self.scheduler_config,
            metrics_config=self.metrics_config,
        )
        return system_config
