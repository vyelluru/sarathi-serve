"""A layer that samples the next tokens from the model's outputs."""

from typing import List, Tuple

import torch
import torch.nn as nn

try:
    from flashinfer.sampling import sampling_from_probs as flashinfer_sampling_from_probs
    from flashinfer.sampling import (
        top_k_top_p_sampling_from_logits as flashinfer_top_k_top_p_sampling_from_logits,
    )
    HAS_FLASHINFER = True
except Exception:  # pragma: no cover - depends on local runtime
    HAS_FLASHINFER = False

from sarathi.core.datatypes.sequence import (
    SamplerOutput,
    SamplerOutputs,
    SequenceMetadata,
)
from sarathi.model_executor.parallel_utils.tensor_parallel import (
    gather_from_tensor_model_parallel_region,
)

_SAMPLING_EPS = 1e-5


class Sampler(nn.Module):
    """Samples the next tokens from the model's outputs."""

    def __init__(self, embedding: torch.Tensor, vocab_size: int) -> None:
        super().__init__()
        self.embedding = embedding
        self.vocab_size = vocab_size

    def forward(
        self,
        hidden_states: torch.Tensor,
        seq_metadata_list: List[SequenceMetadata],
    ) -> SamplerOutputs:
        hidden_states = _prune_hidden_states(hidden_states, seq_metadata_list)
        logits = _get_logits(hidden_states, self.embedding, self.vocab_size)

        temperatures = _get_temperatures(seq_metadata_list)
        assert len(temperatures) == logits.shape[0]
        if any(temperature != 1.0 for temperature in temperatures):
            temperature_tensor = torch.tensor(
                temperatures, dtype=logits.dtype, device=logits.device
            )
            logits.div_(temperature_tensor.unsqueeze(dim=1))

        top_ps, top_ks = _get_top_p_top_k(seq_metadata_list, self.vocab_size)
        assert len(top_ps) == len(top_ks) == logits.shape[0]
        do_top_p = any(top_p < 1.0 - _SAMPLING_EPS for top_p in top_ps)
        do_top_k = any(top_k != self.vocab_size for top_k in top_ks)

        if not do_top_p and not do_top_k:
            probs = torch.softmax(logits, dim=-1, dtype=torch.float)
            sample_result = _sample(probs).cpu()
        else:
            top_ps_tensor = torch.tensor(
                top_ps, dtype=logits.dtype, device=logits.device
            )
            top_ks_tensor = torch.tensor(
                top_ks, dtype=torch.int, device=logits.device
            )
            sample_result = _top_k_top_p_sample(
                logits, top_ks_tensor, top_ps_tensor
            ).cpu()

        return [
            SamplerOutput(seq_metadata_list[i].seq.seq_id, sample_result[i])
            for i in range(len(seq_metadata_list))
        ]


def _get_logits(
    hidden_states: torch.Tensor, embedding: torch.Tensor, vocab_size: int
) -> torch.Tensor:
    logits = torch.matmul(hidden_states, embedding.t())
    logits = gather_from_tensor_model_parallel_region(logits)
    return logits[:, :vocab_size]


def _prune_hidden_states(
    hidden_states: torch.Tensor,
    seq_metadata_list: List[SequenceMetadata],
) -> torch.Tensor:
    last_token_indices = []
    token_idx = 0
    for seq_metadata in seq_metadata_list:
        if seq_metadata.is_prompt:
            prompt_len = seq_metadata.prompt_chunk_len
            last_token_indices.append(token_idx + prompt_len - 1)
            token_idx += prompt_len
        else:
            last_token_indices.append(token_idx)
            token_idx += 1

    last_token_indices = torch.tensor(
        last_token_indices, dtype=torch.long, device=hidden_states.device
    )
    return hidden_states.index_select(0, last_token_indices)


def _get_temperatures(seq_metadata_list: List[SequenceMetadata]) -> List[float]:
    temperatures: List[float] = []
    for seq_metadata in seq_metadata_list:
        temperature = seq_metadata.seq.sampling_params.temperature
        if temperature < _SAMPLING_EPS:
            temperature = 1.0
        temperatures.append(temperature)
    return temperatures


def _get_top_p_top_k(
    seq_metadata_list: List[SequenceMetadata],
    vocab_size: int,
) -> Tuple[List[float], List[int]]:
    top_ps: List[float] = []
    top_ks: List[int] = []
    for seq_metadata in seq_metadata_list:
        top_p = seq_metadata.seq.sampling_params.top_p
        top_k = min(seq_metadata.seq.sampling_params.top_k, vocab_size)
        top_k = vocab_size if top_k == -1 else top_k
        top_ps.append(top_p)
        top_ks.append(top_k)
    return top_ps, top_ks


def _top_k_top_p_sample(
    logits: torch.Tensor, top_ks: torch.Tensor, top_ps: torch.Tensor
) -> torch.Tensor:
    if HAS_FLASHINFER:
        batch_next_token_ids = flashinfer_top_k_top_p_sampling_from_logits(
            logits, top_ks, top_ps
        )
        return batch_next_token_ids.view(-1)

    probs = torch.softmax(logits, dim=-1, dtype=torch.float)
    filtered_probs = []

    for row_idx in range(probs.shape[0]):
        row_probs = probs[row_idx]
        top_k = top_ks[row_idx].item()
        top_p = top_ps[row_idx].item()

        if top_k < row_probs.shape[0]:
            topk_values, _ = torch.topk(row_probs, top_k)
            threshold = topk_values[-1]
            row_probs = torch.where(
                row_probs >= threshold, row_probs, torch.zeros_like(row_probs)
            )

        if top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(row_probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=0)
            sorted_mask = cumulative_probs > top_p
            sorted_mask[1:] = sorted_mask[:-1].clone()
            sorted_mask[0] = False
            sorted_probs = torch.where(
                sorted_mask, torch.zeros_like(sorted_probs), sorted_probs
            )
            row_probs = torch.zeros_like(row_probs).scatter(
                0, sorted_indices, sorted_probs
            )

        total_prob = row_probs.sum()
        if total_prob <= 0:
            row_probs = probs[row_idx]
            total_prob = row_probs.sum()

        filtered_probs.append(row_probs / total_prob)

    return torch.multinomial(torch.stack(filtered_probs), num_samples=1).view(-1)


def _sample(probs: torch.Tensor) -> torch.Tensor:
    if HAS_FLASHINFER:
        return flashinfer_sampling_from_probs(probs)
    return torch.multinomial(probs, num_samples=1).view(-1)
