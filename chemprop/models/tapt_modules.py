import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import warnings


def safe_scatter_mean(src: torch.Tensor, index: torch.Tensor,
                      dim_size: int) -> torch.Tensor:
    try:
        from torch_scatter import scatter_mean
        return scatter_mean(src, index, dim=0, dim_size=dim_size)
    except ImportError:
        warnings.warn("torch_scatter not available, using slower fallback.", UserWarning)

        result = torch.zeros(
            dim_size, src.size(-1),
            device=src.device, dtype=src.dtype
        )
        count = torch.zeros(
            dim_size, 1,
            device=src.device, dtype=src.dtype
        )
        result.index_add_(0, index, src)
        count.index_add_(0, index, torch.ones(
            src.size(0), 1,
            device=src.device, dtype=src.dtype
        ))
        count = torch.clamp(count, min=1e-8)
        return result / count


def safe_scatter_sum(src: torch.Tensor, index: torch.Tensor,
                     dim_size: int) -> torch.Tensor:
    try:
        from torch_scatter import scatter_add
        return scatter_add(src, index, dim=0, dim_size=dim_size)
    except ImportError:
        result = torch.zeros(
            dim_size, src.size(-1),
            device=src.device, dtype=src.dtype
        )
        result.index_add_(0, index, src)
        return result


class TaskEmbeddingLayer(nn.Module):

    def __init__(self, num_tasks: int, embed_dim: int = 64):
        super().__init__()
        self.num_tasks = num_tasks
        self.embed_dim = embed_dim

        self.task_embed = nn.Embedding(num_tasks, embed_dim)
        nn.init.normal_(self.task_embed.weight, std=0.02)

    def forward(self, task_ids: torch.Tensor) -> torch.Tensor:
        return self.task_embed(task_ids)


class CrossAttentionModule(nn.Module):

    def __init__(self, fg_dim: int, task_embed_dim: int,
                 hidden_dim: int = 128, num_heads: int = 4):
        super().__init__()

        self.fg_dim = fg_dim
        self.task_embed_dim = task_embed_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        assert hidden_dim % num_heads == 0, \
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"

        self.head_dim = hidden_dim // num_heads

        self.fg_proj = nn.Linear(fg_dim, hidden_dim)
        self.task_proj = nn.Linear(task_embed_dim, hidden_dim)
        self.value_proj = nn.Linear(fg_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self._init_weights()

    def _init_weights(self):
        for module in [self.fg_proj, self.task_proj, self.value_proj, self.out_proj]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, fg_states: torch.Tensor, task_embed: torch.Tensor,
                batch_idx: torch.Tensor) -> torch.Tensor:
        batch_size = task_embed.size(0)
        num_atoms = fg_states.size(0)
        device = fg_states.device

        fg_proj = self.fg_proj(fg_states)
        task_proj = self.task_proj(task_embed)
        value_proj = self.value_proj(fg_states)

        fg_head = fg_proj.view(num_atoms, self.num_heads, self.head_dim)
        task_head = task_proj.view(batch_size, self.num_heads, self.head_dim)
        value_head = value_proj.view(num_atoms, self.num_heads, self.head_dim)

        task_head_expanded = task_head.index_select(0, batch_idx)

        scores = (fg_head * task_head_expanded).sum(dim=-1) / (self.head_dim ** 0.5)

        flat_scores = scores.view(-1, 1)
        flat_batch_idx = batch_idx.repeat_interleave(self.num_heads)

        max_scores_per_mol = safe_scatter_mean(
            flat_scores, flat_batch_idx, batch_size * self.num_heads
        )
        max_scores_per_mol = max_scores_per_mol.view(batch_size, self.num_heads)
        max_scores_expanded = max_scores_per_mol.index_select(0, batch_idx)

        scores_exp = torch.exp(scores - max_scores_expanded)

        flat_scores_exp = scores_exp.view(-1, 1)
        sum_exp_per_mol = safe_scatter_sum(
            flat_scores_exp, flat_batch_idx, batch_size * self.num_heads
        )
        sum_exp_per_mol = sum_exp_per_mol.view(batch_size, self.num_heads)
        sum_exp_expanded = sum_exp_per_mol.index_select(0, batch_idx)

        attn_weights = scores_exp / (sum_exp_expanded + 1e-8)

        attn_weighted = attn_weights.unsqueeze(-1) * value_head

        attn_output = attn_weighted.reshape(num_atoms, self.hidden_dim)

        output = self.out_proj(attn_output)

        return output


class TAPTPromptModule(nn.Module):

    def __init__(self, num_tasks: int, fg_dim: int = 133,
                 prompt_dim: int = 128, hidden_dim: int = 256):
        super().__init__()

        self.num_tasks = num_tasks
        self.fg_dim = fg_dim
        self.prompt_dim = prompt_dim
        self.hidden_dim = hidden_dim

        self.task_embed_layer = TaskEmbeddingLayer(
            num_tasks=num_tasks,
            embed_dim=hidden_dim // 2
        )

        self.cross_attn = CrossAttentionModule(
            fg_dim=fg_dim,
            task_embed_dim=hidden_dim // 2,
            hidden_dim=hidden_dim,
            num_heads=4
        )

        self.fusion_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, prompt_dim)
        )

        for module in self.fusion_proj:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def generate_prompt(self, fg_states: torch.Tensor, batch_idx: torch.Tensor,
                        task_id: int) -> torch.Tensor:
        batch_size = batch_idx.max().item() + 1 if len(batch_idx) > 0 else 1
        device = fg_states.device

        task_ids = torch.full(
            (batch_size,), task_id,
            dtype=torch.long, device=device
        )
        task_embed = self.task_embed_layer(task_ids)

        attn_output = self.cross_attn(fg_states, task_embed, batch_idx)

        mol_output = safe_scatter_mean(attn_output, batch_idx, batch_size)

        prompts = self.fusion_proj(mol_output)

        return prompts


__all__ = [
    'safe_scatter_mean',
    'safe_scatter_sum',
    'TaskEmbeddingLayer',
    'CrossAttentionModule',
    'TAPTPromptModule',
]
