# ===== chemprop/models/model_tapt.py (完整版) =====
"""
TAPT (Knowledge-Aware Prompt Tuning) - Ultra-conservative version
"""

import torch
import torch.nn as nn
from argparse import Namespace
from chemprop.models.model import MoleculeModel


class TAPTModule(nn.Module):
    """TAPT 核心模块（移除投影层，强制 prompt_dim = hidden_size）"""

    def __init__(self, num_tokens, hidden_size):
        super().__init__()

        # 直接使用 hidden_size，不接受 prompt_dim
        self.prompt_tokens = nn.Parameter(
            torch.randn(1, num_tokens, hidden_size) * 0.001
        )

        # Gate 初始化为极小值
        self.prompt_gate = nn.Parameter(torch.tensor(-4.0))

    def forward(self, mol_embedding):
        batch_size = mol_embedding.size(0)

        # 扩展并平均池化
        prompt_tokens = self.prompt_tokens.expand(batch_size, -1, -1)
        prompt_aggregated = prompt_tokens.mean(dim=1)

        # Gate 控制
        gate = torch.sigmoid(self.prompt_gate)

        # 残差连接
        final_embedding = (1 - gate) * mol_embedding + gate * prompt_aggregated

        return final_embedding, gate


class MoleculeModelTAPT(MoleculeModel):
    """TAPT 增强的分子属性预测模型（移除 prompt_dim）"""

    def __init__(
            self,
            classification: bool,
            multiclass: bool = False,
            pretrain: bool = False,
            num_prompt_tokens: int = 3
    ):
        super().__init__(classification, multiclass, pretrain)

        self.num_prompt_tokens = num_prompt_tokens
        self.tapt_module = None
        self.tapt_initialized = False

        print(f"[TAPT] ✅ Ultra-conservative TAPT (no projection):")
        print(f"  - Num prompt tokens: {self.num_prompt_tokens}")

    def initialize_tapt(self, args):
        """初始化 TAPT 模块"""
        if not hasattr(self, 'encoder') or self.encoder is None:
            raise RuntimeError("Must call create_encoder() before initialize_tapt()")

        # 创建 TAPT 模块（prompt_dim 自动等于 hidden_size）
        self.tapt_module = TAPTModule(
            self.num_prompt_tokens,
            args.hidden_size
        )

        # 向后兼容
        self.prompt_tokens = self.tapt_module.prompt_tokens
        self.prompt_gate = self.tapt_module.prompt_gate

        self.tapt_initialized = True

        gate_init = torch.sigmoid(self.prompt_gate).item()
        prompt_std = self.prompt_tokens.std().item()

        print(f"[TAPT] ✅ TAPT initialized successfully:")
        print(f"  - Hidden size: {args.hidden_size}")
        print(f"  - Prompt tokens: {self.num_prompt_tokens} x {args.hidden_size}")
        print(f"  - Gate: {gate_init:.6f}")
        print(f"  - Prompt std: {prompt_std:.6f}")

    def forward(self, step: str, aug: bool, batch, features_batch=None):
        """TAPT 前向传播"""
        if not self.tapt_initialized:
            raise RuntimeError("TAPT not initialized. Call initialize_tapt() first.")

        if step == 'pretrain':
            mol_embedding = self.encoder(step, aug, batch, features_batch)
            return mol_embedding

        mol_output = self.encoder(step, aug, batch, features_batch)

        if isinstance(mol_output, tuple):
            mol_embedding = mol_output[0]
        else:
            mol_embedding = mol_output

        # TAPT 调整
        final_embedding, gate = self.tapt_module(mol_embedding)

        # FFN
        output = self.ffn(final_embedding)

        return output


# ============================================================
# Helper Functions
# ============================================================

def build_tapt_model(args, encoder_name='CMPNN'):
    """构建 TAPT 模型（移除 prompt_dim）"""
    required_attrs = ['num_tasks', 'dataset_type', 'hidden_size']
    for attr in required_attrs:
        if not hasattr(args, attr):
            raise ValueError(f"args must have attribute '{attr}'")

    if not hasattr(args, 'output_size'):
        if args.dataset_type == 'multiclass':
            args.output_size = args.num_tasks * args.multiclass_num_classes
        else:
            args.output_size = args.num_tasks

    if not hasattr(args, 'atom_messages'):
        args.atom_messages = (encoder_name == 'DMPNN')
    if not hasattr(args, 'ffn_hidden_size'):
        args.ffn_hidden_size = args.hidden_size
    if not hasattr(args, 'ffn_num_layers'):
        args.ffn_num_layers = 2
    if not hasattr(args, 'dropout'):
        args.dropout = 0.0
    if not hasattr(args, 'activation'):
        args.activation = 'ReLU'

    print(f"[TAPT] Building TAPT model (no projection):")
    print(f"  - num_tasks={args.num_tasks}")
    print(f"  - hidden_size={args.hidden_size}")

    model = MoleculeModelTAPT(
        classification=(args.dataset_type == 'classification'),
        multiclass=(args.dataset_type == 'multiclass'),
        pretrain=False,
        num_prompt_tokens=getattr(args, 'num_prompt_tokens', 3)
    )

    model.create_encoder(args, encoder_name=encoder_name)
    model.create_ffn(args)
    model.initialize_tapt(args)

    print(f"[TAPT] ✅ Model built successfully!")

    return model


def freeze_kano_parameters(model):
    """冻结 KANO 参数"""
    # 冻结 encoder 和 FFN
    for param in model.encoder.parameters():
        param.requires_grad = False

    for param in model.ffn.parameters():
        param.requires_grad = False

    # 确保 TAPT 参数可训练
    if hasattr(model, 'tapt_module') and model.tapt_module is not None:
        for param in model.tapt_module.parameters():
            param.requires_grad = True

    # 验证冻结效果
    kano_frozen = 0
    kano_trainable = 0
    tapt_trainable = 0

    for name, param in model.named_parameters():
        if 'tapt' in name.lower() or 'prompt' in name.lower():
            if param.requires_grad:
                tapt_trainable += param.numel()
            else:
                raise RuntimeError(f"❌ TAPT parameter '{name}' is frozen!")
        else:
            if param.requires_grad:
                kano_trainable += param.numel()
                print(f"⚠️  WARNING: KANO parameter '{name}' is still trainable!")
            else:
                kano_frozen += param.numel()

    print(f"✅ Frozen: encoder + FFN ({kano_frozen:,} params)")
    print(f"✅ Trainable: tapt_module ({tapt_trainable:,} params)")

    if kano_trainable > 0:
        raise RuntimeError(f"❌ Freeze failed! {kano_trainable:,} KANO params still trainable!")
    if tapt_trainable == 0:
        raise RuntimeError(f"❌ No TAPT parameters are trainable!")

    return kano_frozen, tapt_trainable


def get_tapt_parameter_groups(model, kano_lr, prompt_lr):
    """差分学习率参数组"""
    kano_params = []
    kano_params.extend(model.encoder.parameters())
    kano_params.extend(model.ffn.parameters())

    tapt_params = []
    if hasattr(model, 'tapt_module') and model.tapt_module is not None:
        tapt_params.extend(model.tapt_module.parameters())

    param_groups = [
        {'params': kano_params, 'lr': kano_lr},
        {'params': tapt_params, 'lr': prompt_lr}
    ]

    kano_count = sum(p.numel() for p in kano_params if p.requires_grad)
    tapt_count = sum(p.numel() for p in tapt_params if p.requires_grad)

    print(f"✅ KANO params: {kano_count:,} (lr={kano_lr})")
    print(f"✅ TAPT params: {tapt_count:,} (lr={prompt_lr})")

    return param_groups
