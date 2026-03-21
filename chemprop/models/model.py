# ===== chemprop/models/model.py (FINAL FIXED VERSION 2.0) =====
"""
MoleculeModel - 科研严谨版 (Environment Self-Healing & Device Safe & CMPN Adapter)

修复日志:
  ✅ 环境自愈: 自动检测并修复 featurization.py 中 len() 被覆盖的问题。
  ✅ 参数清洗: forward() 严格清洗 prompt 参数。
  ✅ 设备安全: 修复 Mode 4 中 fg_states 在 CPU 导致的设备不匹配错误。
  ✅ 初始化修复: create_encoder 只传递合法的参数。
  ✅ CMPN适配器: 确保 CMPN 的三返回值被正确处理。
"""

from argparse import Namespace
import sys
import builtins


# ============================================================
# 🚑 Environment Self-Healing (环境自愈模块)
# ============================================================
def repair_featurization_module():
    try:
        import chemprop.features.featurization as feat_module
        if 'len' in feat_module.__dict__:
            current_val = feat_module.__dict__['len']
            if current_val is not builtins.len:
                print("\n" + "!" * 60)
                print(f"🚑 [Environment Alert] 检测到 featurization.py 中 'len' 被覆盖！")
                feat_module.len = builtins.len
                if feat_module.len is builtins.len:
                    print(f"✅ 修复成功！环境已净化。")
                else:
                    print(f"❌ 修复失败，请检查文件权限或手动修复 featurization.py")
                print("!" * 60 + "\n")
    except ImportError:
        pass
    except Exception as e:
        print(f"⚠️ [Environment Repair] 检查过程中发生警告: {e}")


repair_featurization_module()

# ============================================================
# 正常模块导入
# ============================================================
from chemprop.models.tapt_modules import TAPTPromptModule
from chemprop.features import mol2graph
from .cmpn import CMPN
from .mpn import MPN
from chemprop.nn_utils import get_activation_function, initialize_weights
import torch
import torch.nn as nn
from typing import Optional, List, Union, Tuple
import torch.nn.functional as F
import math


# ============================================================
# 严谨的配置验证器
# ============================================================

def validate_and_ensure_args(args: Namespace, mode: str):
    missing_keys = []
    critical_params = [
        'hidden_size', 'depth', 'num_tasks', 'dataset_type',
        'dropout', 'activation', 'atom_messages', 'undirected'
    ]
    for param in critical_params:
        if not hasattr(args, param):
            if param == 'dropout':
                setattr(args, param, 0.0)
            elif param == 'activation':
                setattr(args, param, 'relu')
            elif param == 'atom_messages':
                setattr(args, param, False)
            elif param == 'undirected':
                setattr(args, param, False)
            else:
                missing_keys.append(param)

    if missing_keys:
        raise ValueError(f"🛑 [Config Error] args 缺失关键结构参数: {missing_keys}")

    if not hasattr(args, 'features_dim'):
        if hasattr(args, 'features_size'):
            args.features_dim = args.features_size
        else:
            if getattr(args, 'use_input_features', False):
                raise ValueError("🛑 [Logic Error] 开启了 use_input_features，但未找到 features_dim 或 features_size。")
            else:
                args.features_dim = 0

    if not hasattr(args, 'features_only'): setattr(args, 'features_only', False)
    if not hasattr(args, 'use_input_features'): setattr(args, 'use_input_features', False)

    if getattr(args, 'use_tapt', False):
        if not hasattr(args, 'prompt_dim'): setattr(args, 'prompt_dim', 128)
        if not hasattr(args, 'fg_dim'): setattr(args, 'fg_dim', 133)

    return True


# ============================================================
# CMPN 适配器 (Adapter Pattern)
# ============================================================

class CMPNAdapter(nn.Module):
    """
    适配器：确保 CMPN 的返回值格式一致

    CMPN.forward() 返回: (mol_vecs, atom_hiddens, a_scope)
    确保适配器也返回同样的格式
    """

    def __init__(self, cmpn_encoder):
        super(CMPNAdapter, self).__init__()
        self.cmpn = cmpn_encoder

    def forward(self, step, prompt_bool, batch):
        """
        调用 CMPN，接收三个返回值并确保格式一致

        Returns:
            Tuple[mol_vecs, atom_hiddens, a_scope]
        """
        result = self.cmpn.forward(step, prompt_bool, batch)

        # ✅ 确保总是返回三个值
        if isinstance(result, tuple):
            if len(result) == 3:
                return result  # (mol_vecs, atom_hiddens, a_scope)
            elif len(result) == 2:
                return (*result, None)  # (mol_vecs, atom_hiddens, None)
            else:
                return (result[0], None, None)  # (mol_vecs, None, None)
        else:
            return (result, None, None)  # (mol_vecs, None, None)


# ============================================================
# KANO 基础组件 (Attention & Prompt)
# ============================================================

def attention(query, key, value, mask, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class AttentionLayer(nn.Module):
    def __init__(self, args):
        super(AttentionLayer, self).__init__()
        self.hidden_size = args.hidden_size
        self.fg_dim = getattr(args, 'fg_dim', 133)
        self.w_q = nn.Linear(self.fg_dim, 32)
        self.w_k = nn.Linear(self.fg_dim, 32)
        self.w_v = nn.Linear(self.fg_dim, 32)
        self.dense = nn.Linear(32, self.fg_dim)
        self.LayerNorm = nn.LayerNorm(self.fg_dim, eps=1e-6)
        self.dropout = nn.Dropout(0.1)

    def forward(self, fg_hiddens, init_hiddens):
        query = self.w_q(fg_hiddens)
        key = self.w_k(fg_hiddens)
        value = self.w_v(fg_hiddens)
        padding_mask = (init_hiddens != 0) + 0.0
        mask = torch.matmul(padding_mask, padding_mask.transpose(-2, -1))
        x, attn = attention(query, key, value, mask)
        hidden_states = self.dense(x)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + fg_hiddens)
        return hidden_states


class Prompt_generator(nn.Module):
    def __init__(self, args):
        super(Prompt_generator, self).__init__()
        self.hidden_size = args.hidden_size
        self.fg_dim = getattr(args, 'fg_dim', 133)
        self.alpha = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.alpha.data.fill_(0.1)
        self.cls = nn.Parameter(torch.randn(1, self.fg_dim), requires_grad=True)
        self.linear = nn.Linear(self.fg_dim, self.hidden_size)
        self.attention_layer_1 = AttentionLayer(args)
        self.attention_layer_2 = AttentionLayer(args)
        self.norm = nn.LayerNorm(self.hidden_size)

    def forward(self, atom_hiddens, fg_states, atom_num, fg_indexs):
        device = fg_states.device
        for i in range(len(fg_indexs)):
            fg_states.scatter_(0, fg_indexs[i:i + 1], self.cls)
        hidden_states = self.attention_layer_1(fg_states, fg_states)
        hidden_states = self.attention_layer_2(hidden_states, fg_states)
        fg_out = torch.zeros(1, self.hidden_size).to(device)
        cls_hiddens = torch.gather(hidden_states, 0, fg_indexs)
        cls_hiddens = self.linear(cls_hiddens)
        fg_hiddens = torch.repeat_interleave(cls_hiddens, torch.tensor(atom_num).to(device), dim=0)
        fg_out = torch.cat((fg_out, fg_hiddens), 0)
        fg_out = self.norm(fg_out)
        return atom_hiddens + self.alpha * fg_out


class PromptGeneratorOutput(nn.Module):
    def __init__(self, args, self_output):
        super(PromptGeneratorOutput, self).__init__()
        self.self_out = self_output
        self.prompt_generator = Prompt_generator(args)

    def forward(self, hidden_states):
        return self.self_out(hidden_states)


def prompt_generator_output(args):
    return lambda self_output: PromptGeneratorOutput(args, self_output)


def add_functional_prompt(model, args):
    model.encoder.cmpn.encoder.W_i_atom = prompt_generator_output(args)(model.encoder.cmpn.encoder.W_i_atom)
    return model


# ============================================================
# 主模型 MoleculeModel
# ============================================================

class MoleculeModel(nn.Module):
    def __init__(self, classification: bool, multiclass: bool, pretrain: bool, args: Namespace = None):
        super(MoleculeModel, self).__init__()
        if args is None:
            raise ValueError("MoleculeModel 初始化必须提供 args 参数")

        self.args = args
        self.classification = classification
        if self.classification:
            self.sigmoid = nn.Sigmoid()
        self.multiclass = multiclass
        if self.multiclass:
            self.multiclass_softmax = nn.Softmax(dim=2)
        self.pretrain = pretrain

        self.mode = 'mode1'
        self.tapt_module = None
        self.use_tapt = False
        self.noise_scale = 0.01
        self.tapt_alpha = 0.001
        self._tapt_proj = None
        self._last_atom_hiddens = None
        self._last_a_scope = None

    def create_encoder(self, args: Namespace, encoder_name):
        """创建编码器并使用适配器包装"""
        if encoder_name == 'CMPNN':
            cmpn = CMPN(
                args=args,
                atom_fdim=getattr(args, 'atom_fdim', 133),
                bond_fdim=getattr(args, 'bond_fdim', 147)
            )
            # ✅ 用适配器包装 CMPN，确保返回值一致
            self.encoder = CMPNAdapter(cmpn)
        elif encoder_name == 'MPNN':
            self.encoder = MPN(args)
        else:
            raise ValueError(f"Unknown encoder: {encoder_name}")

    def create_ffn(self, args: Namespace):
        self.multiclass = args.dataset_type == 'multiclass'
        if self.multiclass:
            self.num_classes = args.multiclass_num_classes

        if getattr(args, 'features_only', False):
            first_linear_dim = args.features_size
        else:
            first_linear_dim = args.hidden_size
            if getattr(args, 'use_input_features', False):
                first_linear_dim += args.features_dim

        dropout = nn.Dropout(args.dropout)
        activation = get_activation_function(args.activation)

        ffn_hidden_size = getattr(args, 'ffn_hidden_size', args.hidden_size)
        ffn_num_layers = getattr(args, 'ffn_num_layers', 2)
        output_size = args.output_size

        if ffn_num_layers == 1:
            ffn = [dropout, nn.Linear(first_linear_dim, output_size)]
        else:
            ffn = [dropout, nn.Linear(first_linear_dim, ffn_hidden_size)]
            for _ in range(ffn_num_layers - 2):
                ffn.extend([activation, dropout, nn.Linear(ffn_hidden_size, ffn_hidden_size)])
            ffn.extend([activation, dropout, nn.Linear(ffn_hidden_size, output_size)])
        self.ffn = nn.Sequential(*ffn)

    def forward(self, *input, task_id: Optional[int] = 0, use_tapt: bool = None, mode: str = None):
        """
        前向传播 (FIX: 正确处理 CMPN 的三返回值)

        Args:
            *input: 可变长参数，包含 step, prompt, batch 等
            task_id: 任务 ID
            use_tapt: 是否使用 TAPT
            mode: 模式名称

        Returns:
            output: 预测输出
        """
        _use_tapt = use_tapt if use_tapt is not None else self.use_tapt
        current_mode = mode if mode else self.mode

        # ===== Input 解析与严格类型清洗 =====
        step = 'functional_prompt'
        prompt_raw = False
        batch = None

        if len(input) == 4:
            step, prompt_raw, batch, _ = input
        elif len(input) == 3:
            step, prompt_raw, batch = input
        elif len(input) == 1:
            batch = input[0]
            prompt_raw = False
        else:
            for arg in input:
                if hasattr(arg, 'get_components') or isinstance(arg, (list, tuple)):
                    batch = arg
                    break
            prompt_raw = False

        if batch is None:
            raise ValueError(f"❌ [Model Error] forward() 无法从输入中解析出 Batch 数据。Input: {input}")

        # ✅ 清洗 prompt 参数
        if isinstance(prompt_raw, (bool, int)):
            prompt_bool = bool(prompt_raw)
        else:
            prompt_bool = False

        # ===== Encoder (FIX: 正确处理返回值) =====
        encoder_outputs = self.encoder(step, prompt_bool, batch)

        # ✅ 【关键修复】：CMPN 总是返回三个值 (mol_vecs, atom_hiddens, a_scope)
        if isinstance(encoder_outputs, tuple):
            if len(encoder_outputs) == 3:
                mol_vecs, atom_hiddens, a_scope = encoder_outputs
            elif len(encoder_outputs) == 2:
                mol_vecs, atom_hiddens = encoder_outputs
                a_scope = None
            else:
                mol_vecs = encoder_outputs[0]
                atom_hiddens = None
                a_scope = None
        else:
            # 应该不会到这里，但为了安全起见
            mol_vecs = encoder_outputs
            atom_hiddens = None
            a_scope = None

        self._last_atom_hiddens = atom_hiddens
        self._last_a_scope = a_scope

        # ===== TAPT 逻辑 =====
        if _use_tapt and self.tapt_module is not None:
            device = mol_vecs.device

            # 获取 ElementKG 特征
            batch_graph = batch if hasattr(batch, 'f_fgs') else None
            if batch_graph is None:
                try:
                    batch_graph = mol2graph(batch, self.args, prompt_bool)
                except:
                    pass

            fg_states = None
            if batch_graph is not None:
                if hasattr(batch_graph, 'f_fgs'):
                    fg_states = batch_graph.f_fgs
                elif hasattr(batch_graph, 'get_components'):
                    for comp in batch_graph.get_components():
                        if isinstance(comp, torch.Tensor) and comp.dim() == 2 and comp.size(-1) == 133:
                            fg_states = comp
                            break

            # Scope 处理
            scope = a_scope if a_scope is not None else getattr(batch_graph, 'a_scope', None)
            if scope:
                batch_idx = torch.tensor([i for i, (s, l) in enumerate(scope) for _ in range(l)], dtype=torch.long,
                                         device=device)
            else:
                num_mols = mol_vecs.size(0)
                total_atoms = fg_states.size(0) if fg_states is not None else num_mols * 20
                per_mol = max(1, total_atoms // num_mols)
                batch_idx = torch.arange(num_mols, device=device).repeat_interleave(per_mol)
                if batch_idx.size(0) < total_atoms:
                    batch_idx = torch.cat(
                        [batch_idx, torch.full((total_atoms - batch_idx.size(0),), num_mols - 1, device=device)])
                elif batch_idx.size(0) > total_atoms:
                    batch_idx = batch_idx[:total_atoms]

            num_atoms = batch_idx.size(0)
            tapt_input = None

            # Mode 分流
            if current_mode in ['task_only', 'mode2', 'kano_task', 'mode3']:
                tapt_input = torch.randn(num_atoms, 133, device=device) * self.noise_scale
            elif current_mode in ['kano_kg_task', 'mode4']:
                if fg_states is not None:
                    # ✅ 关键修复：确保 fg_states 在正确的设备上
                    fg_states = fg_states.to(device)

                    if fg_states.size(0) != num_atoms:
                        if fg_states.size(0) > num_atoms:
                            fg_states = fg_states[:num_atoms]
                        else:
                            fg_states = torch.cat(
                                [fg_states, torch.zeros(num_atoms - fg_states.size(0), 133, device=device)], dim=0)
                    tapt_input = fg_states
                else:
                    print("🛑 [Warning] Mode 4 缺失 ElementKG 特征，实验准确性可能受影响！(回退到噪声)")
                    tapt_input = torch.randn(num_atoms, 133, device=device) * 0.01

            # TAPT 前向
            if tapt_input is not None:
                fused_prompt = self.tapt_module.generate_prompt(
                    fg_states=tapt_input,
                    batch_idx=batch_idx,
                    task_id=task_id
                )

                if self._tapt_proj is None:
                    self._tapt_proj = nn.Linear(fused_prompt.size(1), mol_vecs.size(1), bias=False).to(device)
                    nn.init.normal_(self._tapt_proj.weight, 0.0, 0.0001)
                else:
                    self._tapt_proj = self._tapt_proj.to(device)

                mol_vecs = mol_vecs + self.tapt_alpha * self._tapt_proj(fused_prompt)

        # ===== FFN =====
        output = self.ffn(mol_vecs)
        if self.classification and not self.training:
            output = self.sigmoid(output)
        if self.multiclass:
            output = output.reshape((output.size(0), -1, self.num_classes))
            if not self.training:
                output = self.multiclass_softmax(output)
        return output


# ============================================================
# 工厂构建函数
# ============================================================

def build_model(args: Namespace, encoder_name: str) -> nn.Module:
    validate_and_ensure_args(args, 'base')
    args.output_size = args.num_tasks * (args.multiclass_num_classes if args.dataset_type == 'multiclass' else 1)
    model = MoleculeModel(args.dataset_type == 'classification', args.dataset_type == 'multiclass', False, args)
    model.create_encoder(args, encoder_name)
    model.create_ffn(args)
    initialize_weights(model)
    return model


def build_pretrain_model(args: Namespace, encoder_name: str, num_tasks: int = None) -> nn.Module:
    validate_and_ensure_args(args, 'pretrain')
    args.ffn_hidden_size = args.hidden_size // 2
    args.output_size = args.hidden_size
    model = MoleculeModel(args.dataset_type == 'classification', args.dataset_type == 'multiclass', True, args)
    model.create_encoder(args, encoder_name)
    model.create_ffn(args)
    initialize_weights(model)
    return model


def build_model_mode1_pure_kano(args: Namespace, encoder_name: str, num_tasks: int) -> nn.Module:
    print("\n" + "=" * 60 + "\n🎯 [Mode 1] Pure KANO (Baseline)\n" + "=" * 60)
    validate_and_ensure_args(args, 'mode1')
    args.step = 'functional_prompt'
    model = build_model(args, encoder_name)
    model.mode, model.use_tapt = 'mode1', False
    return add_functional_prompt(model, args)


def build_model_mode2_task_only(args: Namespace, encoder_name: str, num_tasks: int) -> nn.Module:
    print("\n" + "=" * 60 + "\n🎯 [Mode 2] Task-Only\n" + "=" * 60)
    validate_and_ensure_args(args, 'mode2')
    args.step = 'pretrain'
    model = build_model(args, encoder_name)
    model.mode, model.use_tapt = 'mode2', True
    return add_tapt_prompt(model, args, num_tasks)


def build_model_mode3_kano_task(args: Namespace, encoder_name: str, num_tasks: int) -> nn.Module:
    print("\n" + "=" * 60 + "\n🎯 [Mode 3] KANO + Task\n" + "=" * 60)
    validate_and_ensure_args(args, 'mode3')
    args.step = 'functional_prompt'
    model = build_model(args, encoder_name)
    model.mode, model.use_tapt = 'mode3', True
    model = add_functional_prompt(model, args)
    return add_tapt_prompt(model, args, num_tasks)


def build_model_mode4_kano_kg_task(args: Namespace, encoder_name: str, num_tasks: int) -> nn.Module:
    print("\n" + "=" * 60 + "\n🎯 [Mode 4] KANO + KG + Task\n" + "=" * 60)
    validate_and_ensure_args(args, 'mode4')
    args.step = 'functional_prompt'
    model = build_model(args, encoder_name)
    model.mode, model.use_tapt = 'mode4', True
    model = add_functional_prompt(model, args)
    return add_tapt_prompt(model, args, num_tasks)


def add_tapt_prompt(model: MoleculeModel, args: Namespace, num_tasks: int) -> MoleculeModel:
    tapt_module = TAPTPromptModule(
        num_tasks=num_tasks,
        fg_dim=getattr(args, 'fg_dim', 133),
        prompt_dim=getattr(args, 'prompt_dim', 128),
        hidden_dim=getattr(args, 'tapt_hidden_dim', 256)
    )
    model.tapt_module = tapt_module
    model.tapt_alpha = getattr(args, 'tapt_alpha', 0.001)
    model.noise_scale = getattr(args, 'noise_scale', 0.01)
    if hasattr(args, 'device'):
        model.tapt_module = model.tapt_module.to(args.device)
    return model


def build_tapt_model(args: Namespace, encoder_name: str, num_tasks: int, mode: str = 'mode1') -> nn.Module:
    mode = getattr(args, 'mode', mode).lower()
    mode_map = {
        'mode1': build_model_mode1_pure_kano, 'pure_kano': build_model_mode1_pure_kano,
        'mode2': build_model_mode2_task_only, 'task_only': build_model_mode2_task_only,
        'mode3': build_model_mode3_kano_task, 'kano_task': build_model_mode3_kano_task,
        'mode4': build_model_mode4_kano_kg_task, 'kano_kg_task': build_model_mode4_kano_kg_task
    }
    if mode not in mode_map:
        raise ValueError(f"Unknown mode: {mode}")
    return mode_map[mode](args, encoder_name, num_tasks)


__all__ = [
    'MoleculeModel', 'build_model', 'build_pretrain_model', 'build_tapt_model',
    'build_model_mode1_pure_kano', 'build_model_mode2_task_only',
    'build_model_mode3_kano_task', 'build_model_mode4_kano_kg_task',
    'CMPNAdapter'
]
