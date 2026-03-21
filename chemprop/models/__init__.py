
from .loss.loss import ContrastiveLoss

# ===== chemprop/models/__init__.py (完整导出版) =====

# 1. 核心模型与构建函数
from .model import (
    MoleculeModel,
    build_model,
    build_pretrain_model,          # ✅ 必须加：修复 pretrain 报错
    build_tapt_model,              # ✅ 必须加：TAPT 入口
    add_functional_prompt,         # ✅ 必须加：KANO 入口
    add_tapt_prompt,               # ✅ 必须加：TAPT 模块
    # 以下为四模式构建函数（可选导出，但建议加上以防万一）
    build_model_mode1_pure_kano,
    build_model_mode2_task_only,
    build_model_mode3_kano_task,
    build_model_mode4_kano_kg_task
)

# 2. 编码器
from .mpn import MPN
from .cmpn import CMPN

# 3. 导出列表 (__all__)
__all__ = [
    'MoleculeModel',
    'build_model',
    'build_pretrain_model',
    'build_tapt_model',
    'add_functional_prompt',
    'add_tapt_prompt',
    'build_model_mode1_pure_kano',
    'build_model_mode2_task_only',
    'build_model_mode3_kano_task',
    'build_model_mode4_kano_kg_task',
    'MPN',
    'CMPN'
]