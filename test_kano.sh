#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate kano

echo "🧪 KANO环境全面测试"
echo "==================="

# 基础测试
python -c "
import sys, os
print(f'🐍 Python: {sys.version}')
print(f'📍 环境: \$CONDA_DEFAULT_ENV') 
print(f'💾 执行路径: {sys.executable}')
print(f'📂 工作目录: {os.getcwd()}')
print()

# 导入测试
test_imports = [
    ('torch', 'PyTorch深度学习'),
    ('numpy', '数值计算'),
    ('pandas', '数据处理'),
    ('matplotlib', '绘图'),
    ('sklearn', '机器学习'),
    ('rdkit.Chem', 'RDKit化学'),
    ('Bio', '生物信息'),
    ('networkx', '图论'),
    ('gensim', '自然语言处理'),
    ('xgboost', 'XGBoost'),
    ('jupyter', 'Jupyter'),
]

print('🔍 功能模块测试:')
for module, desc in test_imports:
    try:
        __import__(module)
        print(f'  ✅ {desc}: 正常')
    except ImportError:
        print(f'  ❌ {desc}: 缺失') 
    except Exception as e:
        print(f'  ⚠️  {desc}: 异常({str(e)[:30]})')

# 简单功能测试
print()
print('⚡ 功能测试:')

try:
    import torch
    x = torch.randn(3, 3)
    print(f'  ✅ PyTorch张量运算: {x.shape}')
except:
    print('  ❌ PyTorch张量运算失败')

try:
    import numpy as np
    import pandas as pd
    df = pd.DataFrame(np.random.randn(5, 3))
    print(f'  ✅ Pandas数据框: {df.shape}')
except:
    print('  ❌ Pandas数据框失败')

try:
    from rdkit import Chem
    mol = Chem.MolFromSmiles('CCO')
    print(f'  ✅ RDKit分子解析: {mol.GetNumAtoms()}原子')
except:
    print('  ❌ RDKit分子解析失败')

print()
print('🎉 环境测试完成！')
"
