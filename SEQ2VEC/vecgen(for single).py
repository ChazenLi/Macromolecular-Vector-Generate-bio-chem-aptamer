import os
import re
import numpy as np
from gensim.models import FastText
from sklearn.preprocessing import StandardScaler
import torch

# ------------------------------
# 步骤1: 从文件路径提取模型名称
# ------------------------------
def get_model_name(file_path):
    # 提取文件名（如 "7ZQS-1.mol2" → "7ZQS-1"）
    base_name = os.path.basename(file_path)          # 获取文件名（带扩展名）
    model_name = os.path.splitext(base_name)[0]      # 去除扩展名
    return model_name

# ------------------------------
# 步骤2: 解析 mol2 文件并分割子结构名
# ------------------------------
def parse_mol2(file_path):
    atoms = []
    bonds = []
    current_section = None

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('@<TRIPOS>ATOM'):
                current_section = 'ATOM'
                continue
            elif line.startswith('@<TRIPOS>BOND'):
                current_section = 'BOND'
                continue
            elif line.startswith('@<TRIPOS>'):
                current_section = None
                continue

            if current_section == 'ATOM':
                parts = line.split()
                if len(parts) >= 9:
                    atom_info = {
                        'atom_name': parts[1],
                        'atom_type': parts[5],
                        'substruct_name': parts[7],  # e.g., Lys1234
                        'x': float(parts[2]),
                        'y': float(parts[3]),
                        'z': float(parts[4]),
                        'charge': float(parts[8])
                    }
                    atoms.append(atom_info)
            elif current_section == 'BOND':
                parts = line.split()
                if len(parts) >= 4:
                    bonds.append({'bond_type': parts[3]})

    return atoms, bonds

# ------------------------------
# 步骤2: 分割子结构名为字母和数字部分
# ------------------------------
def split_substruct_name(name):
    match = re.match(r'^([A-Za-z]+)(\d+)$', name)
    if match:
        return [match.group(1), match.group(2)]
    else:
        return [name]

# ------------------------------
# 步骤3: 构建训练数据（显式分割子结构名）
# ------------------------------
def build_fasttext_data(atoms, bonds):
    atom_sentences = []
    for atom in atoms:
        # 分割子结构名（如 DA11 → ['DA', '11']）
        substruct_parts = split_substruct_name(atom['substruct_name'])
        sentence = [atom['atom_name'], atom['atom_type']] + substruct_parts
        atom_sentences.append(sentence)
    
    bond_sentences = [[bond['bond_type']] for bond in bonds]
    return atom_sentences + bond_sentences

# ------------------------------
# 步骤4: 训练支持长字符和大小写的 FastText
# ------------------------------
def train_fasttext(sentences, vector_size=128):
    model = FastText(
        sentences,
        vector_size=vector_size,
        window=3,
        min_count=1,
        workers=4,
        min_n=1,       # 允许 1 字符子词（如 'L', '1'）
        max_n=7,       # 处理长子结构名（如 123456）
        bucket=2000000,
        word_ngrams=1,
        sg=1,
        sorted_vocab=0
    )
    return model

# ------------------------------
# 步骤5: 生成原子特征（固定维度）
# ------------------------------
def generate_atom_features(atoms, model, vector_size):
    # 数值特征标准化
    numeric_features = np.array([
        [atom['x'], atom['y'], atom['z'], atom['charge']] for atom in atoms
    ])
    scaler = StandardScaler()
    numeric_features = scaler.fit_transform(numeric_features)
    
    # 词向量拼接（原子名、原子类型、子结构字母、子结构数字）
    word_vectors = []
    for atom in atoms:
        # 原子名和类型
        vec_name = model.wv[atom['atom_name']]
        vec_type = model.wv[atom['atom_type']]
        
        # 子结构名分割处理
        substruct_parts = split_substruct_name(atom['substruct_name'])
        vec_substruct = []
        for part in substruct_parts:
            vec_substruct.append(model.wv[part])
        # 补齐为两部分（若无数字部分，第二部分补零）
        if len(substruct_parts) == 1:
            vec_substruct.append(np.zeros(vector_size))
        
        # 拼接向量：[原子名] + [原子类型] + [子结构字母] + [子结构数字]
        combined_vec = np.concatenate([vec_name, vec_type, vec_substruct[0], vec_substruct[1]])
        word_vectors.append(combined_vec)
    
    # 合并所有特征
    word_vectors = np.array(word_vectors)
    full_features = np.concatenate([word_vectors, numeric_features], axis=1)
    return torch.tensor(full_features, dtype=torch.float32)

# ------------------------------
# 主流程（支持自定义保存路径）
# ------------------------------
if __name__ == "__main__":
    # 输入参数
    mol2_file = r"E:/APTAMER-GEN/mol2/7ZQS-1.mol2"  # 替换为你的文件路径
    save_dir = r"E:/APTAMER-GEN/models/vecgen-model"              # 指定保存目录
    
    # 1. 提取模型名称（如 "7ZQS-1"）
    model_name = get_model_name(mol2_file)
    print(f"模型名称: {model_name}")
    
    # 2. 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 3. 解析数据
    atoms, bonds = parse_mol2(mol2_file)
    
    # 4. 构建训练数据
    sentences = build_fasttext_data(atoms, bonds)
    
    # 5. 训练模型
    model = train_fasttext(sentences, vector_size=128)
    
    # 6. 生成特征
    atom_features = generate_atom_features(atoms, model, 128)
    bond_features = [model.wv[bond['bond_type']] for bond in bonds]
    
    # 7. 保存模型和特征到指定路径
    model_save_path = os.path.join(save_dir, f"{model_name}_fasttext.model")
    atom_save_path = os.path.join(save_dir, f"{model_name}_atom_features.pt")
    bond_save_path = os.path.join(save_dir, f"{model_name}_bond_features.pt")
    
    model.save(model_save_path)
    torch.save(atom_features, atom_save_path)
    torch.save(bond_features, bond_save_path)
    
    print(f"模型已保存至: {model_save_path}")
    print(f"原子特征已保存至: {atom_save_path}")
    print(f"键特征已保存至: {bond_save_path}")
