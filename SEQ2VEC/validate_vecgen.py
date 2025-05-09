import os
import torch
import logging
from vecgen import parse_mol2, process_folder
from gensim.models import FastText
from joblib import load
import numpy as np
from collections import defaultdict
from torch_geometric.data import Data
from torch.serialization import get_safe_globals
from torch_geometric.data.data import DataEdgeAttr

# 添加安全类
# get_safe_globals(DataEdgeAttr)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("validate_vecgen.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_models(model_dir):
    """加载FastText模型和标准化器"""
    ft_model_path = os.path.join(model_dir, "global_fasttext.model")
    scaler_path = os.path.join(model_dir, "global_scaler.joblib")
    
    if not os.path.exists(ft_model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError("模型文件不存在")
    
    ft_model = FastText.load(ft_model_path)
    scaler = load(scaler_path)
    return ft_model, scaler

def find_closest_word(vector, ft_model):
    """找到最接近给定向量的词"""
    return ft_model.wv.similar_by_vector(vector, topn=1)[0][0]

def reconstruct_mol2_from_pt(pt_file, ft_model, scaler, vector_size=128):
    """从PT文件重建mol2格式数据"""
    try:
        data = torch.load(pt_file, weights_only=False, map_location=torch.device('cpu'))
        
        # 解析原子特征
        atom_features = data.x.numpy()
        num_atoms = data.num_nodes
        
        # 分离特征
        name_vectors = atom_features[:, :vector_size]
        type_vectors = atom_features[:, vector_size:2*vector_size]
        substruct_name_vectors = atom_features[:, 2*vector_size:3*vector_size]
        substruct_num_vectors = atom_features[:, 3*vector_size:4*vector_size]
        charges = atom_features[:, 4*vector_size:]
        
        # 还原坐标
        coords = scaler.inverse_transform(data.y.numpy())
        
        # 还原原子信息
        atoms = []
        for i in range(num_atoms):
            atom_name = find_closest_word(name_vectors[i], ft_model)
            atom_type = find_closest_word(type_vectors[i], ft_model)
            substruct_base = find_closest_word(substruct_name_vectors[i], ft_model)
            
            # 处理子结构编号
            if not np.allclose(substruct_num_vectors[i], 0):
                substruct_num = find_closest_word(substruct_num_vectors[i], ft_model)
                substruct_name = f"{substruct_base}{substruct_num}"
            else:
                substruct_name = substruct_base
            
            atom_info = {
                'atom_id': i + 1,
                'atom_name': atom_name,
                'atom_type': atom_type,
                'substruct_name': substruct_name,
                'x': coords[i][0],
                'y': coords[i][1],
                'z': coords[i][2],
                'charge': float(scaler.inverse_transform(charges[i].reshape(1, -1))[0][0])
            }
            atoms.append(atom_info)
        
        # 还原键信息
        bonds = []
        if data.edge_index.shape[1] > 0:
            edge_index = data.edge_index.numpy()
            edge_attr = data.edge_attr.numpy()
            
            for i in range(edge_index.shape[1]):
                bond_type = find_closest_word(edge_attr[i], ft_model)
                bond_info = {
                    'atom1_id': edge_index[0, i] + 1,
                    'atom2_id': edge_index[1, i] + 1,
                    'bond_type': bond_type
                }
                bonds.append(bond_info)
        
        return atoms, bonds
    except Exception as e:
        logger.error(f"重建mol2数据时出错: {str(e)}")
        return None, None

def compare_mol2_data(original_atoms, original_bonds, reconstructed_atoms, reconstructed_bonds):
    """比较原始和重建的mol2数据"""
    differences = defaultdict(list)
    
    # 比较原子数量
    if len(original_atoms) != len(reconstructed_atoms):
        differences['atom_count'].append(
            f"原子数量不匹配: 原始={len(original_atoms)}, 重建={len(reconstructed_atoms)}")
        return differences
    
    # 比较键数量
    if len(original_bonds) != len(reconstructed_bonds):
        differences['bond_count'].append(
            f"键数量不匹配: 原始={len(original_bonds)}, 重建={len(reconstructed_bonds)}")
    
    # 比较原子信息
    for i, (orig, recon) in enumerate(zip(original_atoms, reconstructed_atoms)):
        for key in ['atom_name', 'atom_type', 'substruct_name']:
            if orig[key] != recon[key]:
                differences[f'atom_{key}'].append(
                    f"原子 {i+1}: 原始={orig[key]}, 重建={recon[key]}")
        
        # 比较数值特征（允许小误差）
        for key in ['x', 'y', 'z', 'charge']:
            if abs(orig[key] - recon[key]) > 0.01:
                differences[f'atom_{key}'].append(
                    f"原子 {i+1}: 原始={orig[key]:.4f}, 重建={recon[key]:.4f}")
    
    # 比较键信息
    bond_pairs_orig = {(b['atom1_id'], b['atom2_id']): b['bond_type'] for b in original_bonds}
    bond_pairs_recon = {(b['atom1_id'], b['atom2_id']): b['bond_type'] for b in reconstructed_bonds}
    
    for (a1, a2), btype in bond_pairs_orig.items():
        if (a1, a2) not in bond_pairs_recon:
            differences['missing_bonds'].append(f"缺失键: {a1}-{a2} ({btype})")
        elif bond_pairs_recon[(a1, a2)] != btype:
            differences['bond_type'].append(
                f"键类型不匹配 {a1}-{a2}: 原始={btype}, 重建={bond_pairs_recon[(a1, a2)]}")
    
    return differences

def validate_conversion(mol2_file, pt_file, ft_model, scaler):
    """验证单个文件的转换"""
    logger.info(f"验证文件: {mol2_file}")
    
    # 读取原始mol2文件
    original_atoms, original_bonds = parse_mol2(mol2_file)
    if not original_atoms:
        logger.error("无法读取原始mol2文件")
        return False
    
    # 从PT文件重建mol2数据
    reconstructed_atoms, reconstructed_bonds = reconstruct_mol2_from_pt(pt_file, ft_model, scaler)
    if not reconstructed_atoms:
        logger.error("无法从PT文件重建数据")
        return False
    
    # 比较数据
    differences = compare_mol2_data(original_atoms, original_bonds, reconstructed_atoms, reconstructed_bonds)
    
    if not differences:
        logger.info("验证通过：数据完全匹配")
        return True
    else:
        logger.warning("发现差异:")
        for key, diffs in differences.items():
            for diff in diffs:
                logger.warning(f"  {key}: {diff}")
        return False

def main():
    # 配置路径
    input_dir = r"E:/APTAMER-GEN/mol2"
    model_dir = r"E:/APTAMER-GEN/models/vecgen-model/all"
    
    try:
        # 加载模型
        ft_model, scaler = load_models(model_dir)
        logger.info("模型加载成功")
        
        # 获取所有mol2文件
        mol2_files = [f for f in os.listdir(input_dir) if f.endswith('.mol2')]
        if not mol2_files:
            logger.error("没有找到mol2文件")
            return
        
        # 验证结果统计
        total = len(mol2_files)
        success = 0
        failed = 0
        
        # 验证每个文件
        for mol2_file in mol2_files:
            mol2_path = os.path.join(input_dir, mol2_file)
            pt_file = os.path.join(model_dir, f"{os.path.splitext(mol2_file)[0]}_graph.pt")
            
            if not os.path.exists(pt_file):
                logger.warning(f"PT文件不存在: {pt_file}")
                failed += 1
                continue
            
            if validate_conversion(mol2_path, pt_file, ft_model, scaler):
                success += 1
            else:
                failed += 1
        
        # 输出总结
        logger.info("\n验证总结:")
        logger.info(f"总文件数: {total}")
        logger.info(f"验证成功: {success}")
        logger.info(f"验证失败: {failed}")
        logger.info(f"成功率: {(success/total)*100:.2f}%")
        
    except Exception as e:
        logger.error(f"验证过程出错: {str(e)}")

if __name__ == "__main__":
    main()