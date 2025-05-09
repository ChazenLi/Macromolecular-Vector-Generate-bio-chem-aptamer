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
safe_classes = get_safe_globals()
safe_classes.append(DataEdgeAttr)

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

def validate_specific_data(mol2_file, pt_file, ft_model, scaler, atom_id=None, data_type='atom'):
    """验证特定的数据组合"""
    # 读取原始和重建的数据
    original_atoms, original_bonds = parse_mol2(mol2_file)
    reconstructed_atoms, reconstructed_bonds = reconstruct_mol2_from_pt(pt_file, ft_model, scaler)
    
    if not original_atoms or not reconstructed_atoms:
        logger.error("无法读取或重建数据")
        return
    
    if data_type == 'atom' and atom_id is not None:
        if 1 <= atom_id <= len(original_atoms):
            orig = original_atoms[atom_id-1]
            recon = reconstructed_atoms[atom_id-1]
            
            logger.info(f"\n原子 {atom_id} 的详细比对:")
            logger.info("原始数据:")
            for key, value in orig.items():
                logger.info(f"  {key}: {value}")
            
            logger.info("\n重建数据:")
            for key, value in recon.items():
                logger.info(f"  {key}: {value}")
            
            # 计算数值特征的误差
            for key in ['x', 'y', 'z', 'charge']:
                error = abs(orig[key] - recon[key])
                logger.info(f"\n{key} 误差: {error:.6f}")
            
            # 检查非数值特征的匹配
            for key in ['atom_name', 'atom_type', 'substruct_name']:
                match = orig[key] == recon[key]
                logger.info(f"{key} 匹配: {match}")
                
    elif data_type == 'bond':
        # 验证特定的键
        orig_bonds = {(b['atom1_id'], b['atom2_id']): b for b in original_bonds}
        recon_bonds = {(b['atom1_id'], b['atom2_id']): b for b in reconstructed_bonds}
        
        logger.info("\n键的详细比对:")
        for (a1, a2), orig_bond in orig_bonds.items():
            if (a1, a2) in recon_bonds:
                recon_bond = recon_bonds[(a1, a2)]
                logger.info(f"\n键 {a1}-{a2}:")
                logger.info(f"  原始类型: {orig_bond['bond_type']}")
                logger.info(f"  重建类型: {recon_bond['bond_type']}")
                logger.info(f"  匹配: {orig_bond['bond_type'] == recon_bond['bond_type']}")

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

def validate_custom_input(ft_model, scaler, num_samples=5):
    """验证自定义输入数据"""
    logger.info("\n开始自定义输入验证...")
    
    while True:
        print("\n请选择输入类型:")
        print("1. 原子名称")
        print("2. 原子类型")
        print("3. 子结构名称")
        print("4. 坐标和电荷")
        print("5. 查看模型参数")
        print("6. 退出")
        
        choice = input("请输入选择 (1-6): ")
        
        if choice == '6':
            break

        if choice not in ['1', '2', '3', '4', '5']:
            logger.warning("无效的选择，请重新输入")
            continue
            
        if choice == '1':
            word = input("请输入原子名称: ")
            if word in ft_model.wv:
                vector = ft_model.wv[word]
                similar = ft_model.wv.most_similar([vector], topn=3)
                
                logger.info(f"\n原子名称: {word}")
                logger.info(f"向量还原 - 最相似: {similar[0][0]} (相似度: {similar[0][1]:.4f})")
                logger.info(f"其他相似名称: {similar[1:3]}")
                logger.info(f"还原正确: {similar[0][0] == word}")
            else:
                logger.warning("输入的原子名称不在词汇表中")
                
        elif choice == '2':
            word = input("请输入原子类型: ")
            if word in ft_model.wv:
                vector = ft_model.wv[word]
                similar = ft_model.wv.most_similar([vector], topn=3)
                
                logger.info(f"\n原子类型: {word}")
                logger.info(f"向量还原 - 最相似: {similar[0][0]} (相似度: {similar[0][1]:.4f})")
                logger.info(f"其他相似类型: {similar[1:3]}")
                logger.info(f"还原正确: {similar[0][0] == word}")
            else:
                logger.warning("输入的原子类型不在词汇表中")
                
        elif choice == '3':
            word = input("请输入子结构名称: ")
            if word in ft_model.wv:
                vector = ft_model.wv[word]
                similar = ft_model.wv.most_similar([vector], topn=3)
                
                logger.info(f"\n子结构名称: {word}")
                logger.info(f"向量还原 - 最相似: {similar[0][0]} (相似度: {similar[0][1]:.4f})")
                logger.info(f"其他相似名称: {similar[1:3]}")
                logger.info(f"还原正确: {similar[0][0] == word}")
            else:
                logger.warning("输入的子结构名称不在词汇表中")
                
        elif choice == '4':
            try:
                print("请输入坐标和电荷 (格式: x y z charge):")
                coords = list(map(float, input().split()))
                if len(coords) != 4:
                    logger.warning("输入格式错误，需要4个数值")
                    continue
                    
                # 标准化输入
                scaled_coords = scaler.transform(np.array([coords]))
                # 还原数据
                restored_coords = scaler.inverse_transform(scaled_coords)
                
                logger.info("\n坐标和电荷验证:")
                logger.info(f"原始值: {coords}")
                logger.info(f"还原值: {restored_coords[0]}")
                logger.info(f"误差: {np.abs(coords - restored_coords[0])}")
            except ValueError:
                logger.warning("输入格式错误，请确保输入的是数字")

        elif choice == '5':
            print("\n请选择要查看的参数:")
            print("1. 词汇表大小和示例")
            print("2. 向量维度")
            print("3. 标准化器参数")
            print("4. 打印完整词汇表")
            print("5. 返回上级菜单")
            
            param_choice = input("请输入选择 (1-5): ")
            
            if param_choice == '1':
                vocab = ft_model.wv.index_to_key
                vocab_size = len(vocab)
                logger.info(f"\n词汇表大小: {vocab_size}")
                sample_size = min(100, vocab_size)
                logger.info(f"词汇表示例 (随机{sample_size}个):")
                for word in np.random.choice(vocab, sample_size, replace=False):
                    logger.info(f"  - {word}")
                    
            elif param_choice == '2':
                vector_size = ft_model.vector_size
                logger.info(f"\n向量维度: {vector_size}")
                logger.info(f"窗口大小: {ft_model.window}")
                logger.info(f"最小词频: {ft_model.min_count}")
                
            elif param_choice == '3':
                logger.info("\n标准化器参数:")
                logger.info(f"特征范围: {scaler.feature_range}")
                logger.info(f"均值: {scaler.mean_}")
                logger.info(f"方差: {scaler.scale_}")
                logger.info(f"数据最小值: {scaler.data_min_}")
                logger.info(f"数据最大值: {scaler.data_max_}")
                
            elif param_choice == '4':
                vocab = ft_model.wv.index_to_key
                # 创建词汇分类
                vocab_data = []
                for word in vocab:
                    # 确定词的类型
                    if word.startswith(('C', 'H', 'O', 'N', 'P', 'S')) and len(word) <= 3:
                        word_type = "atom_name"
                    elif '@' in word or '.' in word or '*' in word:
                        word_type = "atom_type"
                    elif word in ['ar', 'am', '1', '2', '3']:
                        word_type = "bond_type"
                    elif word.isdigit() or word[1:].isdigit():
                        word_type = "substruct_number"
                    else:
                        word_type = "other"
                    
                    # 获取词向量
                    vector = ft_model.wv[word]
                    vocab_data.append([word, word_type, vector.tolist()])
                
                # 保存到CSV文件
                import csv
                import json
                csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vocabulary.csv')
                
                with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Word', 'Type', 'Vector'])
                    for item in vocab_data:
                        # 将向量转换为JSON字符串以便存储
                        writer.writerow([item[0], item[1], json.dumps(item[2])])
                
                logger.info(f"\n词汇表已保存到: {csv_path}")
                logger.info(f"总词汇量: {len(vocab)}")
                
            elif param_choice == '5':
                continue
                
            else:
                logger.warning("无效的选择")

def main():
    # 配置路径
    model_dir = r"E:/APTAMER-GEN/models/vecgen-model/all"
    
    try:
        # 加载模型
        ft_model, scaler = load_models(model_dir)
        logger.info("模型加载成功")
        
        # 直接进入自定义输入验证
        validate_custom_input(ft_model, scaler)
                
    except Exception as e:
        logger.error(f"程序运行出错: {str(e)}")

if __name__ == "__main__":
    main()