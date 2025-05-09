
import os
import re
import numpy as np
from gensim.models import FastText
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
import torch
from torch_geometric.data import Data
from tqdm import tqdm
import logging
import time
import random

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("vecgen_process.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ------------------------------
# 工具函数
# ------------------------------
def parse_mol2(file_path):
    """解析 mol2 文件，返回原子和键信息"""
    atoms = []
    bonds = []
    current_section = None
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
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
                    # 使用更灵活的方式解析原子行
                    try:
                        parts = line.split()
                        if len(parts) >= 9:
                            # 确保原子名称完整保留，包括可能的特殊字符
                            atom_name = parts[1]
                            atom_type = parts[5]
                            
                            # 添加子结构编号信息
                            substruct_num = parts[6]  # 子结构编号
                            substruct_name = parts[7]  # 子结构名称
                            
                            # 尝试解析数值字段
                            try:
                                atom_id = int(parts[0])
                                x = float(parts[2])
                                y = float(parts[3])
                                z = float(parts[4])
                                charge = float(parts[8])
                                
                                atom_info = {
                                    'atom_id': atom_id,
                                    'atom_name': atom_name,
                                    'atom_type': atom_type,
                                    'substruct_num': substruct_num,  # 新增字段
                                    'substruct_name': substruct_name,
                                    'x': x,
                                    'y': y,
                                    'z': z,
                                    'charge': charge
                                }
                                atoms.append(atom_info)
                            except (ValueError, IndexError) as e:
                                logger.warning(f"解析原子数值字段出错 '{line}': {str(e)}")
                        else:
                            logger.warning(f"原子行格式不正确 '{line}'")
                    except Exception as e:
                        logger.warning(f"解析原子行出错 '{line}': {str(e)}")
                
                # 键的解析
                elif current_section == 'BOND':
                    parts = line.split()
                    if len(parts) >= 4:
                        try:
                            bond_info = {
                                'atom1_id': int(parts[1]),
                                'atom2_id': int(parts[2]),
                                'bond_type': parts[3].lower()
                            }
                            bonds.append(bond_info)
                        except (ValueError, IndexError) as e:
                            logger.warning(f"解析键行出错 '{line}': {str(e)}")
    except Exception as e:
        logger.error(f"解析文件 {file_path} 时出错: {str(e)}")
        return [], []
    
    return atoms, bonds

#### 这个地方可能存在问题
# 目前是采用的将G6等数字加字符组合进行分割，因为子结构信息中的数字其实在子结构序号中已经存在了，没有必要再进行分割运算
def split_substruct_name(name, num=None):
    """
    分割子结构名称为基础名称和编号
    
    Args:
        name: 子结构名称，如 'G6'
        num: 子结构编号，如 '6'
    
    Returns:
        包含基础名称和编号的列表
    """
    # 如果提供了编号，直接使用
    if num is not None:
        # 提取名称中的字母部分
        base_name = re.match(r'^([A-Za-z]+)', name)
        if base_name:
            return [base_name.group(1), num]
    
    # 否则使用原来的方法
    match = re.match(r'^([A-Za-z]+)(\d+)$', name)
    return [match.group(1), match.group(2)] if match else [name]

def validate_graph_data(atoms, bonds):
    """验证图数据的有效性"""
    if not atoms:
        return False, "没有原子数据"
    
    if not bonds:
        logger.warning("没有键数据，可能是单原子分子")
    
    # 检查原子ID是否连续
    atom_ids = set(atom['atom_id'] for atom in atoms)
    if len(atom_ids) != len(atoms):
        return False, "原子ID不连续或有重复"
    
    # 检查键引用的原子是否存在
    for bond in bonds:
        if bond['atom1_id'] not in atom_ids or bond['atom2_id'] not in atom_ids:
            return False, f"键引用了不存在的原子ID: {bond['atom1_id']} 或 {bond['atom2_id']}"
    
    return True, ""

def validate_model(ft_model, test_words=None):
    """验证FastText模型的正确性"""
    if test_words is None:
        # 默认测试词汇
        test_words = ['C5', 'N', 'O.3', 'H', 'P', 'S', '1', '2', 'ar', 'am']
    
    logger.info("验证FastText模型...")
    
    # 检查词汇表大小
    vocab_size = len(ft_model.wv.index_to_key)
    logger.info(f"词汇表大小: {vocab_size}")
    
    # 检查向量维度
    vector_size = ft_model.wv.vector_size
    logger.info(f"向量维度: {vector_size}")
    
    # 检查测试词汇是否在词汇表中
    missing_words = [word for word in test_words if word not in ft_model.wv]
    if missing_words:
        logger.warning(f"以下词汇不在模型中: {missing_words}")
        test_words = [word for word in test_words if word in ft_model.wv]
    
    # 获取测试词汇的向量
    test_results = []
    for word in test_words:
        if word in ft_model.wv:
            vector = ft_model.wv[word]
            # 找到最相似的词
            similar_words = ft_model.wv.most_similar(word, topn=3)
            test_results.append({
                'word': word,
                'vector_norm': np.linalg.norm(vector),
                'similar_words': similar_words
            })
    
    # 输出测试结果
    for result in test_results:
        logger.info(f"词汇: {result['word']}")
        logger.info(f"  向量范数: {result['vector_norm']:.4f}")
        logger.info(f"  相似词: {result['similar_words']}")
    
    # 测试词汇相似度
    if len(test_words) >= 2:
        for i in range(len(test_words)):
            for j in range(i+1, len(test_words)):
                word1, word2 = test_words[i], test_words[j]
                if word1 in ft_model.wv and word2 in ft_model.wv:
                    similarity = ft_model.wv.similarity(word1, word2)
                    logger.info(f"相似度 {word1} - {word2}: {similarity:.4f}")
    
    return test_results

# ------------------------------
# 核心流程
# ------------------------------
def process_folder(input_dir, output_dir, vector_size=128, resume=False, validate=True):
    """处理文件夹中的所有mol2文件，生成图数据"""
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置随机种子以确保结果可复现
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)
    
    # 检查是否继续之前的处理
    ft_model_path = os.path.join(output_dir, "global_fasttext.model")
    scaler_path = os.path.join(output_dir, "global_scaler.joblib")
    
    if resume and os.path.exists(ft_model_path) and os.path.exists(scaler_path):
        logger.info("加载已有模型和标准化器...")
        ft_model = FastText.load(ft_model_path)
        scaler = load(scaler_path)
        
        # 验证加载的模型
        if validate:
            validate_model(ft_model)
    else:
        # ========== 第一次遍历：收集全局数据 ==========
        logger.info("开始第一次遍历：收集全局数据...")
        all_sentences = []
        all_numeric = []  # 收集所有原子坐标和电荷
        mol2_files = []
        
        # 收集所有mol2文件
        for fname in os.listdir(input_dir):
            if fname.endswith(".mol2"):
                file_path = os.path.join(input_dir, fname)
                mol2_files.append(file_path)
        
        if not mol2_files:
            logger.error(f"在 {input_dir} 中没有找到mol2文件")
            return
        
        logger.info(f"找到 {len(mol2_files)} 个mol2文件")
        
        # 处理每个文件
        for file_path in tqdm(mol2_files, desc="收集数据"):
            atoms, _ = parse_mol2(file_path)  # 只收集原子信息
            if not atoms:
                logger.warning(f"文件 {file_path} 中没有找到原子数据，跳过")
                continue
            
            all_numeric.extend([[a['x'], a['y'], a['z'], a['charge']] for a in atoms])
    
            for atom in atoms:
                # 使用改进的 split_substruct_name 函数，传入 substruct_num
                substruct_parts = split_substruct_name(atom['substruct_name'], atom['substruct_num'])
                all_sentences.append([atom['atom_name'], atom['atom_type']] + substruct_parts)
        
        if not all_sentences or not all_numeric:
            logger.error("没有收集到足够的数据进行训练")
            return
        
        # ========== 训练全局模型 ==========
        logger.info("训练全局FastText模型...")
        # 优化FastText参数
        ft_model = FastText(
            sentences=all_sentences,
            vector_size=vector_size,
            window=3,
            min_count=1,
            workers=8,  # 增加worker数量以利用多核CPU
            min_n=1,
            max_n=7,
            bucket=2000000,
            sg=1,
            epochs=50,
            seed=42  # 添加随机种子以确保结果可复现
        )
        
        # 验证训练的模型
        if validate:
            validate_model(ft_model)
        
        logger.info("训练全局标准化器...")
        # 转换为numpy数组以提高效率
        all_numeric_array = np.array(all_numeric, dtype=np.float32)
        scaler = StandardScaler()
        scaler.fit(all_numeric_array)
        
        # 保存全局模型和标准化器
        ft_model.save(ft_model_path)
        dump(scaler, scaler_path)
        logger.info(f"模型和标准化器已保存至: {output_dir}")
    
    # ========== 第二次遍历：生成图数据 ==========
    logger.info("开始第二次遍历：生成图数据...")
    
    # 重新收集mol2文件（以防resume模式）
    mol2_files = []
    for fname in os.listdir(input_dir):
        if fname.endswith(".mol2"):
            file_path = os.path.join(input_dir, fname)
            mol2_files.append(file_path)
    
    # 检查已处理的文件
    processed_files = set()
    for fname in os.listdir(output_dir):
        if fname.endswith("_graph.pt"):
            base_name = fname[:-9]  # 移除 "_graph.pt"
            processed_files.add(base_name)
    
    skipped = 0
    processed = 0
    errors = 0
    
    # 预先加载常用键类型的向量以避免重复查询
    common_bond_types = ['1', '2', '3', 'ar', 'am', 'du', 'un', 'nc']
    bond_type_vectors = {}
    for bond_type in common_bond_types:
        try:
            bond_type_vectors[bond_type] = ft_model.wv[bond_type]
        except KeyError:
            pass
    
    for file_path in tqdm(mol2_files, desc="生成图数据"):
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        save_path = os.path.join(output_dir, f"{base_name}_graph.pt")
        
        # 如果已处理且resume模式，则跳过
        if resume and base_name in processed_files:
            skipped += 1
            continue
        
        try:
            atoms, bonds = parse_mol2(file_path)
            
            # 验证数据
            is_valid, error_msg = validate_graph_data(atoms, bonds)
            if not is_valid:
                logger.warning(f"文件 {file_path} 数据无效: {error_msg}")
                errors += 1
                continue
            
            # 生成原子特征
            atom_vectors = []
            for atom in atoms:
                try:
                    vec_name = ft_model.wv[atom['atom_name']]
                    vec_type = ft_model.wv[atom['atom_type']]
        
                    # 使用改进的 split_substruct_name 函数
                    substruct_parts = split_substruct_name(atom['substruct_name'], atom['substruct_num'])
                    vec_substruct = [ft_model.wv[p] for p in substruct_parts]
                    if len(vec_substruct) == 1:
                        vec_substruct.append(np.zeros(vector_size))
                    combined_vec = np.concatenate([vec_name, vec_type, vec_substruct[0], vec_substruct[1]])
                    atom_vectors.append(combined_vec)
                except KeyError as e:
                    logger.warning(f"在文件 {file_path} 中找不到向量: {str(e)}")
                    raise
            
            # 标准化数值特征并分离坐标
            numeric_features = scaler.transform([[a['x'], a['y'], a['z'], a['charge']] for a in atoms])
            scaled_coords = numeric_features[:, :3]  # 标准化后的坐标
            scaled_charge = numeric_features[:, 3:]  # 标准化后的电荷
            
            # 合并特征 - 使用numpy操作提高效率
            atom_features = np.concatenate([atom_vectors, scaled_charge], axis=1)
            atom_features = torch.tensor(atom_features, dtype=torch.float32)
            
            # 构建图数据
            if bonds:
                # 优化：先创建numpy数组，再转换为tensor
                edge_index_array = np.array([
                    [b['atom1_id']-1 for b in bonds],
                    [b['atom2_id']-1 for b in bonds]
                ], dtype=np.int64)
                edge_index = torch.tensor(edge_index_array, dtype=torch.long)
                
                # 优化：使用预先缓存的键类型向量
                bond_vectors = []
                for bond in bonds:
                    bond_type = bond['bond_type']
                    if bond_type in bond_type_vectors:
                        bond_vectors.append(bond_type_vectors[bond_type])
                    else:
                        bond_vectors.append(ft_model.wv[bond_type])
                        # 缓存新遇到的键类型
                        bond_type_vectors[bond_type] = ft_model.wv[bond_type]
                
                # 优化：先转换为numpy数组，再创建tensor
                bond_features = torch.tensor(np.array(bond_vectors, dtype=np.float32), dtype=torch.float32)
            else:
                # 处理没有键的情况
                edge_index = torch.zeros((2, 0), dtype=torch.long)
                bond_features = torch.zeros((0, vector_size), dtype=torch.float32)
            
            graph_data = Data(
                x=atom_features,
                edge_index=edge_index,
                edge_attr=bond_features,
                y=torch.tensor(scaled_coords, dtype=torch.float32),  # 坐标作为目标
                num_nodes=len(atoms)
            )
            
            # 保存文件
            torch.save(graph_data, save_path)
            processed += 1
            
        except Exception as e:
            logger.error(f"处理文件 {file_path} 时出错: {str(e)}")
            errors += 1
    
    # 报告处理结果
    elapsed_time = time.time() - start_time
    logger.info(f"处理完成！耗时: {elapsed_time:.2f}秒")
    logger.info(f"总文件数: {len(mol2_files)}, 处理成功: {processed}, 跳过: {skipped}, 错误: {errors}")
    logger.info(f"输出目录: {output_dir}")
    
    # 返回处理统计信息
    return {
        "total": len(mol2_files),
        "processed": processed,
        "skipped": skipped,
        "errors": errors,
        "elapsed_time": elapsed_time,
        "ft_model": ft_model,
        "scaler": scaler
    }

def test_vector_reconstruction(ft_model, test_file_path=None, num_samples=50):
    """测试向量到词的还原正确性"""
    logger.info("测试向量到词的还原...")
    
    # 如果提供了测试文件，从文件中获取测试样本
    if test_file_path and os.path.exists(test_file_path):
        atoms, bonds = parse_mol2(test_file_path)
        if atoms:
            # 从原子中随机选择样本
            if len(atoms) > num_samples:
                test_atoms = random.sample(atoms, num_samples)
            else:
                test_atoms = atoms
                
            for atom in test_atoms:
                # 获取原子名称和类型
                atom_name = atom['atom_name']
                atom_type = atom['atom_type']
                
                # 获取向量
                if atom_name in ft_model.wv and atom_type in ft_model.wv:
                    vec_name = ft_model.wv[atom_name]
                    vec_type = ft_model.wv[atom_type]
                    
                    # 找到最相似的词
                    similar_to_name = ft_model.wv.most_similar([vec_name], topn=1)
                    similar_to_type = ft_model.wv.most_similar([vec_type], topn=1)
                    
                    logger.info(f"原子: {atom_name} (类型: {atom_type})")
                    logger.info(f"  向量还原 - 名称: {similar_to_name[0][0]} (相似度: {similar_to_name[0][1]:.4f})")
                    logger.info(f"  向量还原 - 类型: {similar_to_type[0][0]} (相似度: {similar_to_type[0][1]:.4f})")
                    
                    # 验证还原是否正确
                    name_correct = similar_to_name[0][0] == atom_name
                    type_correct = similar_to_type[0][0] == atom_type
                    logger.info(f"  名称还原正确: {name_correct}, 类型还原正确: {type_correct}")
    
    # 从模型词汇表中随机选择样本
    vocab = ft_model.wv.index_to_key
    if len(vocab) > num_samples:
        test_words = random.sample(vocab, num_samples)
    else:
        test_words = vocab
    
    logger.info("\n随机词汇测试:")
    for word in test_words:
        # 获取向量
        vector = ft_model.wv[word]
        
        # 找到最相似的词
        similar = ft_model.wv.most_similar([vector], topn=3)
        
        logger.info(f"词汇: {word}")
        logger.info(f"  向量还原 - 最相似: {similar[0][0]} (相似度: {similar[0][1]:.4f})")
        logger.info(f"  其他相似词: {similar[1:3]}")
        
        # 验证还原是否正确
        correct = similar[0][0] == word
        logger.info(f"  还原正确: {correct}")

if __name__ == "__main__":
    # 配置路径
    input_dir = r"E:/APTAMER-GEN/mol2"
    output_dir = r"E:/APTAMER-GEN/models/vecgen-model/all"
    
    # 执行预处理
    stats = process_folder(
        input_dir=input_dir, 
        output_dir=output_dir, 
        vector_size=128,
        resume=True,  # 设置为True可以继续之前的处理
        validate=True  # 验证模型
    )
    
    if stats:
        print("\n处理统计:")
        print(f"总文件数: {stats['total']}")
        print(f"成功处理: {stats['processed']}")
        print(f"已跳过: {stats['skipped']}")
        print(f"处理错误: {stats['errors']}")
        print(f"总耗时: {stats['elapsed_time']:.2f}秒")
        
        # 测试向量还原
        if 'ft_model' in stats:
            # 随机选择一个mol2文件进行测试
            mol2_files = [f for f in os.listdir(input_dir) if f.endswith('.mol2')]
            if mol2_files:
                test_file = os.path.join(input_dir, random.choice(mol2_files))
                test_vector_reconstruction(stats['ft_model'], test_file)
            else:
                test_vector_reconstruction(stats['ft_model'])