import os
import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, GCNConv
from sklearn.model_selection import train_test_split
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import logging
import time
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import random
from torch_geometric.transforms import RandomRotate, Compose, RandomTranslate
from joblib import load
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.utils import to_dense_batch

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("train_process.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"使用设备: {device}")

# 设置CUDA内存分配器
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.8)  # 限制GPU内存使用
    torch.backends.cudnn.benchmark = True  # 优化性能

# 设置随机种子
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed()

# 改进的图注意力网络模型
class EnhancedGAT(torch.nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim=512, dropout=0.2):
        super().__init__()
        # 特征投影层
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        
        # 多头图注意力层
        self.conv1 = GATv2Conv(hidden_dim, hidden_dim//2, 
                              edge_dim=edge_dim, heads=8,
                              dropout=dropout, concat=True)
        self.conv2 = GATv2Conv(hidden_dim*4, hidden_dim, 
                              edge_dim=edge_dim, heads=4,
                              dropout=dropout*0.8, concat=False)
        self.conv3 = GATv2Conv(hidden_dim, hidden_dim,
                              edge_dim=edge_dim, heads=1,
                              dropout=dropout*0.6)
        
        # 残差连接
        self.res1 = nn.Linear(hidden_dim, hidden_dim*4)
        self.res2 = nn.Linear(hidden_dim*4, hidden_dim)
        
        # 批归一化层
        self.bn1 = nn.BatchNorm1d(hidden_dim*4)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        
        # 坐标预测头 - 使用多层感知机
        self.coord_pred = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout*0.5),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ELU(),
            nn.Dropout(dropout*0.3),
            nn.Linear(hidden_dim//2, 3)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重以提高训练稳定性"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # 特征投影
        x = self.node_proj(x)
        
        # 第一层图注意力 + 残差
        x1 = self.conv1(x, edge_index, edge_attr)
        res = self.res1(x)
        x = x1 + res
        x = F.leaky_relu(self.bn1(x), negative_slope=0.1)
        
        # 第二层图注意力 + 残差
        x2 = self.conv2(x, edge_index, edge_attr)
        res = self.res2(x)
        x = x2 + res
        x = F.elu(self.bn2(x))
        
        # 第三层图注意力
        x = self.conv3(x, edge_index, edge_attr)
        x = F.elu(self.bn3(x))
        
        # 预测坐标
        coords = self.coord_pred(x)
        
        return coords

# 混合模型：结合GCN和GAT
class HybridGNN(torch.nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim=512, dropout=0.2):
        super().__init__()
        # 特征投影层
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        
        # GCN层
        self.gcn1 = GCNConv(hidden_dim, hidden_dim)
        
        # GAT层
        self.gat1 = GATv2Conv(hidden_dim, hidden_dim//2, 
                             edge_dim=edge_dim, heads=8,
                             dropout=dropout, concat=True)
        self.gat2 = GATv2Conv(hidden_dim*4, hidden_dim, 
                             edge_dim=edge_dim, heads=1,
                             dropout=dropout*0.5)
        
        # 批归一化层
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim*4)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        
        # 坐标预测头
        self.coord_pred = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout*0.5),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ELU(),
            nn.Linear(hidden_dim//2, 3)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # 特征投影
        x = self.node_proj(x)
        
        # GCN层
        x1 = self.gcn1(x, edge_index)
        x1 = F.relu(self.bn1(x1))
        
        # GAT层
        x2 = self.gat1(x1, edge_index, edge_attr)
        x2 = F.leaky_relu(self.bn2(x2), negative_slope=0.1)
        
        x3 = self.gat2(x2, edge_index, edge_attr)
        x3 = F.elu(self.bn3(x3))
        
        # 预测坐标
        coords = self.coord_pred(x3)
        
        return coords

def geometric_loss(pred_coords, edge_index, true_coords, alpha=0.7, beta=0.3):
    """几何约束损失"""
    src, dst = edge_index
    
    # 坐标损失
    coord_loss = F.mse_loss(pred_coords, true_coords)
    
    # 键长约束
    pred_dist = torch.norm(pred_coords[src] - pred_coords[dst], dim=1)
    true_dist = torch.norm(true_coords[src] - true_coords[dst], dim=1)
    bond_loss = F.mse_loss(pred_dist, true_dist)
    
    # 角度约束 (可选，需要三个相连的原子)
    # 这里简化处理，仅使用坐标和键长约束
    
    return alpha * coord_loss + beta * bond_loss

def calculate_metrics(pred, true, edge_index=None):
    """计算评估指标"""
    # 均方根误差
    rmse = torch.sqrt(F.mse_loss(pred, true))
    # 平均绝对误差
    mae = F.l1_loss(pred, true)
    
    metrics = {
        'rmse': rmse.item(),
        'mae': mae.item()
    }
    
    # 如果提供了边索引，计算键长误差
    if edge_index is not None:
        src, dst = edge_index
        pred_dist = torch.norm(pred[src] - pred[dst], dim=1)
        true_dist = torch.norm(true[src] - true[dst], dim=1)
        bond_rmse = torch.sqrt(F.mse_loss(pred_dist, true_dist))
        metrics['bond_rmse'] = bond_rmse.item()
    
    return metrics

# 改进的数据加载函数
def load_dataset(data_dir, scaler_path=None, max_samples=None):
    """加载数据集并应用标准化器"""
    dataset = []
    file_count = 0
    
    # 获取所有图数据文件
    graph_files = [f for f in os.listdir(data_dir) if f.endswith("_graph.pt")]
    
    # 如果设置了最大样本数，随机选择文件
    if max_samples and max_samples < len(graph_files):
        graph_files = random.sample(graph_files, max_samples)
    
    for fname in tqdm(graph_files, desc="加载数据集"):
        try:
            data = torch.load(os.path.join(data_dir, fname), weights_only=False)
            # 确保数据有效
            if data.x.shape[0] > 0 and data.edge_index.shape[1] > 0:
                dataset.append(data)
                file_count += 1
            else:
                logger.warning(f"跳过空数据: {fname}")
        except Exception as e:
            logger.error(f"加载 {fname} 时出错: {str(e)}")
    
    logger.info(f"加载了 {len(dataset)} 个有效图数据")
    
    # 如果提供了标准化器路径，加载并应用
    if scaler_path and os.path.exists(scaler_path):
        try:
            scaler = load(scaler_path)
            logger.info(f"加载标准化器: {scaler_path}")
        except Exception as e:
            logger.error(f"加载标准化器时出错: {str(e)}")
    
    return dataset

# 增强的数据增强函数
def apply_transforms(dataset, transform_prob=0.5):
    """应用数据增强变换"""
    # 组合多种变换
    transforms = Compose([
        RandomRotate(degrees=180, axis=0),  # 随机旋转
        RandomTranslate(0.01)  # 随机平移
    ])
    
    augmented_dataset = []
    for data in tqdm(dataset, desc="应用数据增强"):
        # 添加原始数据
        augmented_dataset.append(data)
        
        # 有一定概率应用变换
        if random.random() < transform_prob:
            try:
                # 创建数据的深拷贝以避免修改原始数据
                transformed_data = data.clone()
                transformed_data = transforms(transformed_data)
                augmented_dataset.append(transformed_data)
            except Exception as e:
                logger.warning(f"应用变换时出错: {str(e)}")
    
    logger.info(f"数据增强后的数据集大小: {len(augmented_dataset)}")
    return augmented_dataset

def plot_training_history(history, save_path):
    """绘制训练历史并保存"""
    plt.figure(figsize=(12, 8))
    
    # 绘制损失
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='训练损失')
    plt.plot(history['val_loss'], label='验证损失')
    plt.title('损失曲线')
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.legend()
    
    # 绘制RMSE
    plt.subplot(2, 2, 2)
    plt.plot(history['train_rmse'], label='训练RMSE')
    plt.plot(history['val_rmse'], label='验证RMSE')
    plt.title('RMSE曲线')
    plt.xlabel('轮次')
    plt.ylabel('RMSE')
    plt.legend()
    
    # 绘制MAE
    plt.subplot(2, 2, 3)
    plt.plot(history['train_mae'], label='训练MAE')
    plt.plot(history['val_mae'], label='验证MAE')
    plt.title('MAE曲线')
    plt.xlabel('轮次')
    plt.ylabel('MAE')
    plt.legend()
    
    # 绘制学习率
    plt.subplot(2, 2, 4)
    plt.plot(history['lr'])
    plt.title('学习率')
    plt.xlabel('轮次')
    plt.ylabel('学习率')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"训练历史图表已保存至: {save_path}")

# 添加可视化预测结果的函数
def visualize_predictions(model, data_loader, num_samples=3, save_dir=None):
    """可视化模型预测结果"""
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    samples_visualized = 0
    
    with torch.no_grad():
        for batch in data_loader:
            if samples_visualized >= num_samples:
                break
                
            batch = batch.to(device)
            pred_coords = model(batch)
            
            # 获取真实坐标和预测坐标
            true_coords = batch.y.cpu().numpy()
            pred_coords = pred_coords.cpu().numpy()
            
            # 为每个图创建一个可视化
            batch_size = 1  # 假设每个批次只有一个图
            for i in range(batch_size):
                if samples_visualized >= num_samples:
                    break
                    
                # 创建3D图
                fig = plt.figure(figsize=(12, 6))
                
                # 真实坐标
                ax1 = fig.add_subplot(121, projection='3d')
                ax1.scatter(true_coords[:, 0], true_coords[:, 1], true_coords[:, 2], c='blue', marker='o')
                ax1.set_title('真实坐标')
                ax1.set_xlabel('X')
                ax1.set_ylabel('Y')
                ax1.set_zlabel('Z')
                
                # 预测坐标
                ax2 = fig.add_subplot(122, projection='3d')
                ax2.scatter(pred_coords[:, 0], pred_coords[:, 1], pred_coords[:, 2], c='red', marker='^')
                ax2.set_title('预测坐标')
                ax2.set_xlabel('X')
                ax2.set_ylabel('Y')
                ax2.set_zlabel('Z')
                
                plt.tight_layout()
                
                if save_dir:
                    plt.savefig(os.path.join(save_dir, f"prediction_{samples_visualized}.png"))
                    plt.close()
                else:
                    plt.show()
                
                samples_visualized += 1
# ... (保持现有的导入语句和类定义不变，直到 visualize_predictions 函数) ...

def log_memory_usage():
    """记录当前内存使用情况"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        cached = torch.cuda.memory_reserved() / 1024**2
        logger.info(f"GPU 内存使用: 已分配 {allocated:.1f}MB, 缓存 {cached:.1f}MB")

def main():
    """主训练函数"""
    config = {
        "data_dir": r"E:\APTAMER-GEN\models\vecgen-model\all",
        "output_dir": r"E:\Python\DL\SEQ2VEC\outputs",
        "batch_size": 16,  # 减小批次大小
        "accumulation_steps": 2,  # 添加梯度累积步数
        "lr": 2e-4,
        "weight_decay": 2e-5,
        "epochs": 50,
        "val_ratio": 0.15,
        "patience": 20,
        "min_lr": 5e-7,
        "use_mixed_precision": True,
        "scheduler_type": "cosine",
        "use_data_augmentation": True,
        "transform_prob": 0.3,
        "resume_training": False,
        "checkpoint_path": None,
        "model_type": "hybrid",
        "max_samples": 500,  # 限制最大样本数用于测试
        "visualize_results": True,
        "num_visualizations": 5
    }
    
    try:
        # 创建输出目录
        os.makedirs(config["output_dir"], exist_ok=True)
        
        # 保存配置
        config_path = os.path.join(config["output_dir"], "config.txt")
        with open(config_path, 'w') as f:
            for k, v in config.items():
                f.write(f"{k}: {v}\n")
        
        # 加载数据集
        logger.info("开始加载数据集...")
        scaler_path = os.path.join(config["data_dir"], "global_scaler.joblib")
        dataset = load_dataset(config["data_dir"], scaler_path, config["max_samples"])
        
        if len(dataset) == 0:
            raise ValueError("没有加载到有效数据")
        
        # 数据增强
        if config["use_data_augmentation"]:
            dataset = apply_transforms(dataset, config["transform_prob"])
        
        # 划分训练验证集
        train_idx, val_idx = train_test_split(
            range(len(dataset)), 
            test_size=config["val_ratio"],
            random_state=42
        )
        train_dataset = [dataset[i] for i in train_idx]
        val_dataset = [dataset[i] for i in val_idx]
        
        logger.info(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}")

        # 数据加载器
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config["batch_size"], 
            shuffle=True,
            pin_memory=True,
            num_workers=4
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=config["batch_size"]*2,
            pin_memory=True,
            num_workers=2
        )

        # 初始化模型
        node_dim = 513
        edge_dim = 128
        
        if config["model_type"] == "hybrid":
            model = HybridGNN(node_dim=node_dim, edge_dim=edge_dim).to(device)
            logger.info("使用混合GNN模型 (GCN+GAT)")
        else:
            model = EnhancedGAT(node_dim=node_dim, edge_dim=edge_dim).to(device)
            logger.info("使用增强GAT模型")
        
        # 打印模型参数数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"模型总参数: {total_params:,}, 可训练参数: {trainable_params:,}")
        
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=config["lr"], 
            weight_decay=config["weight_decay"]
        )
        scaler = GradScaler()
        
        # 选择学习率调度器
        if config["scheduler_type"] == "plateau":
            scheduler = ReduceLROnPlateau(
                optimizer, 
                mode='min',
                factor=0.5,
                patience=7,
                min_lr=config["min_lr"]
            )
        else:  # cosine
            scheduler = CosineAnnealingLR(
                optimizer, 
                T_max=config["epochs"], 
                eta_min=config["min_lr"]
            )

        # 初始化训练状态
        start_epoch = 0
        best_val_loss = float('inf')
        best_val_rmse = float('inf')
        no_improve = 0
        history = {
            'train_loss': [], 'val_loss': [],
            'train_rmse': [], 'val_rmse': [],
            'train_mae': [], 'val_mae': [],
            'lr': []
        }
        
        # 训练循环
        for epoch in range(start_epoch, config["epochs"]):
            try:
                epoch_start = time.time()
                
                # 训练阶段
                model.train()
                train_loss = 0
                train_metrics = {'rmse': 0, 'mae': 0, 'bond_rmse': 0}
                optimizer.zero_grad()  # 在循环外清零梯度
                
                train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [Train]")
                for batch_idx, batch in enumerate(train_pbar):
                    try:
                        batch = batch.to(device)
                        
                        if config["use_mixed_precision"]:
                            with torch.amp.autocast(device_type='cuda'):
                                pred = model(batch)
                                loss = geometric_loss(pred, batch.edge_index, batch.y)
                                loss = loss / config["accumulation_steps"]  # 缩放损失
                            
                            scaler.scale(loss).backward()
                            
                            # 梯度累积
                            if (batch_idx + 1) % config["accumulation_steps"] == 0:
                                scaler.step(optimizer)
                                scaler.update()
                                optimizer.zero_grad()
                        else:
                            pred = model(batch)
                            loss = geometric_loss(pred, batch.edge_index, batch.y)
                            loss = loss / config["accumulation_steps"]
                            loss.backward()
                            
                            if (batch_idx + 1) % config["accumulation_steps"] == 0:
                                optimizer.step()
                                optimizer.zero_grad()
                        
                        train_loss += loss.item() * config["accumulation_steps"]
                        batch_metrics = calculate_metrics(pred, batch.y, batch.edge_index)
                        for k, v in batch_metrics.items():
                            if k in train_metrics:
                                train_metrics[k] += v
                        
                        # 更新进度条
                        train_pbar.set_postfix({
                            'loss': f"{loss.item()*config['accumulation_steps']:.3e}",
                            'memory': f"{torch.cuda.memory_allocated()/1024**2:.0f}MB"
                        })
                        
                        # 定期清理缓存
                        if batch_idx % 5 == 0:
                            torch.cuda.empty_cache()
                            
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            torch.cuda.empty_cache()
                            logger.warning(f"批次 {batch_idx} 处理时内存不足，跳过该批次")
                            if hasattr(optimizer, 'zero_grad'):
                                optimizer.zero_grad()
                            continue
                        else:
                            raise e
                
                # 验证阶段
                model.eval()
                val_loss = 0
                val_metrics = {'rmse': 0, 'mae': 0, 'bond_rmse': 0}
                
                with torch.no_grad():
                    val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [Val]")
                    for batch in val_pbar:
                        batch = batch.to(device)
                        pred = model(batch)
                        loss = geometric_loss(pred, batch.edge_index, batch.y)
                        val_loss += loss.item()
                        
                        batch_metrics = calculate_metrics(pred, batch.y, batch.edge_index)
                        for k, v in batch_metrics.items():
                            if k in val_metrics:
                                val_metrics[k] += v
                        
                        val_pbar.set_postfix({'loss': f"{loss.item():.3e}"})
                
                # 计算平均损失和指标
                avg_train_loss = train_loss / len(train_loader)
                avg_val_loss = val_loss / len(val_loader)
                
                for k in train_metrics:
                    train_metrics[k] /= len(train_loader)
                    val_metrics[k] /= len(val_loader)
                
                # 更新学习率
                if config["scheduler_type"] == "plateau":
                    scheduler.step(avg_val_loss)
                else:
                    scheduler.step()
                
                current_lr = optimizer.param_groups[0]['lr']
                
                # 更新历史记录
                history['train_loss'].append(avg_train_loss)
                history['val_loss'].append(avg_val_loss)
                history['train_rmse'].append(train_metrics['rmse'])
                history['val_rmse'].append(val_metrics['rmse'])
                history['train_mae'].append(train_metrics['mae'])
                history['val_mae'].append(val_metrics['mae'])
                history['lr'].append(current_lr)
                
                # 检查模型改进
                improved = False
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    improved = True
                
                if val_metrics['rmse'] < best_val_rmse:
                    best_val_rmse = val_metrics['rmse']
                    improved = True
                    torch.save(model.state_dict(), 
                             os.path.join(config["output_dir"], "best_rmse_model.pt"))
                
                if improved:
                    no_improve = 0
                    torch.save(model.state_dict(), 
                             os.path.join(config["output_dir"], "best_model.pt"))
                else:
                    no_improve += 1
                
                # 保存检查点
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'best_val_rmse': best_val_rmse,
                    'no_improve': no_improve,
                    'history': history
                }, os.path.join(config["output_dir"], "latest_checkpoint.pt"))
                
                # 记录训练信息
                epoch_time = time.time() - epoch_start
                logger.info(
                    f"Epoch {epoch+1:03d}/{config['epochs']} | "
                    f"Time: {epoch_time:.1f}s | "
                    f"Train: {avg_train_loss:.3e} | Val: {avg_val_loss:.3e} | "
                    f"Train RMSE: {train_metrics['rmse']:.3e} | "
                    f"Val RMSE: {val_metrics['rmse']:.3e} | "
                    f"LR: {current_lr:.1e} | "
                    f"NoImprove: {no_improve}/{config['patience']}"
                )
                
                # 绘制训练历史
                if (epoch + 1) % 10 == 0 or epoch == config["epochs"] - 1:
                    plot_training_history(history, 
                        os.path.join(config["output_dir"], 
                                   f"training_history_epoch_{epoch+1}.png"))
                
                # 检查早停
                if no_improve >= config["patience"]:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
                
            except Exception as e:
                logger.error(f"训练过程中出错: {str(e)}", exc_info=True)
                torch.cuda.empty_cache()  # 清理GPU内存
                continue  # 继续下一个epoch而不是直接退出
        
        # 训练结束，进行最终评估
        try:
            total_time = time.time() - start_time
            logger.info(f"训练完成! 总耗时: {total_time:.2f}秒")
            
            # 加载最佳模型
            model.load_state_dict(torch.load(os.path.join(config["output_dir"], "best_model.pt")))
            
            # 最终验证
            model.eval()
            final_metrics = evaluate_model(model, val_loader)
            
            logger.info("最终验证结果:")
            for metric_name, value in final_metrics.items():
                logger.info(f"{metric_name}: {value:.3e}")
            
            # 可视化预测结果
            if config["visualize_results"]:
                visualize_dir = os.path.join(config["output_dir"], "visualizations")
                visualize_loader = DataLoader(
                    random.sample(val_dataset, min(config["num_visualizations"], len(val_dataset))),
                    batch_size=1,
                    shuffle=False
                )
                visualize_predictions(model, visualize_loader, 
                                   num_samples=config["num_visualizations"], 
                                   save_dir=visualize_dir)
        
        except Exception as e:
            logger.error(f"最终评估时出错: {str(e)}", exc_info=True)
            
    except Exception as e:
        logger.error(f"训练过程出错: {str(e)}", exc_info=True)
        raise e

def evaluate_model(model, data_loader):
    """评估模型性能"""
    metrics = {'loss': 0, 'rmse': 0, 'mae': 0, 'bond_rmse': 0}
    count = 0
    
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            pred = model(batch)
            loss = geometric_loss(pred, batch.edge_index, batch.y)
            batch_metrics = calculate_metrics(pred, batch.y, batch.edge_index)
            
            metrics['loss'] += loss.item()
            for k, v in batch_metrics.items():
                metrics[k] += v
            count += 1
    
    # 计算平均值
    for k in metrics:
        metrics[k] /= count
    
    return metrics

if __name__ == "__main__":
    try:
        # 设置CUDA内存分配器
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.8)  # 限制GPU内存使用
            torch.backends.cudnn.benchmark = True  # 优化性能
        
        # 确保输出目录存在
        output_dir = r"E:\Python\DL\SEQ2VEC\outputs"
        os.makedirs(output_dir, exist_ok=True)
        
        # 配置日志文件路径
        log_file = os.path.join(output_dir, "train_process.log")
        
        # 配置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        # 记录初始GPU内存状态
        log_memory_usage()
        
        # 运行主程序
        main()
        
    except Exception as e:
        logger.error(f"程序执行出错: {str(e)}", exc_info=True)
        raise e