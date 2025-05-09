import os
import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import autocast, GradScaler

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 设置随机种子
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed()

# 增强的GAT模型
class EnhancedGAT(torch.nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim=512):
        super().__init__()
        # 第一层：8头注意力
        self.conv1 = GATv2Conv(node_dim, hidden_dim, 
                             edge_dim=edge_dim, heads=8,
                             dropout=0.2, concat=True)
        # 第二层：4头注意力
        self.conv2 = GATv2Conv(hidden_dim*8, hidden_dim,
                             edge_dim=edge_dim, heads=4,
                             dropout=0.1, concat=False)
        # 第三层：特征压缩
        self.conv3 = GATv2Conv(hidden_dim, hidden_dim,
                             edge_dim=edge_dim)
        # 残差连接
        self.res_linear = torch.nn.Linear(node_dim, hidden_dim*8)
        
        # 坐标预测头
        self.coord_pred = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim//2),
            torch.nn.ELU(),
            torch.nn.Linear(hidden_dim//2, 3)
        )
    
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # 残差连接
        x1 = self.conv1(x, edge_index, edge_attr)
        x1 += self.res_linear(x)  # 残差连接
        x = F.leaky_relu(x1, negative_slope=0.2)
        
        # 深层特征提取
        x = F.elu(self.conv2(x, edge_index, edge_attr))
        x = F.elu(self.conv3(x, edge_index, edge_attr))
        
        return self.coord_pred(x)

# 几何约束损失函数
def geometric_loss(pred_coords, edge_index, true_coords, alpha=0.3):
    # 坐标损失
    coord_loss = F.mse_loss(pred_coords, true_coords)
    
    # 键长约束
    src, dst = edge_index
    pred_dist = torch.norm(pred_coords[src] - pred_coords[dst], dim=1)
    true_dist = torch.norm(true_coords[src] - true_coords[dst], dim=1)
    bond_loss = F.mse_loss(pred_dist, true_dist)
    
    return (1 - alpha) * coord_loss + alpha * bond_loss

# 数据增强：分子随机旋转
def random_rotation(coords):
    theta = torch.rand(3) * 2 * np.pi
    # X轴旋转矩阵
    Rx = torch.tensor([
        [1, 0, 0],
        [0, torch.cos(theta[0]), -torch.sin(theta[0])],
        [0, torch.sin(theta[0]), torch.cos(theta[0])]
    ]).to(coords.device)
    # Y轴旋转矩阵
    Ry = torch.tensor([
        [torch.cos(theta[1]), 0, torch.sin(theta[1])],
        [0, 1, 0],
        [-torch.sin(theta[1]), 0, torch.cos(theta[1])]
    ]).to(coords.device)
    # Z轴旋转矩阵
    Rz = torch.tensor([
        [torch.cos(theta[2]), -torch.sin(theta[2]), 0],
        [torch.sin(theta[2]), torch.cos(theta[2]), 0],
        [0, 0, 1]
    ]).to(coords.device)
    return coords @ (Rz @ Ry @ Rx).T

def main():
    config = {
        "data_dir": r"E:/APTAMER-GEN/models/vecgen-model/all",
        "batch_size": 64,        # 增大批处理尺寸
        "lr": 3e-4,              # 调整初始学习率
        "epochs": 200,
        "val_ratio": 0.15,
        "patience": 15,          # 宽松的早停耐心值
        "min_lr": 1e-6,          # 最小学习率
        "amp": True              # 启用混合精度
    }
    
    # 加载数据集
    dataset = []
    for fname in os.listdir(config["data_dir"]):
        if fname.endswith("_graph.pt"):
            data = torch.load(os.path.join(config["data_dir"], fname),
                            weights_only=False)
            dataset.append(data)
    
    # 划分训练验证集
    train_idx, val_idx = train_test_split(range(len(dataset)),
                        test_size=config["val_ratio"], random_state=42)
    train_dataset = [dataset[i] for i in train_idx]
    val_dataset = [dataset[i] for i in val_idx]

    # 数据加载器
    train_loader = DataLoader(train_dataset,
                            batch_size=config["batch_size"],
                            shuffle=True,
                            pin_memory=True,
                            num_workers=4)  # 增加数据加载线程
    val_loader = DataLoader(val_dataset,
                          batch_size=config["batch_size"]*2,
                          pin_memory=True)

    # 初始化模型和优化器
    model = EnhancedGAT(node_dim=513, edge_dim=128).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),
                                lr=config["lr"],
                                weight_decay=1e-5)  # 权重衰减
    scaler = GradScaler(device_type='cuda', enabled=config["amp"])
    scheduler = ReduceLROnPlateau(optimizer,
                                mode='min',
                                factor=0.5,
                                patience=5,
                                min_lr=config["min_lr"])

    best_val_loss = float('inf')
    no_improve = 0

    # 训练循环
    for epoch in range(config["epochs"]):
        model.train()
        train_loss = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            
            # 数据增强：仅对训练集应用随机旋转
            if model.training:
                batch.y = random_rotation(batch.y)
            
            optimizer.zero_grad()
            
            # 混合精度上下文
            with autocast(enabled=config["amp"]):
                pred = model(batch)
                loss = geometric_loss(pred, batch.edge_index, batch.y, alpha=0.3)
            
            # 梯度缩放和反向传播
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
        
        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred = model(batch)
                val_loss += geometric_loss(pred, batch.edge_index, batch.y).item()
        
        # 计算平均损失
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        scheduler.step(avg_val)
        
        # 早停机制
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            no_improve = 0
            torch.save(model.state_dict(),
                      os.path.join(config["data_dir"], "best_model.pt"))
        else:
            no_improve += 1
        
        # 打印训练信息
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:03d}/{config['epochs']} | "
              f"Train: {avg_train:.2e} | Val: {avg_val:.2e} | "
              f"LR: {current_lr:.1e} | NoImp: {no_improve}/{config['patience']}")
        
        if no_improve >= config["patience"]:
            print(f"Early stopping at epoch {epoch+1}")
            break

if __name__ == "__main__":
    main()