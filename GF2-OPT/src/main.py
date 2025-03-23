import torch
from pathlib import Path
from torch.utils.data import DataLoader, IterableDataset
from data_utils.parser import MolParser
from data_utils.tokenizer import LightTokenizer
from models.transformer import MemEfficientTransformer
from utils.logger import TrainingLogger
from utils.memory import HDF5FeatureWriter
from torch.cuda.amp import autocast, GradScaler
import torch.nn as nn
import sys

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent))

class StreamDataset(IterableDataset):
    def __init__(self, tokenizer, parser):
        self.tokenizer = tokenizer
        self.parser = parser
        
    def __iter__(self):
        for block in self.parser.stream_blocks():
            yield self.tokenizer.encode(block)

def collate_fn(batch):
    return torch.stack(batch)

def main():
    # 初始化
    parser = MolParser("E:/APTAMER-GEN/mol2")
    tokenizer = LightTokenizer()
    logger = TrainingLogger()
    
    # 构建词汇表
    tokenizer.build_vocab(parser.stream_blocks())
    print(f"词汇表大小: {len(tokenizer.char2idx)}")
    
    # 模型配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MemEfficientTransformer(len(tokenizer.char2idx)).to(device)
    
    # 优化设置
    optimizer = torch.optim.AdamW(model.parameters(), lr=tokenizer.config['training']['learning_rate'])
    criterion = nn.CrossEntropyLoss(ignore_index=0).to(device)
    scaler = GradScaler()
    accum_steps = tokenizer.config['training']['grad_accum_steps']
    
    # 数据加载
    dataset = StreamDataset(tokenizer, parser)
    dataloader = DataLoader(
        dataset,
        batch_size=tokenizer.config['training']['batch_size'],
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    # 训练循环
    try:
        with HDF5FeatureWriter("features.h5") as writer:
            model.train()
            for epoch in range(tokenizer.config['training']['epochs']):
                total_loss = 0.0
                optimizer.zero_grad()
                
                for step, batch in enumerate(dataloader):
                    batch = batch.to(device, non_blocking=True)
                    
                    with autocast():
                        outputs = model(batch)
                        loss = criterion(outputs.view(-1, len(tokenizer.char2idx)), 
                                    batch.view(-1)) / accum_steps
                    
                    scaler.scale(loss).backward()
                    
                    if (step+1) % accum_steps == 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                    
                    # 小批量存储特征
                    if step % 10 == 0 and step < 10000:  # 仅存储前100步样本
                        with torch.no_grad():
                            features = model.encode(batch[:10])  # 取前8个样本
                            writer.write_batch(features.cpu().numpy())
                    
                    total_loss += loss.item() * accum_steps
                    if step % 20 == 0:
                        mem = torch.cuda.max_memory_allocated()/1024**3
                        print(f"Step {step} | Loss: {loss.item()*accum_steps:.3f} | Mem: {mem:.2f}GB", end='\r')
                
                print(f"Epoch {epoch+1} | Avg Loss: {total_loss/(step+1):.3f}")
                
    except Exception as e:
        logger.log_error(e)
    finally:
        torch.save({
                    'model_state': model.state_dict(),
                    'tokenizer': tokenizer,
                    'config': tokenizer.config
                    }, "full_model.pth")

if __name__ == "__main__":
    main()