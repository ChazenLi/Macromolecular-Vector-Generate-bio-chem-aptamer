import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from pathlib import Path
import yaml

class MemEfficientTransformer(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.config = self._load_config()
        
        self.embedding = nn.Embedding(vocab_size, self.config['model']['d_model'])
        encoder_layer = TransformerEncoderLayer(
            d_model=self.config['model']['d_model'],
            nhead=self.config['model']['nhead'],
            dim_feedforward=self.config['model']['dim_feedforward'],
            dropout=self.config['model']['dropout'],
            batch_first=True
        )
        self.encoder = TransformerEncoder(encoder_layer, self.config['model']['num_layers'])
        self.decoder = nn.Linear(self.config['model']['d_model'], vocab_size)
        
    def _load_config(self):
        config_path = Path(__file__).parents[2] / "configs" / "params.yaml"
        with open(config_path, encoding='utf-8') as f:  # 修复编码
            return yaml.safe_load(f)
    
    def forward(self, src: torch.Tensor) -> torch.Tensor:
        src = self.embedding(src) * (self.config['model']['d_model'] ** 0.5)
        memory = self.encoder(src)
        return self.decoder(memory)
    
    def encode(self, src: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            src_emb = self.embedding(src)
            return self.encoder(src_emb).mean(dim=1)