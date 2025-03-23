from collections import defaultdict
from pathlib import Path
import string
import yaml
import torch

class LightTokenizer:
    def __init__(self):
        self.config = self._load_config()
        self._build_base_vocab()
        
    def _load_config(self):
        config_path = Path(__file__).parents[2] / "configs" / "params.yaml"
        with open(config_path, encoding='utf-8') as f:  # 修复编码
            return yaml.safe_load(f)
    
    def _build_base_vocab(self):
        self.char2idx = {'<pad>':0, '<sos>':1, '<eos>':2, '<unk>':3}
        self.idx2char = {v:k for k,v in self.char2idx.items()}
    
    def build_vocab(self, stream):
        char_counts = defaultdict(int)
        for block in stream:
            for c in block[:self.config['training']['max_seq_len']-2]:
                char_counts[c] += 1
                
        valid_chars = [c for c, cnt in char_counts.items() 
                      if cnt >= self.config['data']['min_char_freq'] or c in string.whitespace]
        
        for idx, c in enumerate(sorted(valid_chars), start=4):
            self.char2idx[c] = idx
            self.idx2char[idx] = c
    
    def encode(self, text: str) -> torch.Tensor:
        max_len = self.config['training']['max_seq_len']
        encoded = [1]  # <sos>
        for c in text.replace('\n', '↨')[:max_len-2]:
            encoded.append(self.char2idx.get(c, 3))  # 3=<unk>
        encoded.append(2)  # <eos>
        
        if len(encoded) < max_len:
            encoded += [0]*(max_len - len(encoded))
        else:
            encoded = encoded[:max_len-1] + [2]
            
        return torch.tensor(encoded, dtype=torch.long)