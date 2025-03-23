from pathlib import Path
from typing import Generator
import yaml

class MolParser:
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.config = self._load_config()
        
    def _load_config(self):
        config_path = Path(__file__).parents[2] / "configs" / "params.yaml"
        with open(config_path, encoding='utf-8') as f:  # 修复编码
            return yaml.safe_load(f)
    
    def stream_blocks(self) -> Generator[str, None, None]:
        for mol_file in self.data_path.glob("**/*.mol2"):
            with open(mol_file, 'r', encoding='utf-8') as f:
                buffer = []
                in_target = False
                for line in f:
                    line = line.strip()
                    if line.startswith(("@<TRIPOS>ATOM", "@<TRIPOS>BOND")):
                        if buffer:
                            yield '\n'.join(buffer)
                        buffer = [line]
                        in_target = True
                    elif in_target:
                        if line.startswith("@<TRIPOS>"):
                            yield '\n'.join(buffer)
                            buffer = []
                            in_target = line.startswith(("@<TRIPOS>ATOM", "@<TRIPOS>BOND"))
                            if in_target:
                                buffer.append(line)
                        elif line:
                            buffer.append(line)
                if buffer:
                    yield '\n'.join(buffer)