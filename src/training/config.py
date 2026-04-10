import yaml
from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class Config:
    data: Dict[str, Any] = field(default_factory=dict)
    model: Dict[str, Any] = field(default_factory=dict)
    training: Dict[str, Any] = field(default_factory=dict)
    loss: Dict[str, Any] = field(default_factory=dict)
    checkpoint: Dict[str, Any] = field(default_factory=dict)
    logging: Dict[str, Any] = field(default_factory=dict)
    segmentation: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(
            data=config_dict.get('data', {}),
            model=config_dict.get('model', {}),
            training=config_dict.get('training', {}),
            loss=config_dict.get('loss', {}),
            checkpoint=config_dict.get('checkpoint', {}),
            logging=config_dict.get('logging', {}),
            segmentation=config_dict.get('segmentation', {}),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'data': self.data,
            'model': self.model,
            'training': self.training,
            'loss': self.loss,
            'checkpoint': self.checkpoint,
            'logging': self.logging,
            'segmentation': self.segmentation,
        }
