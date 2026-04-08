"""
Predictor类 - 评测提交预测器（优化版V2）

优化点：
1. 支持新模型架构（TCN+Transformer / LSTM）
2. 自动检测模型类型
3. 更好的特征处理
"""

from __future__ import annotations

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from typing import List

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from model import HFTPredictor, HFTPredictorLite


class Predictor:
    """
    高频量化价格预测器（优化版）
    """
    
    def __init__(self) -> None:
        current_dir = os.path.dirname(__file__)
        
        config_path = os.path.join(current_dir, "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        model_path = os.path.join(current_dir, "best_model.pt")
        
        self._load_model(model_path)
        
        self.feature_cols = self.config.get("feature", [])
        
        print(f"模型已加载到设备: {self.device}")
    
    def _load_model(self, model_path: str) -> None:
        try:
            checkpoint = torch.load(
                model_path, 
                map_location=self.device, 
                weights_only=False
            )
        except TypeError:
            checkpoint = torch.load(model_path, map_location=self.device)
        
        if "model_state" in checkpoint:
            self.meta = dict(checkpoint.get("meta") or {})
            num_classes = self.meta.get("num_classes_per_head", [3, 3, 3, 3, 3])
            model_type = self.meta.get("model_type", "lite")
            num_features = self.meta.get("num_features", 154)
            
            if model_type == "full":
                print("加载模型: HFTPredictor (TCN+Transformer)")
                self.model = HFTPredictor(
                    num_features=num_features,
                    num_classes_per_head=num_classes
                )
            else:
                print("加载模型: HFTPredictorLite (LSTM)")
                self.model = HFTPredictorLite(
                    num_features=num_features,
                    num_classes_per_head=num_classes
                )
            
            self.model.load_state_dict(checkpoint["model_state"], strict=False)
        else:
            state_keys = list(checkpoint.keys())
            num_classes = [3, 3, 3, 3, 3]
            
            if any("tcn" in k.lower() or "transformer" in k.lower() for k in state_keys):
                print("自动检测: HFTPredictor (TCN+Transformer)")
                self.model = HFTPredictor(num_features=154, num_classes_per_head=num_classes)
            else:
                print("自动检测: HFTPredictorLite (LSTM)")
                self.model = HFTPredictorLite(num_features=154, num_classes_per_head=num_classes)
            
            self.model.load_state_dict(checkpoint, strict=False)
            self.meta = {}
        
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self, x: List[pd.DataFrame]) -> List[List[int]]:
        arrs = []
        for df in x:
            if self.feature_cols and len(self.feature_cols) > 0:
                try:
                    available_cols = [c for c in self.feature_cols if c in df.columns]
                    if len(available_cols) == len(self.feature_cols):
                        df_subset = df[self.feature_cols]
                    else:
                        for col in self.feature_cols:
                            if col not in df.columns:
                                df[col] = 0.0
                        df_subset = df[self.feature_cols]
                except (KeyError, TypeError):
                    df_subset = df
            else:
                df_subset = df
            
            arr = df_subset.to_numpy(dtype=np.float32, copy=False)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            arrs.append(arr)
        
        x_np = np.ascontiguousarray(np.stack(arrs, axis=0))
        
        x_tensor = torch.from_numpy(x_np).to(self.device, dtype=torch.float32)
        
        with torch.no_grad():
            outputs = self.model(x_tensor)
        
        preds_per_head = [output.argmax(dim=1).cpu().numpy() for output in outputs]
        pred_matrix = np.stack(preds_per_head, axis=1)
        
        return pred_matrix.astype(int).tolist()


if __name__ == "__main__":
    predictor = Predictor()
    
    np.random.seed(42)
    test_data = [pd.DataFrame(np.random.randn(100, 154)) for _ in range(4)]
    
    results = predictor.predict(test_data)
    
    print(f"输入批次数: {len(test_data)}")
    print(f"每个批次的时间步数: {test_data[0].shape[0]}")
    print(f"每个批次的特征数: {test_data[0].shape[1]}")
    print(f"预测结果形状: {len(results)} x {len(results[0])}")
    print(f"预测结果示例:")
    for i, result in enumerate(results[:5]):
        labels = ["label_5", "label_10", "label_20", "label_40", "label_60"]
        directions = ["下跌", "不变", "上涨"]
        print(f"  样本{i+1}: {dict(zip(labels, [directions[r] for r in result]))}")
