import os
import sys
sys.path.append(os.path.dirname(__file__))
from model import TKANEncoder  # 导入T-KAN编码器
import numpy as np
import pandas as pd
from typing import List
import lightgbm as lgb
import torch

class Predictor():
    def __init__(self):
        # 加载T-KAN编码器
        self.encoder = TKANEncoder(input_dim=40, hidden_dim=128, num_layers=3)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        encoder_path = os.path.join(os.path.dirname(__file__), 'tkan_encoder.pth')  # 假设编码器模型文件
        self.encoder.load_state_dict(torch.load(encoder_path, map_location=self.device))
        self.encoder.to(self.device)
        self.encoder.eval()
        
        # 加载LightGBM模型
        lgb_path = os.path.join(os.path.dirname(__file__), 'lightgbm_model.txt')  # 假设模型文件
        self.lgb_model = lgb.Booster(model_file=lgb_path)
        
        # 阈值参数（动态阈值：label_60/40降低阈值，更激进；5/10/20提高阈值，更保守）
        self.confidence_threshold = {
            'label_5': 0.60,   # 高阈值，更保守
            'label_10': 0.60,
            'label_20': 0.60,
            'label_40': 0.45,  # 低阈值，更激进
            'label_60': 0.45
        }
        # 默认阈值
        self.default_threshold = 0.52
        
        # 类别损失权重（加强label_40/60侧重，在训练时使用）
        self.class_weights = {
            0: 1.0,  # 跌
            1: 2.0   # 涨（label_40/60时加强）
        }

    def predict(self, x: List[pd.DataFrame], label_type: str = 'label_60') -> List[List[int]]:
        with torch.no_grad():
            # 预处理数据
            processed_data = [self.preprocess(df) for df in x]
            # 假设每个df有100行（seq_len=100）
            batch_data = []
            for df in processed_data:
                if len(df) < 100:
                    # 填充到100行
                    padding = pd.DataFrame(0, index=range(100 - len(df)), columns=df.columns)
                    df = pd.concat([df, padding], ignore_index=True)
                elif len(df) > 100:
                    df = df.head(100)
                batch_data.append(df.values)
            batch_data = np.stack(batch_data)  # (batch, 100, 40)
            
            # T-KAN编码
            x_tensor = torch.tensor(batch_data, dtype=torch.float32).to(self.device)
            features = self.encoder(x_tensor)  # (batch, 128)
            features_np = features.cpu().numpy()
            
            # LightGBM预测概率（二分类，输出[跌概率, 涨概率]）
            probs = self.lgb_model.predict_proba(features_np)  # (batch, 2)
            
            # 阈值决策：用概率阈值过滤低置信度预测为不变
            signals = []
            for prob in probs:
                prob_fall = prob[0]
                prob_rise = prob[1]
                max_prob = max(prob_fall, prob_rise)
                
                # 动态阈值
                threshold = self.confidence_threshold.get(label_type, self.default_threshold)
                
                if max_prob < threshold:
                    signals.append([1])  # 低置信度，预测不变
                else:
                    if prob_rise > prob_fall:
                        signals.append([2])  # 涨
                    else:
                        signals.append([0])  # 跌
            
            return signals
    
    def preprocess(self, df):
        ''' 数据预处理，输出40特征：bid1-10, ask1-10, bsize1-10, asize1-10 '''
        # 假设df有列：n_bid1到n_bid10, n_ask1到n_ask10, n_bsize1到n_bsize10, n_asize1到n_asize10
        # 如果没有，填充为0或扩展
        bid_cols = [f'n_bid{i}' for i in range(1, 11)]
        ask_cols = [f'n_ask{i}' for i in range(1, 11)]
        bsize_cols = [f'n_bsize{i}' for i in range(1, 11)]
        asize_cols = [f'n_asize{i}' for i in range(1, 11)]
        
        # 确保列存在，如果不足10个，填充0
        for cols in [bid_cols, ask_cols, bsize_cols, asize_cols]:
            for col in cols:
                if col not in df.columns:
                    df[col] = 0.0
        
        # 归一化或其他清洗（根据描述）
        # 这里简化，假设数据已清洗
        
        feature_cols = bid_cols + ask_cols + bsize_cols + asize_cols
        return df[feature_cols]


