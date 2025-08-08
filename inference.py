import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import math
import os
import random

# 과거 주행 기록으로부터 실제 미래 주행을 예측, 예측 평균(30개의 속도 벡터) 및 예측의 불확실성

# ===== 1. 학습 스크립트와 동일한 모델/설정 클래스 정의 =====
# 저장된 모델을 불러오려면, 모델의 '설계도'가 되는 클래스들이 코드에 그대로 있어야 합니다.

class ModelConfig:
    """모델 및 훈련 하이퍼파라미터"""
    DATA_PATH = r'C:\Users\bgh\Desktop\Framework\real\data\VED_final_with_grade.parquet'
    FEATURES = ['speed_mps', 'acceleration_mps2', 'hour', 'day_of_week', 'grade']
    TARGET_FEATURE = 'speed_mps'
    INPUT_SEQ_LEN = 60
    PREDICTION_SEQ_LEN = 30
    D_MODEL = 128
    N_HEAD = 8
    N_ENCODER_LAYERS = 4
    N_DECODER_LAYERS = 4
    DIM_FEEDFORWARD = 512
    DROPOUT = 0.1
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class ProbabilisticTransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.input_projection = nn.Linear(len(config.FEATURES), config.D_MODEL)
        self.pos_encoder = PositionalEncoding(config.D_MODEL, config.DROPOUT)
        self.transformer = nn.Transformer(
            d_model=config.D_MODEL, nhead=config.N_HEAD,
            num_encoder_layers=config.N_ENCODER_LAYERS, num_decoder_layers=config.N_DECODER_LAYERS,
            dim_feedforward=config.DIM_FEEDFORWARD, dropout=config.DROPOUT, batch_first=True
        )
        self.output_projection = nn.Linear(config.D_MODEL, 2)

    def forward(self, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        src = self.input_projection(src) * math.sqrt(self.config.D_MODEL)
        src = self.pos_encoder(src)
        tgt = torch.zeros(src.size(0), self.config.PREDICTION_SEQ_LEN, self.config.D_MODEL, device=src.device)
        output = self.transformer(src, tgt)
        output = self.output_projection(output)
        mu, log_var = output[..., 0], output[..., 1]
        return mu, log_var

# ===== 2. 추론 및 시각화 실행 함수 =====

def run_inference_and_visualize():
    """학습된 모델로 단일 샘플에 대한 추론을 수행하고 결과를 시각화합니다."""
    
    cfg = ModelConfig()
    model_path = 'best_probabilistic_transformer.pth'
    
    if not os.path.exists(model_path):
        print(f"오류: 모델 파일 '{model_path}'을 찾을 수 없습니다. 학습을 먼저 완료해주세요.")
        return

    # --- 데이터 및 스케일러 준비 ---
    print("데이터 및 스케일러를 준비합니다...")
    df = pd.read_parquet(cfg.DATA_PATH)
    
    all_trip_ids = df['trip_id'].unique()
    train_val_ids, test_ids = train_test_split(all_trip_ids, test_size=0.2, random_state=42)
    train_ids, _ = train_test_split(train_val_ids, test_size=0.25, random_state=42)

    train_df = df[df['trip_id'].isin(train_ids)]
    test_df = df[df['trip_id'].isin(test_ids)].copy()

    # 입력 X를 스케일링하기 위해 feature_scaler는 여전히 필요합니다.
    feature_scaler = StandardScaler().fit(train_df[cfg.FEATURES])

    # --- 모델 불러오기 ---
    print(f"저장된 모델 '{model_path}'를 불러옵니다...")
    model = ProbabilisticTransformer(cfg).to(cfg.DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=cfg.DEVICE))
    model.eval()

    # --- 추론할 테스트 샘플 선택 ---
    random_trip_id = random.choice(test_ids)
    trip_data = test_df[test_df['trip_id'] == random_trip_id]
    
    seq_len = cfg.INPUT_SEQ_LEN + cfg.PREDICTION_SEQ_LEN
    if len(trip_data) < seq_len:
        print(f"샘플링된 주행 데이터(ID: {random_trip_id})가 너무 짧아 시각화할 수 없습니다. 다시 실행해주세요.")
        return

    start_idx = random.randint(0, len(trip_data) - seq_len)
    data_sample = trip_data.iloc[start_idx : start_idx + seq_len]

    # --- 예측 수행 ---
    input_features_scaled = feature_scaler.transform(data_sample[cfg.FEATURES].values[:cfg.INPUT_SEQ_LEN])
    input_tensor = torch.FloatTensor(input_features_scaled).unsqueeze(0).to(cfg.DEVICE)

    with torch.no_grad():
        # 모델의 출력은 이미 m/s 단위입니다. 변수명에서 _scaled를 제거합니다.
        mu, log_var = model(input_tensor)

    # --- 결과값 처리 (역변환 불필요) ---
    mu = mu.cpu().numpy().squeeze()
    ground_truth = data_sample[cfg.TARGET_FEATURE].values[cfg.INPUT_SEQ_LEN:]
    sigma = np.sqrt(np.exp(log_var.cpu().numpy().squeeze()))

    # --- 성능 지표 계산 ---
    mae = np.mean(np.abs(mu - ground_truth))
    rmse = np.sqrt(np.mean((mu - ground_truth)**2))
    print(f"\n--- 예측 성능 (샘플: {random_trip_id}) ---")
    print(f"MAE: {mae:.4f} m/s  (예측값과 실제값의 평균 차이)")
    print(f"RMSE: {rmse:.4f} m/s (오차의 변동성, 클수록 나쁨)")

    # --- 시각화 ---
    plt.figure(figsize=(16, 7))
    
    input_times = np.arange(cfg.INPUT_SEQ_LEN)
    pred_times = np.arange(cfg.INPUT_SEQ_LEN, cfg.INPUT_SEQ_LEN + cfg.PREDICTION_SEQ_LEN)
    
    past_speed = data_sample[cfg.TARGET_FEATURE].values[:cfg.INPUT_SEQ_LEN]
    plt.plot(input_times, past_speed, label='Past Input (Ground Truth)', color='gray', linestyle='--')
    
    plt.plot(pred_times, ground_truth, 'o-', color='blue', label='Future (Ground Truth)')
    plt.plot(pred_times, mu, 'o-', color='red', label='Predicted Mean (μ)')
    
    # sigma_unscaled 대신 sigma를 사용합니다.
    plt.fill_between(pred_times, mu - 2 * sigma, mu + 2 * sigma, 
                     color='red', alpha=0.2, label='Uncertainty (95% Confidence)')
    
    plt.title(f"Transformer Prediction vs. Actual (Trip ID: {random_trip_id})")
    plt.xlabel("Time (seconds from start of sequence)")
    plt.ylabel("Speed (m/s)")
    plt.legend()
    plt.grid(True, linestyle=':')
    plt.savefig("prediction_visualization.png")
    print("\n예측 결과 시각화 그래프가 'prediction_visualization.png' 파일로 저장되었습니다.")
    
if __name__ == '__main__':
    run_inference_and_visualize()