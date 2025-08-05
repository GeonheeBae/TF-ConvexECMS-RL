import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
import math
import os

# ===== 성능 튜닝 =====
torch.backends.cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# --- 1. 설정 (Configuration) -------------------------------------------------
class ModelConfig:
    """모델 및 훈련 하이퍼파라미터"""
    # 데이터 경로 및 특징
    DATA_PATH = r'C:\Users\CNL\Desktop\train\data\VED_final_with_grade.parquet'
    FEATURES = ['speed_mps', 'acceleration_mps2', 'hour', 'day_of_week', 'grade']
    TARGET_FEATURE = 'speed_mps'
    
    # 데이터 및 시퀀스 설정
    INPUT_SEQ_LEN = 60
    PREDICTION_SEQ_LEN = 30

    # 모델 하이퍼파라미터
    D_MODEL = 128
    N_HEAD = 8
    N_ENCODER_LAYERS = 4
    N_DECODER_LAYERS = 4
    DIM_FEEDFORWARD = 512
    DROPOUT = 0.1

    # 훈련 하이퍼파라미터 (요청대로 고정)
    BATCH_SIZE = 256            # 3080이면 128~256 권장. OOM시 64로 낮추세요.
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    INDEX_STRIDE = 5
    EPOCHS = 20                 # ★ 요청: 고정 20 epoch
    NUM_WORKERS = 0             # ★ 요청: 고정 0 (Windows 안전)
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    ACCUM_STEPS = 1             # VRAM 부족시 2~4로

    PATIENCE = 5
    SAVE_PATH = 'best_probabilistic_transformer.pth'

# --- 2. 모델 아키텍처 ---------------------------------------------------------
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

# --- 3. 손실 함수 --------------------------------------------------------------
class GaussianNLLLoss(nn.Module):
    def __init__(self, epsilon: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, mu: torch.Tensor, log_var: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        var = torch.exp(log_var) + self.epsilon
        log_prob = -0.5 * (torch.log(2 * math.pi * var) + (y_true - mu)**2 / var)
        return -torch.mean(log_prob)

# --- 4. 데이터셋 ---------------------------------------------------------------
class VEDTimeSeriesDataset(Dataset):
    """stride 간격 샘플링 + 트립별 랜덤 offset + 에폭별 재샘플링 지원"""
    def __init__(self, df: pd.DataFrame, config: ModelConfig, scaler: StandardScaler,
                 stride: int = 1, seed: int = 42):
        self.config = config
        self.stride = max(1, int(stride))
        self._base_seed = int(seed)
        self.rng = np.random.default_rng(self._base_seed)

        self.df = df.copy()
        # 입력 특징만 스케일 (타깃은 스케일 X)
        self.df[config.FEATURES] = scaler.transform(self.df[config.FEATURES])

        # 트립별 그룹화 (num_workers=0 기준 안전)
        self.grouped_by_trip = dict(tuple(self.df.groupby('trip_id')))

        # 초기 인덱스 구성
        self._build_indices()

    def _build_indices(self):
        """stride와 랜덤 offset을 반영해 (trip_id, start) 인덱스 리스트 구성"""
        self.indices = []
        L_in, L_out = self.config.INPUT_SEQ_LEN, self.config.PREDICTION_SEQ_LEN
        for trip_id, group in self.grouped_by_trip.items():
            nseq = len(group) - L_in - L_out + 1
            if nseq <= 0:
                continue
            if self.stride == 1:
                start_offset = 0
            else:
                # 트립마다 시작 offset을 0~(stride-1)에서 무작위 선택 → 편향 완화
                start_offset = int(self.rng.integers(0, self.stride))
            self.indices.extend([(trip_id, i) for i in range(start_offset, nseq, self.stride)])

    def set_epoch(self, epoch: int):
        """에폭마다 다른 offset으로 재샘플링(커버리지↑)"""
        if self.stride > 1:
            self.rng = np.random.default_rng(self._base_seed + int(epoch))
            self._build_indices()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        trip_id, start_offset = self.indices[idx]
        trip_data = self.grouped_by_trip[trip_id]

        end_offset = start_offset + self.config.INPUT_SEQ_LEN + self.config.PREDICTION_SEQ_LEN
        sl = trip_data.iloc[start_offset:end_offset]

        x = sl[self.config.FEATURES].values
        y = sl[self.config.TARGET_FEATURE].values
        x_seq = x[:self.config.INPUT_SEQ_LEN]
        y_seq = y[self.config.INPUT_SEQ_LEN:]

        return torch.from_numpy(x_seq).float(), torch.from_numpy(y_seq).float()

# --- 5. 학습 / 평가 루프 -------------------------------------------------------
def train_one_epoch(model, dataloader, loss_fn, optimizer, device, scaler, accum_steps=1):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    for X, y in tqdm(dataloader, desc="Training"):
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.amp.autocast('cuda', dtype=torch.float16):
            mu, log_var = model(X)
            loss = loss_fn(mu, log_var, y) / accum_steps

        scaler.scale(loss).backward()

        # gradient accumulation
        if accum_steps == 1 or (torch.distributed.is_available() and torch.distributed.is_initialized()):
            pass
        if (train_one_epoch.step_count + 1) % accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * accum_steps
        train_one_epoch.step_count += 1

    # 마지막에 남은 그라디언트 처리
    if (train_one_epoch.step_count % accum_steps) != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    return total_loss / max(1, len(dataloader))
train_one_epoch.step_count = 0

@torch.no_grad()
def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss, total_mae, total_rmse = 0.0, 0.0, 0.0
    for X, y in tqdm(dataloader, desc="Evaluating"):
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.amp.autocast('cuda', dtype=torch.float16):
            mu, log_var = model(X)
            loss = loss_fn(mu, log_var, y)

        total_loss += loss.item()
        # 타깃은 스케일 안 했으므로 그대로 사용
        total_mae  += torch.mean(torch.abs(mu - y)).item()
        total_rmse += torch.sqrt(torch.mean((mu - y) ** 2)).item()

    n = max(1, len(dataloader))
    return total_loss / n, total_mae / n, total_rmse / n

# --- 6. 유틸: DataLoader 생성기 ------------------------------------------------
def make_loader(dataset, config: ModelConfig, shuffle: bool):
    return DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=config.NUM_WORKERS,  # 요청대로 0 고정
        pin_memory=True,
    )

# --- 7. 메인 실행 --------------------------------------------------------------
if __name__ == '__main__':
    cfg = ModelConfig()
    print(f"Device = {cfg.DEVICE}, Batch Size = {cfg.BATCH_SIZE}, Num Workers = {cfg.NUM_WORKERS}, Epochs = {cfg.EPOCHS}")

    print(f"Loading full dataset '{cfg.DATA_PATH}' into RAM...")
    df = pd.read_parquet(cfg.DATA_PATH)

    all_trip_ids = df['trip_id'].unique()
    train_ids, val_ids = train_test_split(all_trip_ids, test_size=0.2, random_state=42)

    train_df = df[df['trip_id'].isin(train_ids)]
    val_df   = df[df['trip_id'].isin(val_ids)]

    print(f"Fitting scalers on training data ({len(train_ids)} trips)...")
    feature_scaler = StandardScaler().fit(train_df[cfg.FEATURES])

    print("Creating datasets...")
    train_dataset = VEDTimeSeriesDataset(train_df, cfg, feature_scaler, stride=cfg.INDEX_STRIDE, seed=42)
    val_dataset   = VEDTimeSeriesDataset(val_df,   cfg, feature_scaler, stride=1,                seed=43)
    
    train_loader = make_loader(train_dataset, cfg, shuffle=True)
    val_loader   = make_loader(val_dataset,   cfg, shuffle=False)

    model = ProbabilisticTransformer(cfg).to(cfg.DEVICE)
    # 선택적: torch.compile (되면 속도 ↑)
    try:
        import triton  # 어차피 없으니 except로 빠짐
        model = torch.compile(model)
        print(">> torch.compile enabled")
    except Exception as e:
        print(">> torch.compile disabled (fallback to eager):", e)

    loss_fn = GaussianNLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, verbose=True)
    scaler = torch.amp.GradScaler('cuda', enabled=(cfg.DEVICE == 'cuda'))

    best_val_loss = float('inf')
    patience_counter = 0

    print("\n--- Starting Training ---")
    for epoch in range(cfg.EPOCHS):
        train_dataset.set_epoch(epoch)
        train_loader = make_loader(train_dataset, cfg, shuffle=True)
        
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, cfg.DEVICE, scaler, accum_steps=cfg.ACCUM_STEPS)
        val_loss, val_mae, val_rmse = evaluate(model, val_loader, loss_fn, cfg.DEVICE)

        scheduler.step(val_loss)
        print(f"Epoch {epoch+1:02d}/{cfg.EPOCHS} | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val MAE: {val_mae:.4f} m/s | Val RMSE: {val_rmse:.4f} m/s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), cfg.SAVE_PATH)
            print(f"----> Best model saved at epoch {epoch+1} (val loss: {best_val_loss:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg.PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break
