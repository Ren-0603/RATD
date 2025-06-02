import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
import torch
from torch.utils.data import DataLoader

from main_model import RATD_Forecasting
from dataset_forecasting import Dataset_Electricity as Dataset_Forecasting

import yaml

# ============================
# ① 設定ファイルとデータパスの読み込み
# ============================
with open('config/base_forecasting.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

csv_path = config["path"]["dataset_path"] + config["path"]["data_file"]
seq_len = config["diffusion"]["h_size"]
pred_len = config["diffusion"]["ref_size"]
label_len = 0
feature_dim = config["model"]["num_sample_features"]
target_dim = 1

# ============================
# ② データローダーの作成
# ============================
train_set = Dataset_Forecasting(config["path"]["dataset_path"], 'train',
    size=[seq_len, label_len, pred_len, feature_dim], features='M',
    data_path=config["path"]["data_file"], target='Close', scale=True, timeenc=0)
val_set = Dataset_Forecasting(config["path"]["dataset_path"], 'val',
    size=[seq_len, label_len, pred_len, feature_dim], features='M',
    data_path=config["path"]["data_file"], target='Close', scale=True, timeenc=0)
test_set = Dataset_Forecasting(config["path"]["dataset_path"], 'test',
    size=[seq_len, label_len, pred_len, feature_dim], features='M',
    data_path=config["path"]["data_file"], target='Close', scale=True, timeenc=0)

train_loader = DataLoader(train_set, batch_size=config["train"]["batch_size"], shuffle=True)
val_loader = DataLoader(val_set, batch_size=config["train"]["batch_size"], shuffle=False)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

# ============================
# ③ Diffusionモデル訓練
# ============================
model = RATD_Forecasting(config, config["device"], target_dim).to(config["device"])
optimizer = torch.optim.Adam(model.parameters(), lr=config["train"]["lr"])

loss_list = []

model.train()
for epoch in range(config["train"]["epochs"]):
    total_loss = 0
    for batch in train_loader:
        loss = model(batch, is_train=1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    loss_list.append(avg_loss)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")


# ============================
# ④ 30日間の予測（1サンプル）
# ============================
model.eval()
with torch.no_grad():
    for batch in test_loader:
        samples, observed_data, target_mask, observed_mask, observed_tp,cut_length = model.evaluate(batch, n_samples=1)
        print('samples.shape:', samples.shape)  # torch.Size([1, 1, 5, 210])
        one_sample = samples[0, 0, 0]  # shape: (5, 210)
        reshaped = one_sample.reshape(30, 7)  # shape: (30, 7)
        predicted = reshaped[:, 0].cpu().numpy()  # Close を予測と仮定
        break


pred_start_date = pd.read_csv(csv_path)['date'].iloc[-1]
pred_dates = pd.date_range(start=pred_start_date, periods=pred_len+1, freq='D')[1:]

plt.figure(figsize=(14, 6))
plt.plot(pred_dates, predicted, label='Forecast (Next 30 Days)', color='red')
plt.legend()
plt.grid(True)
plt.title("USD/JPY Forecast")
plt.show()

# ============================
# ⑤ Lossグラフの可視化
# ============================
plt.figure(figsize=(10, 4))
plt.plot(loss_list, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("RATD Diffusion Training Loss")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()



