import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.data import Batch
from torch.nn import LSTM, Linear, Dropout

class STGAT(torch.nn.Module):
    def __init__(self, node_features, edge_features, hidden_channels, out_channels=1):
        super(STGAT, self).__init__()
        
        # --- 1. 空間的特徴抽出 (Spatial) ---
        self.gat1 = GATv2Conv(node_features, hidden_channels, heads=3, edge_dim=edge_features)
        self.gat2 = GATv2Conv(hidden_channels * 3, hidden_channels, heads=1, edge_dim=edge_features)

        # --- 2. 時間的特徴抽出 (Temporal) ---
        self.lstm = LSTM(input_size=hidden_channels, 
                         hidden_size=hidden_channels, 
                         batch_first=True)

        # --- 3. 分類器 (Classifier) ---
        self.dropout = Dropout(p=0.5)
        self.lin1 = Linear(hidden_channels, hidden_channels // 2)
        self.lin2 = Linear(hidden_channels // 2, out_channels)

    def forward(self, data_list):
        """
        data_list: 1プレー分のグラフデータのリスト (長さ: フレーム数)
        """
        
        # ★★★ 修正箇所 ★★★
        # 1. まずCPU上でリストを結合してバッチ化する (Batch.from_data_list)
        # 2. その後、このモデルのパラメータが存在するデバイス(GPU)へ転送する
        #    next(self.parameters()).device は「このモデルが今いるデバイス」を自動取得します
        device = next(self.parameters()).device
        batch_data = Batch.from_data_list(data_list).to(device)
        
        x, edge_index, edge_attr = batch_data.x, batch_data.edge_index, batch_data.edge_attr
        
        # 1. GCN (Spatial) - 全フレーム一括計算
        x = self.gat1(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)
        x = self.gat2(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)

        # 2. Pooling - 各フレームごとの代表ベクトルを取り出す
        frame_features = global_mean_pool(x, batch_data.batch)
        
        # 3. LSTM (Temporal)
        # [Time, Features] -> [1, Time, Features] に変形 (Batch size = 1 assumption)
        seq_tensor = frame_features.unsqueeze(0)
        
        lstm_out, (h_n, c_n) = self.lstm(seq_tensor)

        # 最終時刻の隠れ層
        last_hidden = h_n[-1]

        # 4. Classifier
        out = self.dropout(last_hidden)
        out = F.relu(self.lin1(out))
        out = self.lin2(out)
        
        return out