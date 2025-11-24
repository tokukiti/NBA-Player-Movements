import pandas as pd
import torch
from torch_geometric.data import Data
from tqdm import tqdm
import numpy as np

# --- 設定 ---
INPUT_CSV = 'features_and_labels_20_GAMES.csv' # v3.2 pipeline で作成したもの
# ★ v5.1用の出力ファイル名
OUTPUT_PT = 'final_play_sequence_dataset_v5.1_with_ID.pt' 
BASKET_COORDS = (88.75, 25.0) 
HALF_COURT_X = 47.0
MIN_MOMENTS_PER_PLAY = 10 

# --- データの読み込み ---
try:
    df = pd.read_csv(INPUT_CSV)
except FileNotFoundError:
    print(f"エラー: {INPUT_CSV} が見つかりません。pipeline.py (v3.2) を先に実行してください。")
    exit()

print(f"--- {INPUT_CSV} から「ID付き(v5.1)」時系列プレーデータセットを構築します ---")

grouped_plays = df.groupby(['game_id', 'event_id'])
play_list = [] 

for (game_id, event_id), play_df in tqdm(grouped_plays, desc="プレーを構築中"):
    
    play_df = play_df.sort_values(by='moment_index')
    moment_graphs = [] 
    play_label = play_df['play_result_label'].iloc[0]

    for moment_idx, moment_df in play_df.groupby('moment_index'):
        
        players_df = moment_df[moment_df['player_id'] != -1].copy()
        ball_df = moment_df[moment_df['player_id'] == -1].copy()
        
        if len(players_df) != 10 or ball_df.empty:
            continue

        # --- 座標標準化 (変更なし) ---
        is_attacking_left_basket = ball_df['x'].iloc[0] < HALF_COURT_X
        if is_attacking_left_basket:
            players_df['x'] = 94.0 - players_df['x']
            players_df['y'] = 50.0 - players_df['y']
            ball_df['x'] = 94.0 - ball_df['x']
            ball_df['y'] = 50.0 - ball_df['y']
            players_df['vx'] = -players_df['vx']
            players_df['vy'] = -players_df['vy']
            players_df['ax'] = -players_df['ax']
            players_df['ay'] = -players_df['ay']
        
        # --- 特徴量計算 (変更なし) ---
        player_coords = players_df[['x', 'y']].values
        ball_coords = ball_df[['x', 'y']].iloc[0].values
        players_df['dist_to_basket'] = np.linalg.norm(player_coords - BASKET_COORDS, axis=1)
        players_df['dist_to_ball'] = np.linalg.norm(player_coords - ball_coords, axis=1)

        # --- ノード特徴量 (8次元) (変更なし) ---
        feature_columns = ['x', 'y', 'vx', 'vy', 'ax', 'ay', 'dist_to_basket', 'dist_to_ball']
        x = torch.tensor(players_df[feature_columns].values, dtype=torch.float)
        
        # --- エッジとエッジ特徴量 (変更なし) ---
        edge_list, edge_features = [], []
        for i in range(10):
            for j in range(10):
                if i != j:
                    edge_list.append([i, j])
                    dist = np.linalg.norm(player_coords[i] - player_coords[j])
                    edge_features.append([dist])
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float)

        graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        moment_graphs.append(graph_data)

    if len(moment_graphs) >= MIN_MOMENTS_PER_PLAY: 
        # ★★★ v5.1 変更点 ★★★
        # 辞書に game_id と event_id を追加
        play_list.append({
            'game_id': game_id,
            'event_id': event_id,
            'graphs': moment_graphs,
            'label': torch.tensor([play_label], dtype=torch.long)
        })

if play_list:
    torch.save(play_list, OUTPUT_PT)
    print(f"\n✅ 処理完了！ {len(play_list)}個のプレーを '{OUTPUT_PT}' に保存しました。")
    print(f"ノード特徴量の次元: {play_list[0]['graphs'][0].num_node_features}")
else:
    print("\n❌ プレーを作成できるデータが見つかりませんでした。")