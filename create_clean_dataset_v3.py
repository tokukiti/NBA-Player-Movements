import pandas as pd
import numpy as np
import json
import os
import time
import random
import torch
from torch_geometric.data import Data
from nba_api.stats.endpoints import playbyplayv2
from tqdm import tqdm

# --- 設定 ---
GAME_IDS = [
    '0021500003', '0021500021', '0021500030', '0021500044', '0021500055',
    '0021500062', '0021500073', '0021500086', '0021500095', '0021500109',
    '0021500118', '0021500129', '0021500143', '0021500155', '0021500168',
    '0021500178', '0021500189', '0021500202', '0021500213', '0021500223'
]
TRACKING_DIR = './data/2016.NBA.Raw.SportVU.Game.Logs'
PBP_CACHE_DIR = './data/pbp_cache' 
OUTPUT_PT = 'clean_shot_dataset_v4.pt'  

# バスケコート設定
BASKET_COORDS = np.array([88.75, 25.0]) 
HALF_COURT_X = 47.0
RIM_HEIGHT = 10.0

# 切り出し設定
WINDOW_PRE_PEAK = 3.0
WINDOW_POST_PEAK = 2.0
FPS = 25

# フォルダ作成
if not os.path.exists(PBP_CACHE_DIR):
    os.makedirs(PBP_CACHE_DIR)

# --- ★★★ 対策: ブラウザのフリをするためのヘッダー ★★★ ---
custom_headers = {
    'Host': 'stats.nba.com',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'en-US,en;q=0.9',
    'Referer': 'https://www.nba.com/',
    'Origin': 'https://www.nba.com',
    'Connection': 'keep-alive',
    'x-nba-stats-origin': 'stats',
    'x-nba-stats-token': 'true'
}

# --- 関数群 ---

def get_pbp_data_robust(game_id):
    csv_path = os.path.join(PBP_CACHE_DIR, f"pbp_{game_id}.csv")
    df = None

    # 1. キャッシュ確認
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            # print(f"  (Cache hit: {game_id})") # ログがうるさければコメントアウト
        except Exception as e:
            print(f"キャッシュ読み込みエラー: {e}")

    # 2. キャッシュがない場合、APIから取得
    if df is None:
        try:
            # ブロック回避のためランダムに待機 (2秒〜4秒)
            time.sleep(random.uniform(2.0, 4.0))
            
            # ★ ヘッダーとタイムアウトを指定してリクエスト
            pbp = playbyplayv2.PlayByPlayV2(
                game_id=game_id, 
                headers=custom_headers, 
                timeout=60
            )
            df = pbp.get_data_frames()[0]
            
            # 保存
            df.to_csv(csv_path, index=False)
            print(f" -> API取得成功: {game_id}")
            
        except Exception as e:
            print(f"PBP取得エラー(API) {game_id}: {e}")
            return {}

    # 3. ラベル辞書作成
    try:
        shot_df = df[df['EVENTMSGTYPE'].isin([1, 2])].copy()
        shot_map = {}
        for _, row in shot_df.iterrows():
            label = 1 if row['EVENTMSGTYPE'] == 1 else 0
            shot_map[int(row['EVENTNUM'])] = label
        return shot_map
    except Exception as e:
        print(f"データ処理エラー {game_id}: {e}")
        return {}

def calculate_kinematics(df):
    df = df.sort_values(by=['player_id', 'game_clock'], ascending=[True, False])
    df['dt'] = df['game_clock'].diff() * -1
    df['dt'] = df['dt'].fillna(0.04)
    df.loc[df['dt'] <= 0, 'dt'] = 0.04

    df['vx'] = df['x'].diff() / df['dt']
    df['vy'] = df['y'].diff() / df['dt']
    df['ax'] = df['vx'].diff() / df['dt']
    df['ay'] = df['vy'].diff() / df['dt']
    df[['vx', 'vy', 'ax', 'ay']] = df[['vx', 'vy', 'ax', 'ay']].fillna(0)
    return df

def analyze_and_extract_window(event_data, event_id):
    moments = event_data['moments']
    if not moments:
        return None

    ball_z = []
    for m in moments:
        b = next((x for x in m[5] if x[0] == -1), None)
        ball_z.append(b[4] if b else 0)

    ball_z = np.array(ball_z)
    
    # 物理フィルタ (10ftチェック)
    if np.max(ball_z) < RIM_HEIGHT:
        return None

    peak_idx = np.argmax(ball_z)
    pre_frames = int(WINDOW_PRE_PEAK * FPS)
    post_frames = int(WINDOW_POST_PEAK * FPS)
    
    start_idx = max(0, peak_idx - pre_frames)
    end_idx = min(len(moments), peak_idx + post_frames)
    
    if end_idx - start_idx < 25:
        return None

    valid_moments = moments[start_idx:end_idx]
    flat_rows = []
    moment_counter = 0
    
    for m in valid_moments:
        quarter, game_clock, shot_clock = m[0], m[2], m[3]
        for entity in m[5]:
            team_id, player_id, x, y, z = entity
            flat_rows.append([
                quarter, game_clock, shot_clock, 
                player_id, team_id, x, y, z, moment_counter
            ])
        moment_counter += 1
        
    df = pd.DataFrame(flat_rows, columns=[
        'quarter', 'game_clock', 'shot_clock', 
        'player_id', 'team_id', 'x', 'y', 'z', 'moment_index'
    ])
    df = calculate_kinematics(df)
    return df

def build_graph_data(play_df, label, game_id, event_id):
    moment_graphs = []
    for m_idx, m_df in play_df.groupby('moment_index'):
        ball_df = m_df[m_df['player_id'] == -1]
        players_df = m_df[m_df['player_id'] != -1].copy()
        
        if ball_df.empty or len(players_df) < 10:
            continue
            
        ball_x = ball_df['x'].iloc[0]
        if ball_x < HALF_COURT_X:
            players_df['x'] = 94.0 - players_df['x']
            players_df['y'] = 50.0 - players_df['y']
            ball_df['x'] = 94.0 - ball_df['x']
            ball_df['y'] = 50.0 - ball_df['y']
            players_df['vx'] = -players_df['vx']
            players_df['vy'] = -players_df['vy']
            players_df['ax'] = -players_df['ax']
            players_df['ay'] = -players_df['ay']
            ball_x = ball_df['x'].iloc[0]
            ball_y = ball_df['y'].iloc[0]
        else:
            ball_x = ball_df['x'].iloc[0]
            ball_y = ball_df['y'].iloc[0]

        player_coords = players_df[['x', 'y']].values
        basket_dist = np.linalg.norm(player_coords - BASKET_COORDS, axis=1)
        ball_dist = np.linalg.norm(player_coords - np.array([ball_x, ball_y]), axis=1)
        
        features = np.column_stack([
            players_df[['x', 'y', 'vx', 'vy', 'ax', 'ay']].values,
            basket_dist,
            ball_dist
        ])
        
        x_tensor = torch.tensor(features, dtype=torch.float)
        
        num_players = len(players_df)
        edge_index = []
        edge_attr = []
        for i in range(num_players):
            for j in range(num_players):
                if i != j:
                    edge_index.append([i, j])
                    dist = np.linalg.norm(features[i, :2] - features[j, :2])
                    edge_attr.append([dist])
                    
        edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr_tensor = torch.tensor(edge_attr, dtype=torch.float)
        
        data = Data(x=x_tensor, edge_index=edge_index_tensor, edge_attr=edge_attr_tensor)
        moment_graphs.append(data)
        
    if not moment_graphs:
        return None

    return {
        'game_id': game_id,
        'event_id': event_id,
        'graphs': moment_graphs,
        'label': torch.tensor([label], dtype=torch.long)
    }

def main():
    final_dataset = []
    print(f"--- 開始: 結果は {OUTPUT_PT} に保存されます ---")

    for game_id in GAME_IDS:
        print(f"\nProcessing Game {game_id}...")
        
        shot_labels = get_pbp_data_robust(game_id)
        if not shot_labels:
            print(" -> PBPデータ取得失敗、スキップ")
            continue
            
        json_path = os.path.join(TRACKING_DIR, f"{game_id}.json")
        if not os.path.exists(json_path):
            print(" -> JSONファイルなし、スキップ")
            continue
            
        try:
            with open(json_path, 'r') as f:
                game_data = json.load(f)
        except Exception as e:
            print(f" -> JSON読み込みエラー: {e}")
            continue
            
        valid_count = 0
        dropped_count = 0
        
        events = game_data.get('events', [])
        for event in tqdm(events, desc="Events"):
            eid = int(event.get('eventId', -1))
            
            if eid in shot_labels:
                label = shot_labels[eid]
                play_df = analyze_and_extract_window(event, eid)
                
                if play_df is not None:
                    graph_data = build_graph_data(play_df, label, game_id, eid)
                    if graph_data:
                        final_dataset.append(graph_data)
                        valid_count += 1
                    else:
                        dropped_count += 1
                else:
                    dropped_count += 1
        
        print(f" -> Game {game_id}: 採用 {valid_count} / 除外 {dropped_count}")

    if final_dataset:
        torch.save(final_dataset, OUTPUT_PT)
        print(f"\n🎉 完了！ 合計 {len(final_dataset)} 件のデータを保存しました。")
    else:
        print("\n😱 データが生成されませんでした。")

if __name__ == "__main__":
    main()