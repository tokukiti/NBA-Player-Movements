import pandas as pd
import numpy as np
import json
import os
import torch
from torch_geometric.data import Data
from tqdm import tqdm

# --- 設定 ---
INPUT_CSV = 'cleaned_shots_data.csv' 
TRACKING_DIR = './data/2016.NBA.Raw.SportVU.Game.Logs' # ★ここが正しいか要確認
OUTPUT_PT = 'dataset_v11_pipeline.pt'

# コート設定
BASKET_COORDS = np.array([88.75, 25.0])
HALF_COURT_X = 47.0
FPS = 25

# 時間切り出し設定
PRE_EVENT_SEC = 4.0 
POST_EVENT_SEC = 1.0
MIN_FRAMES = 25

# チームID辞書
TEAM_ABBR_TO_ID = {
    'ATL': 1610612737, 'BOS': 1610612738, 'CLE': 1610612739, 'NOP': 1610612740,
    'CHI': 1610612741, 'DAL': 1610612742, 'DEN': 1610612743, 'GSW': 1610612744,
    'HOU': 1610612745, 'LAC': 1610612746, 'LAL': 1610612747, 'MIA': 1610612748,
    'MIL': 1610612749, 'MIN': 1610612750, 'BKN': 1610612751, 'NYK': 1610612752,
    'ORL': 1610612753, 'IND': 1610612754, 'PHI': 1610612755, 'PHX': 1610612756,
    'POR': 1610612757, 'SAC': 1610612758, 'SAS': 1610612759, 'OKC': 1610612760,
    'TOR': 1610612761, 'UTA': 1610612762, 'MEM': 1610612763, 'WAS': 1610612764,
    'DET': 1610612765, 'CHA': 1610612766
}

def get_game_moments(json_path):
    if not os.path.exists(json_path): 
        # ★デバッグ出力: ファイルが見つからない場合
        print(f"  [DEBUG] File not found: {os.path.abspath(json_path)}")
        return None
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        all_moments = []
        for event in data['events']:
            all_moments.extend(event.get('moments', []))
        return all_moments
    except Exception as e:
        print(f"  [DEBUG] JSON Load Error: {e}")
        return None

def extract_time_window(moments, target_q, target_sec):
    indices_in_q = [i for i, m in enumerate(moments) if m[0] == target_q]
    if not indices_in_q: return None

    best_idx = -1
    min_diff = 9999.0
    for i in indices_in_q:
        time_remaining = moments[i][2]
        diff = abs(time_remaining - target_sec)
        if diff < min_diff:
            min_diff = diff
            best_idx = i
        if diff < 0.1: break
            
    if best_idx == -1 or min_diff > 2.0: return None

    start_idx = max(0, best_idx - int(PRE_EVENT_SEC * FPS))
    end_idx = min(len(moments), best_idx + int(POST_EVENT_SEC * FPS))
    
    if end_idx - start_idx < MIN_FRAMES: return None
    return moments[start_idx : end_idx]

def calculate_kinematics_and_graph(play_df, label, game_id, event_id, offense_team_id):
    play_df = play_df.sort_values(by=['player_id', 'moment_index'])
    play_df['dt'] = 0.04 
    
    play_df['vx'] = play_df['x'].diff() / play_df['dt']
    play_df['vy'] = play_df['y'].diff() / play_df['dt']
    play_df['ax'] = play_df['vx'].diff() / play_df['dt']
    play_df['ay'] = play_df['vy'].diff() / play_df['dt']
    play_df.fillna(0, inplace=True)

    ball_start_x = play_df[play_df['player_id'] == -1]['x'].iloc[0]
    if ball_start_x < HALF_COURT_X:
        play_df['x'] = 94.0 - play_df['x']
        play_df['y'] = 50.0 - play_df['y']
        play_df['vx'] = -play_df['vx']
        play_df['vy'] = -play_df['vy']
        play_df['ax'] = -play_df['ax']
        play_df['ay'] = -play_df['ay']

    play_df['is_offense'] = play_df.apply(
        lambda row: 1.0 if (row['team_id'] == offense_team_id or row['player_id'] == -1) else 0.0, 
        axis=1
    )

    moment_graphs = []
    for _, m_df in play_df.groupby('moment_index'):
        ball_df = m_df[m_df['player_id'] == -1]
        players_df = m_df[m_df['player_id'] != -1]
        
        if ball_df.empty or len(players_df) < 10: continue

        p_coords = players_df[['x', 'y']].values
        b_coords = ball_df[['x', 'y']].iloc[0].values

        dist_basket = np.linalg.norm(p_coords - BASKET_COORDS, axis=1)
        dist_ball = np.linalg.norm(p_coords - b_coords, axis=1)
        
        node_feats = np.column_stack([
            players_df[['x', 'y', 'vx', 'vy', 'ax', 'ay']].values,
            dist_basket, dist_ball, players_df[['is_offense']].values
        ])
        
        x = torch.tensor(node_feats, dtype=torch.float)
        
        num_nodes = len(players_df)
        edge_index, edge_attr = [], []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    edge_index.append([i, j])
                    dist = np.linalg.norm(node_feats[i, :2] - node_feats[j, :2])
                    edge_attr.append([dist])
        
        data = Data(
            x=x, 
            edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(), 
            edge_attr=torch.tensor(edge_attr, dtype=torch.float)
        )
        moment_graphs.append(data)
    
    if not moment_graphs: return None

    return {
        'game_id': game_id,
        'event_id': event_id,
        'label': torch.tensor([label], dtype=torch.long), 
        'graphs': moment_graphs
    }

def main():
    if not os.path.exists(INPUT_CSV):
        print(f"エラー: {INPUT_CSV} がありません。pipeline.py を先に実行してください。")
        return

    # ★ CSVの読み込み時に型を指定して、GAME_IDが '00215...' の文字列として扱われるようにする
    df = pd.read_csv(INPUT_CSV, dtype={'GAME_ID': str}) 
    
    unique_games = df['GAME_ID'].unique()
    target_games = unique_games[:20] 
    df = df[df['GAME_ID'].isin(target_games)]
    
    print(f"--- Processing {len(target_games)} Games ---")
    print(f"Target Game IDs example: {target_games[:3]}") # IDが正しく文字列になっているか確認
    
    final_dataset = []
    
    for game_id in tqdm(target_games, desc="Games"):
        game_shots = df[df['GAME_ID'] == game_id]
        
        # JSONファイル名のパスを作成
        json_filename = f"{game_id}.json"
        json_path = os.path.join(TRACKING_DIR, json_filename)
        
        all_moments = get_game_moments(json_path)
        if all_moments is None:
            # 理由が詳しく出るように修正済み
            continue
            
        for _, row in game_shots.iterrows():
            target_q = row['Quarter']
            target_sec = row['SecLeft']
            event_id = row['EVENTNUM']
            
            is_away_play = pd.notna(row['AwayPlay']) and str(row['AwayPlay']).strip() != ""
            team_abbr = row['AwayTeam'] if is_away_play else row['HomeTeam']
            offense_team_id = TEAM_ABBR_TO_ID.get(team_abbr)
            if offense_team_id is None: continue

            label = 1 if row['ShotOutcome'] == 'make' else 0
            
            shot_moments = extract_time_window(all_moments, target_q, target_sec)
            
            if shot_moments:
                flat_rows = []
                for i, m in enumerate(shot_moments):
                    for entity in m[5]:
                        flat_rows.append([
                            m[0], m[2], m[3], 
                            entity[0], entity[1], entity[2], entity[3], entity[4], 
                            i 
                        ])
                
                play_df = pd.DataFrame(flat_rows, columns=[
                    'quarter', 'game_clock', 'shot_clock', 
                    'team_id', 'player_id', 'x', 'y', 'z', 'moment_index'
                ])
                
                graph_data = calculate_kinematics_and_graph(play_df, label, game_id, event_id, offense_team_id)
                if graph_data:
                    final_dataset.append(graph_data)

    if final_dataset:
        torch.save(final_dataset, OUTPUT_PT)
        print(f"\n✅ 完了！ {len(final_dataset)} プレーを保存しました: {OUTPUT_PT}")
    else:
        print("\n❌ データが生成されませんでした。")

if __name__ == "__main__":
    main()