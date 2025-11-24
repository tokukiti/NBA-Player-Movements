import pandas as pd
from nba_api.stats.endpoints import playbyplayv2
import time
import os
from tqdm import tqdm
import numpy as np
import json

# --- 設定 (変更なし) ---
GAME_IDS = [
    '0021500003', '0021500021', '0021500030', '0021500044', '0021500055',
    '0021500062', '0021500073', '0021500086', '0021500095', '0021500109',
    '0021500118', '0021500129', '0021500143', '0021500155', '0021500168',
    '0021500178', '0021500189', '0021500202', '0021500213', '0021500223'
]
TRACKING_DIR = './data/2016.NBA.Raw.SportVU.Game.Logs'
OUTPUT_CSV = 'features_and_labels_20_GAMES.csv'
SECONDS_BEFORE_SHOT = 5
MOMENTS_BEFORE = int(25 * SECONDS_BEFORE_SHOT) 
MOMENTS_AFTER = int(25 * 0.5) 

# --- PBPデータ取得 (変更なし) ---
def get_pbp_data(game_id):
    print(f"\nProcessing {game_id}: PBPデータを取得中...")
    try:
        pbp = playbyplayv2.PlayByPlayV2(game_id)
        pbp_df = pbp.get_data_frames()[0]
        return pbp_df
    except Exception as e:
        print(f"  エラー: PBP取得失敗。スキップします。 {e}")
        return None

# --- ★★★ v3.2: JSONトラッキングデータ取得ロジック (加速度対応) ★★★ ---
def parse_tracking_data_from_json(file_path):
    print(f"    .json ファイル {file_path} をパース中...")
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    moments_data = []
    moment_index_counter = 0
    
    for event in data['events']:
        for moment in event['moments']:
            quarter, game_clock, shot_clock = moment[0], moment[2], moment[3]
            # ボールデータ
            ball_data = moment[5][0]
            moments_data.append([
                quarter, game_clock, shot_clock, -1, 
                ball_data[2], ball_data[3], ball_data[4], moment_index_counter
            ])
            # 選手データ
            for player_data in moment[5][1:]:
                moments_data.append([
                    quarter, game_clock, shot_clock, player_data[1], 
                    player_data[2], player_data[3], np.nan, moment_index_counter
                ])
            moment_index_counter += 1
            
    df = pd.DataFrame(moments_data, columns=[
        'quarter', 'game_clock', 'shot_clock', 
        'player_id', 'x', 'y', 'z', 'moment_index'
    ])
    
    # --- 速度 (vx, vy) と 加速度 (ax, ay) の計算 ---
    print("    速度(vx, vy)と加速度(ax, ay)を計算中...")
    df = df.sort_values(by=['player_id', 'moment_index'])
    
    # 時間差 (dt)
    df['dt'] = df['game_clock'].diff(-1) * -1 
    df.loc[df['player_id'] != df['player_id'].shift(-1), 'dt'] = np.nan
    df['dt'] = df['dt'].replace(0, np.nan).fillna(0.04) # 25fps
    
    # 速度 (v = dx/dt)
    df['vx'] = df['x'].diff(-1) * -1 / df['dt']
    df['vy'] = df['y'].diff(-1) * -1 / df['dt']
    df.loc[df['player_id'] != df['player_id'].shift(-1), ['vx', 'vy']] = np.nan

    # ★★★ ここからが v3.2 の追加箇所 ★★★
    # 加速度 (a = dv/dt)
    # 速度(vx, vy)の差分を計算
    df['ax'] = df['vx'].diff(-1) * -1 / df['dt']
    df['ay'] = df['vy'].diff(-1) * -1 / df['dt']
    # プレーヤーが変わる境界で加速度もリセット
    df.loc[df['player_id'] != df['player_id'].shift(-1), ['ax', 'ay']] = np.nan
    
    # 計算不能な値（NaN）を0で埋める
    # ★ ax, ay を追加
    df[['vx', 'vy', 'ax', 'ay']] = df[['vx', 'vy', 'ax', 'ay']].fillna(0)
    # ★★★ v3.2 追加箇所ここまで ★★★

    # game_clock_seconds の計算 (変更なし)
    df['game_clock_seconds'] = (df['quarter'] > 4) * (720 + (df['quarter'] - 5) * 300) + \
                              (df['quarter'] <= 4) * (720 - (df['quarter'] - 1) * 720) + \
                              (720 - df['game_clock']) 
    return df

def load_tracking_data(game_id):
    tracking_file = os.path.join(TRACKING_DIR, f"{game_id}.json")
    if not os.path.exists(tracking_file):
        print(f"  エラー: トラッキングファイル {tracking_file} が見つかりません。")
        return None
    print(f"  トラッキングデータ {tracking_file} をロード中...")
    try:
        df = parse_tracking_data_from_json(tracking_file)
        return df
    except Exception as e:
        print(f"  エラー: JSONファイルのパースに失敗しました。 {e}")
        return None

# --- (これ以降の PBP照合、ゲーム処理、CSV保存ロジックは v3.1 から変更なし) ---

def find_closest_moment(pbp_time_seconds, tracking_times):
    pbp_time_seconds = float(pbp_time_seconds)
    time_diffs = -np.abs(tracking_times - pbp_time_seconds)
    closest_moment_index = time_diffs.idxmax()
    if np.abs(tracking_times[closest_moment_index] - pbp_time_seconds) > 1.5:
        return None
    return closest_moment_index

def process_game(game_id, pbp_df, tracking_df):
    print(f"  {game_id}: PBPとトラッキングデータを照合中...")
    shot_events = pbp_df[pbp_df['EVENTMSGTYPE'].isin([1, 2])].copy()
    
    shot_events['PCTIMESTRING'] = shot_events['PCTIMESTRING'].astype(str)
    shot_events[['Minutes', 'Seconds']] = shot_events['PCTIMESTRING'].str.split(':', expand=True).astype(float)
    shot_events['pbp_time_seconds'] = (shot_events['PERIOD'] > 4) * (720 + (shot_events['PERIOD'] - 5) * 300) + \
                                      (shot_events['PERIOD'] <= 4) * (720 - (shot_events['PERIOD'] - 1) * 720) + \
                                      (720 - (shot_events['Minutes'] * 60 + shot_events['Seconds']))
    shot_events['play_result_label'] = shot_events['EVENTMSGTYPE'].apply(lambda x: 1 if x == 1 else 0)
    
    tracking_times = tracking_df['game_clock_seconds']
    grouped_tracking = tracking_df.groupby('moment_index')
    all_game_features = []
    
    for _, pbp_row in tqdm(shot_events.iterrows(), total=len(shot_events), desc="  ショットイベント処理中"):
        closest_row_index = find_closest_moment(pbp_row['pbp_time_seconds'], tracking_times)
        if closest_row_index is None: continue
        closest_moment_index = tracking_df.loc[closest_row_index, 'moment_index']
            
        start_index = max(0, closest_moment_index - MOMENTS_BEFORE)
        end_index = closest_moment_index + MOMENTS_AFTER
        play_moments_df_list = []
        moment_in_play_counter = 0 
        
        for moment_index in range(start_index, end_index):
            try:
                moment_df = grouped_tracking.get_group(moment_index).copy()
                moment_df['game_id'] = game_id 
                moment_df['event_id'] = pbp_row['EVENTNUM'] 
                moment_df['moment_index'] = moment_in_play_counter
                play_moments_df_list.append(moment_df)
                moment_in_play_counter += 1
            except KeyError:
                continue 

        if play_moments_df_list:
            play_df = pd.concat(play_moments_df_list)
            all_game_features.append((play_df, pbp_row['play_result_label']))
            
    return all_game_features

def save_features_to_csv(all_features, filename):
    print(f"\n全ゲームの特徴量を {filename} に保存中...")
    features_list = []
    for play_df, label in all_features:
        play_df['play_result_label'] = label
        features_list.append(play_df)
    if not features_list:
        print("CSVに保存する特徴量がありません。")
        return
    final_df = pd.concat(features_list)
    final_df.to_csv(filename, index=False)
    print(f"✅ {len(features_list)} プレー分のデータを保存完了。")

def main():
    all_features_all_games = []
    for game_id in GAME_IDS:
        pbp_df = get_pbp_data(game_id)
        if pbp_df is None: continue
        tracking_df = load_tracking_data(game_id)
        if tracking_df is None: continue
        game_features = process_game(game_id, pbp_df, tracking_df)
        all_features_all_games.extend(game_features)
        time.sleep(1) 
    save_features_to_csv(all_features_all_games, OUTPUT_CSV)

if __name__ == "__main__":
    main()