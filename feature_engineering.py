import json
import pandas as pd
from tqdm import tqdm

# --- 設定 ---
GAME_ID = '0021500001'
LABELED_DATA_FILE = 'master_labeled_dataset_TIMEMATCH.csv'
TRACKING_FILE = f'data/2016.NBA.Raw.SportVU.Game.Logs/{GAME_ID}.json'
OUTPUT_CSV = 'features_and_labels.csv'

# --- データの読み込み ---
try:
    labels_df = pd.read_csv(LABELED_DATA_FILE)
    with open(TRACKING_FILE, 'r') as f:
        tracking_data = json.load(f)
    all_tracking_events = tracking_data['events']
except FileNotFoundError as e:
    print(f"エラー: ファイルが見つかりません。{e}")
    exit()

# --- 特徴量計算 ---
final_data = []

# ★★★★★ 新しい核心ロジック ★★★★★
# まず、トラッキングデータを「開始時間」をキーとする辞書に変換して高速に検索できるようにする
print("トラッキングデータを時間で索引付けしています...")
tracking_clock_map = {}
for event in all_tracking_events:
    if event['moments']:
        start_clock = event['moments'][0][2]
        tracking_clock_map[start_clock] = event

print(f"'{LABELED_DATA_FILE}'に基づき、{len(labels_df)}件のショットプレーの特徴量を計算します...")

# labels_dfの各行（＝ショットプレー）をループ処理
for index, play in tqdm(labels_df.iterrows(), total=labels_df.shape[0], desc="特徴量を計算中"):
    event_id_pbp = play['event_id_PBP']
    tracking_start_clock = play['tracking_start_clock']
    play_result_label = play['play_result_label']

    # 索引から、開始時間が一致するトラッキングイベントを直接取得
    if tracking_start_clock in tracking_clock_map:
        tracking_event = tracking_clock_map[tracking_start_clock]
        moments = tracking_event['moments']
        
        for i in range(1, len(moments)): # 速度計算のため2番目のモーメントから開始
            moment = moments[i]
            prev_moment = moments[i-1]

            # 時間差を計算
            time_delta = moment[2] - prev_moment[2]
            if time_delta <= 0: continue

            # 選手とボールのリストを取得
            entities = moment[5]
            prev_entities = prev_moment[5]

            for j in range(len(entities)):
                team_id, player_id, x, y, z = entities[j]
                _, _, prev_x, prev_y, _ = prev_entities[j]

                # 速度を計算
                vx = (x - prev_x) / time_delta
                vy = (y - prev_y) / time_delta

                final_data.append({
                    'game_id': GAME_ID,
                    'event_id': event_id_pbp,
                    'moment_index': i,
                    'team_id': team_id,
                    'player_id': player_id,
                    'x': x, 'y': y, 'z': z,
                    'vx': vx, 'vy': vy,
                    'play_result_label': play_result_label
                })

# --- 結果をCSVに保存 ---
if final_data:
    final_df = pd.DataFrame(final_data)
    final_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"\n✅ 処理完了！特徴量計算後のデータセットを {OUTPUT_CSV} に保存しました。")
    print(f"合計 {final_df['event_id'].nunique()} 件のプレーから、{len(final_df)}行の時系列データを生成しました。")
    print("\n--- 完成したデータセットの形式（先頭5行） ---")
    print(final_df.head())
else:
    print("\n❌ 特徴量を計算できるデータが見つかりませんでした。")