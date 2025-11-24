import json
import pandas as pd
from tqdm import tqdm # 進捗バーを表示するためのライブラリ

# --- 設定 ---
GAME_ID = '0021500001'
TRACKING_FILE = f'data/2016.NBA.Raw.SportVU.Game.Logs/{GAME_ID}.json'
PBP_FILE = f'pbp_{GAME_ID}.csv'
OUTPUT_CSV = 'master_labeled_dataset_TIMEMATCH.csv'

# --- データの読み込み ---
try:
    pbp_df = pd.read_csv(PBP_FILE)
    with open(TRACKING_FILE, 'r') as f:
        tracking_data = json.load(f)
except FileNotFoundError as e:
    print(f"エラー: ファイルが見つかりません。{e}")
    exit()

# --- マスターデータセット作成処理 ---
master_data = []
all_events_in_tracking_file = tracking_data['events']

print(f"PBPデータを基準に、{len(all_events_in_tracking_file)}個のトラッキングデータとの時間照合を開始します...")

# ショットプレーだけをPBPデータから抽出 (EVENTMSGTYPE 1 or 2)
shot_pbp_events = pbp_df[pbp_df['EVENTMSGTYPE'].isin([1, 2])].copy()

# PBPデータのゲームクロックを秒に変換
def clock_to_seconds(time_str):
    if isinstance(time_str, str) and ':' in time_str:
        minutes, seconds = map(int, time_str.split(':'))
        return minutes * 60 + seconds
    return None

shot_pbp_events['GAME_CLOCK_SECONDS'] = shot_pbp_events['PCTIMESTRING'].apply(clock_to_seconds)

# トラッキングデータの各イベントをループ
for tracking_event in tqdm(all_events_in_tracking_file, desc="トラッキングデータを解析中"):
    if not tracking_event['moments']:
        continue
    
    # トラッキングデータの開始時間（秒）を取得
    tracking_start_clock = tracking_event['moments'][0][2]

    # 時間が最も近いショットプレーをPBPデータから探す (誤差1秒以内)
    closest_shot = shot_pbp_events.iloc[(shot_pbp_events['GAME_CLOCK_SECONDS'] - tracking_start_clock).abs().argsort()[:1]]

    if not closest_shot.empty:
        time_difference = abs(closest_shot.iloc[0]['GAME_CLOCK_SECONDS'] - tracking_start_clock)
        
        if time_difference < 1.0: # 誤差1秒未満なら、同じプレーと見なす
            pbp_event = closest_shot.iloc[0]
            event_msg_type = pbp_event['EVENTMSGTYPE']
            play_result_label = 1 if event_msg_type == 1 else 0

            play_description = ""
            if pd.notna(pbp_event['HOMEDESCRIPTION']):
                play_description = str(pbp_event['HOMEDESCRIPTION'])
            elif pd.notna(pbp_event['VISITORDESCRIPTION']):
                play_description = str(pbp_event['VISITORDESCRIPTION'])
            
            master_data.append({
                'game_id': GAME_ID,
                'event_id_PBP': int(pbp_event['EVENTNUM']),
                'tracking_start_clock': tracking_start_clock,
                'play_description': play_description,
                'play_result_label': play_result_label
            })

# --- 結果をCSVに保存 ---
if master_data:
    # 重複を除去
    master_df = pd.DataFrame(master_data).drop_duplicates(subset=['event_id_PBP'])
    master_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"\n✅ 処理完了！ {len(master_df)} 件のショットプレーを時間ベースで照合し、{OUTPUT_CSV} に保存しました。")
    print("\n--- 完成したデータセットの先頭5行 ---")
    print(master_df.head())
else:
    print("\n❌ 時間ベースで照合しても、対応するショットプレーが見つかりませんでした。")