import json
import pandas as pd

# --- 診断対象の試合ID ---
GAME_ID = '0021500002' # 2試合目を調査する

# --- 必要なファイルのパス ---
PBP_FILE = f'pbp_{GAME_ID}.csv'
TRACKING_FILE = f'data/2016.NBA.Raw.SportVU.Game.Logs/{GAME_ID}.json'

print(f"--- 試合ID: {GAME_ID} のデータ不整合を診断します ---")

# --- データの読み込み ---
try:
    pbp_df = pd.read_csv(PBP_FILE)
    with open(TRACKING_FILE, 'r') as f:
        tracking_data = json.load(f)
except FileNotFoundError as e:
    print(f"エラー: 診断に必要なファイルが見つかりません。{e}")
    exit()

# --- 診断1: PBPデータのショットプレー時間を確認 ---
print("\n【診断1】PBPデータに含まれるショットプレーの時間（先頭10件）")
shot_pbp_events = pbp_df[pbp_df['EVENTMSGTYPE'].isin([1, 2])].copy()

def clock_to_seconds(time_str):
    if isinstance(time_str, str) and ':' in time_str:
        m, s = map(int, time_str.split(':'))
        return m * 60 + s
    return None

shot_pbp_events['GAME_CLOCK_SECONDS'] = shot_pbp_events['PCTIMESTRING'].apply(clock_to_seconds)
print(shot_pbp_events[['EVENTNUM', 'PCTIMESTRING', 'GAME_CLOCK_SECONDS']].head(10))


# --- 診断2: トラッキングデータのイベント開始時間を確認 ---
print("\n【診断2】トラッキングデータに含まれるイベントの開始時間（先頭10件）")
all_tracking_events = tracking_data.get('events', [])
if all_tracking_events:
    for i, event in enumerate(all_tracking_events[:10]):
        if event.get('moments'):
            start_clock = event['moments'][0][2]
            print(f"  イベント {i}: 開始時間 = {start_clock:.2f} 秒")
        else:
            print(f"  イベント {i}: momentsデータがありません。")
else:
    print("  トラッキングデータに 'events' が見つかりません。")

# --- 診断3: 実際に照合を試みてみる ---
print("\n【診断3】この試合で、時間ベースの照合が成功するか試行")
matched_count = 0
for _, pbp_event in shot_pbp_events.iterrows():
    pbp_time = pbp_event['GAME_CLOCK_SECONDS']
    if pbp_time is None: continue

    for tracking_event in all_tracking_events:
        if tracking_event.get('moments'):
            tracking_start_clock = tracking_event['moments'][0][2]
            if abs(tracking_start_clock - pbp_time) < 1.0:
                matched_count += 1
                break # 1つ見つかったら次のPBPイベントへ

print(f"\n--- 診断完了 ---")
if matched_count > 0:
    print(f"✅ 発見！ この試合では {matched_count} 件のプレーが照合に成功しました。")
    print("pipeline.pyのロジックは正しいですが、他の要因でデータが生成されなかった可能性があります。")
else:
    print(f"❌ 根本原因特定。この試合では、時間ベースで照合できるプレーが1つも見つかりませんでした。")
    print("診断1と診断2の時間を比較し、なぜ一致しないのか（例：時間の起点が違う、フォーマットが違うなど）を調査する必要があります。")