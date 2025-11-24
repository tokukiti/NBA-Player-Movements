import json
import pandas as pd

# --- 設定 ---
GAME_ID = '0021500001'
TRACKING_JSON_FILE = f'data/2016.NBA.Raw.SportVU.Game.Logs/{GAME_ID}.json'

# --- データの読み込み ---
try:
    pbp_df = pd.read_csv(f'pbp_{GAME_ID}.csv')
    with open(TRACKING_JSON_FILE, 'r') as f:
        tracking_data = json.load(f)
except FileNotFoundError as e:
    print(f"エラー: ファイルが見つかりません。 {e}")
    exit()

# --- 調査開始 ---
print(f"--- ファイル: {TRACKING_JSON_FILE} の内容を調査します ---")

# 1. トラッキングデータから 'eventid' を取得 (現在は最初のイベントのみ調査)
tracking_event = tracking_data['events'][0]
event_id = tracking_event.get('eventid') or tracking_event.get('eventId')

if event_id is not None:
    print(f"\n[トラッキングデータ情報]")
    print(f"このファイル（の最初のイベント）に含まれる eventid は: {event_id}")

    # --- 修正点：データ型を揃える ---
    # 1. PBPデータのEVENTNUM列を、強制的に数値に変換する
    #    変換できない値があった場合は、エラーにせず無視(NaT)する
    pbp_df['EVENTNUM'] = pd.to_numeric(pbp_df['EVENTNUM'], errors='coerce')
    # 2. 比較するevent_idも、念のため数値に変換する
    event_id = int(event_id)
    # --- 修正ここまで ---

    # 2. 取得した eventid を使ってPBPデータから対応するプレーを検索
    pbp_match = pbp_df[pbp_df['EVENTNUM'] == event_id]

    if not pbp_match.empty:
        home_desc = pbp_match.iloc[0]['HOMEDESCRIPTION']
        visitor_desc = pbp_match.iloc[0]['VISITORDESCRIPTION']
        pbp_time = pbp_match.iloc[0]['PCTIMESTRING']

        print(f"\n[対応するプレイバイプレイデータ]")
        print(f"時間: {pbp_time}")
        print(f"内容(H): {home_desc}")
        print(f"内容(V): {visitor_desc}")
        print("\n✅ 照合成功！このJSONファイルの最初のイベントは、上記のプレーに対応しています。")
    else:
        print(f"\n❌ PBPデータに EVENTNUM {event_id} が見つかりませんでした。（データ型を揃えても失敗）")
else:
    print("\n❌ このJSONファイルには 'eventid' キーが含まれていませんでした。")