import json

# --- 設定項目 ---
file_path = 'data/2016.NBA.Raw.SportVU.Game.Logs/0021500001.json'


# --- データ読み込みと抽出処理 ---
try:
    with open(file_path, 'r', encoding='utf-8') as f:
        game_data = json.load(f)

    # 'events'キーが存在するかチェック
    if 'events' in game_data and game_data['events']:
        events_list = game_data['events']
        first_event_data = events_list[0]
        
        # --- 結果の確認 ---
        # 成功メッセージは、全てのデータアクセスが成功した後に表示するように移動
        
        # 正しいキー名でデータを取得
        # 'eventId' -> 'eventid'
        # 'home'/'visitor'の中の'description' -> 'desc_home'/'desc_away'
        event_id = first_event_data.get('eventid', 'N/A') # .get()を使うとキーが無くてもエラーにならない
        home_description = first_event_data.get('desc_home', 'N/A')
        away_description = first_event_data.get('desc_away', 'N/A')

        print("最初のイベントデータの抽出と解析に成功しました！")
        print(f"イベントID: {event_id}")
        print(f"ホーム側実況: {home_description}")
        print(f"アウェイ側実況: {away_description}")
        
        # 抽出したデータの全体像を綺麗に表示したい場合は、以下のコメントを外してください
        # print("\n--- 最初のイベントデータの全体像 ---")
        # print(json.dumps(first_event_data, indent=2))
        
    else:
        print("エラー: この試合ファイルにはイベントデータが含まれていません。")

except FileNotFoundError:
    print(f"エラー: ファイルが見つかりません。パスを確認してください: {file_path}")
except json.JSONDecodeError:
    print(f"エラー: JSONファイルの形式が正しくありません: {file_path}")
except Exception as e:
    print(f"予期せぬエラーが発生しました: {e}")