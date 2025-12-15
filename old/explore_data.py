import json

file_path = 'data/2016.NBA.Raw.SportVU.Game.Logs/0021500001.json'

# 探したいキーの候補リスト
SEARCH_KEYS = ['description', 'desc_home', 'desc_away', 'pbp', 'play']

try:
    with open(file_path, 'r', encoding='utf-8') as f:
        game_data = json.load(f)

    if 'events' in game_data and game_data['events']:
        events_list = game_data['events']
        
        found_key = False
        # 全てのイベントをループしてチェック
        for i, event in enumerate(events_list):
            for key in SEARCH_KEYS:
                if key in event:
                    print(f"発見！ イベント番号 {i} にキー '{key}' が見つかりました。")
                    print(f"内容: {event[key]}")
                    found_key = True
                    break  # 内側のループを抜ける
            if found_key:
                break  # 外側のループも抜ける

        if not found_key:
            print(f"結論: {len(events_list)}個のイベントを全て調査しましたが、")
            print(f"候補となるキー {SEARCH_KEYS} は、どのイベントにも見つかりませんでした。")

    else:
        print("エラー: イベントデータが見つかりません。")

except Exception as e:
    print(f"エラーが発生しました: {e}")