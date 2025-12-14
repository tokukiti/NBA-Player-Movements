import pandas as pd
import json
import os
import argparse
from Event import Event
from Team import Team
from Constant import Constant
import matplotlib.pyplot as plt

# --- 設定 ---
RAW_DATA_DIR = './data/2016.NBA.Raw.SportVU.Game.Logs'
RESULTS_CSV = 'evaluation_results_v5.csv'

def find_event_index(game_data, target_event_id):
    """
    JSONデータの中から、指定された eventid を持つイベントのインデックス(順番)を探す
    """
    events = game_data['events']
    for i, event in enumerate(events):
        # JSON内のキーは 'eventid' または 'eventId' の場合がある
        eid = event.get('eventid') or event.get('eventId')
        if eid is not None and int(eid) == int(target_event_id):
            return i
    return None

def visualize_specific_play(game_id, event_id):
    """
    指定された試合・イベントIDのプレーを可視化してGIF保存する
    """
    json_path = os.path.join(RAW_DATA_DIR, f"{game_id}.json")
    
    # 1. JSONデータの読み込み
    print(f"Loading data from {json_path} ...")
    try:
        with open(json_path, 'r') as f:
            game_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found {json_path}")
        return

    # 2. イベントIDに対応するインデックスを検索
    event_index = find_event_index(game_data, event_id)
    
    if event_index is None:
        print(f"Error: Event ID {event_id} not found in game {game_id}")
        return

    print(f"Found Event ID {event_id} at index {event_index}")

    # 3. Eventクラスを使って可視化 (既存のクラスを再利用)
    event_data = game_data['events'][event_index]
    event = Event(event_data)
    
    # ホーム・アウェイチーム情報の設定（表示用）
    home_team_id = event_data['home']['teamid']
    visitor_team_id = event_data['visitor']['teamid']
    # Eventクラス内部でTeam情報が必要な場合があるため補完
    # (既存のEvent.pyの実装によっては、ここが自動処理される場合もあります)

    print(f"Generating animation for Game: {game_id}, Event: {event_id}...")
    
    # 4. 表示と保存
    # Event.py の show() メソッドを呼び出す
    # 注意: Event.py の show() は 'animation.gif' という固定名で保存する設定になっているので
    # 実行後にリネームするか、Event.py を少し改造してファイル名を引数で渡せるようにするとベストです。
    # ここでは簡易的に実行します。
    
    try:
# 1. ここでファイル名を確定させる
        # outputフォルダに入れると整理しやすいのでおすすめです
        if not os.path.exists('output'):
            os.makedirs('output')
            
        output_filename = f"output/Game_{game_id}_Event_{event_id}.gif"
        
        print(f"Generating animation for Game: {game_id}, Event: {event_id}...")
        
        # 2. ファイル名を渡して実行！
        event.show(save_path=output_filename)
        
        print(f"✅ Animation saved successfully as: {output_filename}")
            
    except Exception as e:
        print(f"Error during visualization: {e}")

if __name__ == "__main__":
    # 引数で直接IDを指定して実行できるようにする
    parser = argparse.ArgumentParser(description='Visualize a specific play from analysis results.')
    parser.add_argument('--game_id', type=str, help='Game ID (e.g., 0021500001)')
    parser.add_argument('--event_id', type=int, help='Event ID (e.g., 45)')
    
    args = parser.parse_args()

    if args.game_id and args.event_id:
        # コマンドラインから指定された場合
        visualize_specific_play(args.game_id, args.event_id)
    else:
        # 引数がない場合は、CSVから「面白い」データを自動でピックアップする例
        print("Arguments not provided. Reading top uncertain play from CSV...")
        try:
            df = pd.read_csv(RESULTS_CSV)
            
            # 例1: AIが「失敗」と予測したが「成功」だった（見逃し）ケース
            # false_negatives = df[(df['predicted_label'] == 0) & (df['actual_label'] == 1)]
            
            # 例2: AIが最も「迷った」ケース (確率が0.5に近い)
            df['uncertainty'] = abs(df['probability_success'] - 0.5)
            most_uncertain = df.sort_values('uncertainty').iloc[0]
            
            gid = str(most_uncertain['game_id']).zfill(10) # 0埋め
            eid = int(most_uncertain['event_id'])
            
            print(f"Most uncertain play found: Game {gid}, Event {eid}, Prob: {most_uncertain['probability_success']:.4f}")
            visualize_specific_play(gid, eid)
            
        except FileNotFoundError:
            print(f"CSV file {RESULTS_CSV} not found.")