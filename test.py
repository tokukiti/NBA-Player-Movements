import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import numpy as np

# 設定
RAW_DATA_DIR = './data/2016.NBA.Raw.SportVU.Game.Logs'

def check_ball_trajectory(game_id, event_id):
    json_path = os.path.join(RAW_DATA_DIR, f"{game_id}.json")
    
    with open(json_path, 'r') as f:
        data = json.load(f)

    # 指定されたイベントを探す
    target_event = None
    for event in data['events']:
        eid = event.get('eventid') or event.get('eventId')
        if eid is not None and int(eid) == int(event_id):
            target_event = event
            break
    
    if target_event is None:
        print("Event not found.")
        return

    # ボール座標(X, Y, Z)と時間を抽出
    moments = target_event['moments']
    ball_z = []
    game_clock = []
    
    print(f"Total moments in this event: {len(moments)}")

    for moment in moments:
        # moment[5] は player/ball リスト。ボールは teamid = -1
        ball_data = next((item for item in moment[5] if item[0] == -1), None)
        if ball_data:
            # ball_data[2] starts x, y, z. So ball_data[4] is Z (radius/height) usually
            # SportVU format: team_id, player_id, x, y, z
            ball_z.append(ball_data[4]) 
            game_clock.append(moment[2]) # Game Clock

    if not ball_z:
        print("No ball data found in this event.")
        return

    # グラフ描画
    plt.figure(figsize=(10, 4))
    plt.plot(game_clock, ball_z, label='Ball Height (Z)')
    plt.axhline(y=10, color='r', linestyle='--', label='Rim Height (10ft)')
    plt.gca().invert_xaxis() # バスケの時間は減っていくので反転
    plt.title(f"Ball Trajectory: Game {game_id} Event {event_id}")
    plt.xlabel("Game Clock (sec)")
    plt.ylabel("Height (ft)")
    plt.legend()
    plt.grid(True)
    plt.show()

# 怪しいデータのIDを入れて実行してください
# 例：クォーター開始時刻のやつ
check_ball_trajectory('0021500044', 154) 
# 例：さっきの「入ってないのに入った判定」のやつ
check_ball_trajectory('0021500213', 129)