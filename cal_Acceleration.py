import pandas as pd

# データフレームをロード (仮)
# df = pd.read_csv('your_tracking_data.csv') 

# 時間間隔を設定
dt = 0.04  # 1/25秒

# 選手ごと、イベントごとにソートしておくことが重要
df = df.sort_values(['event_id', 'player_id', 'frame'])

# 速度を計算
# groupbyで選手ごとに計算を区切る
df['vx'] = df.groupby(['event_id', 'player_id'])['x'].diff() / dt
df['vy'] = df.groupby(['event_id', 'player_id'])['y'].diff() / dt
# ボールの場合 (z座標がある)
if 'z' in df.columns:
    df['vz'] = df.groupby(['event_id', 'player_id'])['z'].diff() / dt

# 加速度を計算
df['ax'] = df.groupby(['event_id', 'player_id'])['vx'].diff() / dt
df['ay'] = df.groupby(['event_id', 'player_id'])['vy'].diff() / dt
if 'vz' in df.columns:
    df['az'] = df.groupby(['event_id', 'player_id'])['vz'].diff() / dt

print(df.head())