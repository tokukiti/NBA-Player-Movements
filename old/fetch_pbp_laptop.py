import requests
import pandas as pd
import time
import os
import random

# --- 対象の20試合 ---
GAME_IDS = [
    '0021500003', '0021500021', '0021500030', '0021500044', '0021500055',
    '0021500062', '0021500073', '0021500086', '0021500095', '0021500109',
    '0021500118', '0021500129', '0021500143', '0021500155', '0021500168',
    '0021500178', '0021500189', '0021500202', '0021500213', '0021500223'
]

# 保存先フォルダ
OUTPUT_DIR = 'pbp_cache_laptop'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def get_pbp_direct(game_id):
    # nba_apiを使わず、requestsで直接取得（ノートPCの環境構築を楽にするため）
    url = "https://stats.nba.com/stats/playbyplayv2"
    params = {
        'GameID': game_id,
        'StartPeriod': '0',
        'EndPeriod': '14',
    }
    # 偽装ヘッダー
    headers = {
        'Host': 'stats.nba.com',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Referer': 'https://www.nba.com/',
        'Origin': 'https://www.nba.com',
        'Connection': 'keep-alive',
        'x-nba-stats-origin': 'stats',
        'x-nba-stats-token': 'true'
    }

    try:
        response = requests.get(url, params=params, headers=headers, timeout=30)
        if response.status_code == 200:
            json_data = response.json()
            headers_list = json_data['resultSets'][0]['headers']
            row_set = json_data['resultSets'][0]['rowSet']
            return pd.DataFrame(row_set, columns=headers_list)
        else:
            print(f"  [Error] HTTP {response.status_code}")
            return None
    except Exception as e:
        print(f"  [Error] {e}")
        return None

def main():
    print(f"--- ノートPC: PBPダウンロード開始 (全{len(GAME_IDS)}試合) ---")
    
    for i, game_id in enumerate(GAME_IDS):
        print(f"[{i+1}/{len(GAME_IDS)}] {game_id} 取得中...", end="")
        
        df = get_pbp_direct(game_id)
        
        if df is not None and not df.empty:
            save_path = os.path.join(OUTPUT_DIR, f"pbp_{game_id}.csv")
            df.to_csv(save_path, index=False)
            print(" -> 成功 ✅")
            
            # 連続アクセス対策で少し待機
            time.sleep(random.uniform(2, 5))
        else:
            print(" -> 失敗 ❌")
            time.sleep(10)

    print(f"\n完了！ '{OUTPUT_DIR}' フォルダをデスクトップPCにコピーしてください。")

if __name__ == "__main__":
    main()