import pandas as pd
import numpy as np

# --- 設定 ---
INPUT_CSV = 'NBA_PBP_2015-16.csv'
OUTPUT_CSV = 'cleaned_shots_data.csv'

def run_pipeline():
    print(f"Loading raw CSV: {INPUT_CSV} ...")
    try:
        df = pd.read_csv(INPUT_CSV, encoding='ISO-8859-1')
    except FileNotFoundError:
        print(f"エラー: {INPUT_CSV} が見つかりません。")
        exit()

    print(f"Original rows: {len(df)}")

    # 1. URLからGAME_IDを生成 (これまでのロジック)
    unique_urls = df['URL'].unique()
    url_to_id = {url: f"002150{str(i + 1).zfill(4)}" for i, url in enumerate(unique_urls)}
    df['GAME_ID'] = df['URL'].map(url_to_id)

    # 2. 欠損している試合を除外
    missing_games = ['0021500006', '0021500008', '0021500014']
    df = df[~df['GAME_ID'].isin(missing_games)]

    # 3. 最初の20試合に絞る (テスト時間を短縮するため。全データやるならここをコメントアウト)
    # unique_game_ids = df['GAME_ID'].unique()
    # target_game_ids = unique_game_ids[:20] 
    # df = df[df['GAME_ID'].isin(target_game_ids)].copy()
    # print(f"Targeting {len(target_game_ids)} games.")

    # ==========================================
    # ★★★ ここが重要：厳密なフィルタリング ★★★
    # ==========================================
    
    # 4. ShotOutcome が 'make' か 'miss' の行だけを残す
    #    これにより、リバウンド、ファウル、交代、ジャンプボール等を全て削除
    if 'ShotOutcome' in df.columns:
        df['ShotOutcome'] = df['ShotOutcome'].astype(str).str.lower().str.strip()
        df = df[df['ShotOutcome'].isin(['make', 'miss'])]
    else:
        print("Error: 'ShotOutcome' column missing.")
        exit()

    # 5. Shooter(選手名)が入っていない行を削除
    if 'Shooter' in df.columns:
        df = df[df['Shooter'].notna()]
    
    # 6. 時間の整形 (欠損値を0埋めして整数化)
    df['SecLeft'] = pd.to_numeric(df['SecLeft'], errors='coerce').fillna(0).astype(int)
    df['Quarter'] = pd.to_numeric(df['Quarter'], errors='coerce').fillna(1).astype(int)

    # 7. 独自のID (EVENTNUM) を振り直す
    #    これがデータセット内での一意なIDになります
    df = df.reset_index(drop=True)
    df['EVENTNUM'] = df.index

    # 8. 必要な列だけ選んで保存
    columns_to_keep = [
        'GAME_ID', 'EVENTNUM', 'Quarter', 'SecLeft', 
        'AwayTeam', 'AwayPlay', 'HomeTeam', 'HomePlay', 
        'Shooter', 'ShotOutcome'
    ]
    # 存在しない列でエラーにならないようチェック
    existing_cols = [c for c in columns_to_keep if c in df.columns]
    final_df = df[existing_cols]

    print(f"Filtered rows (Valid Shots): {len(final_df)}")
    
    final_df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Cleaned data saved to: {OUTPUT_CSV}")
    print("このCSVをExcelで開いて、おかしな行が含まれていないか確認できます。")

if __name__ == "__main__":
    run_pipeline()