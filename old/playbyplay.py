from nba_api.stats.endpoints import playbyplayv2
import pandas as pd

# ★★★★★ おまじない：全ての列を省略せずに表示する ★★★★★
pd.set_option('display.max_columns', None)

# あなたのトラッキングデータセットの最初の試合ID
game_id = "0021500001" 

try:
    pbp = playbyplayv2.PlayByPlayV2(game_id)
    pbp_df = pbp.get_data_frames()[0]
    
    print(f"--- Game ID: {game_id} のプレイバイプレイデータ（詳細）---")
    
    # プレー内容が分かる重要な列だけを抜き出して表示
    # EVENTNUM: プレーの通し番号
    # PCTIMESTRING: クォーター内の残り時間
    # HOMEDESCRIPTION: ホームチーム側のプレー内容
    # VISITORDESCRIPTION: ビジターチーム側のプレー内容
    important_columns = ['EVENTNUM', 'PCTIMESTRING', 'HOMEDESCRIPTION', 'VISITORDESCRIPTION']
    print(pbp_df[important_columns].head(20)) # 最初の20プレーを表示

    # データをCSVファイルとして保存すると、Excelなどで全体をじっくり見れて便利です
    output_filename = f'pbp_{game_id}.csv'
    pbp_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
    print(f"\n全てのデータを {output_filename} に保存しました。")

except Exception as e:
    print(f"エラーが発生しました: {e}")