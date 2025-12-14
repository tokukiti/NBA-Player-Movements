from Constant import Constant
from Moment import Moment
from Team import Team
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Circle, Rectangle, Arc
import datetime  # 日時取得用に追加

class Event:
    """A class for handling and showing events"""

    def __init__(self, event):
        moments = event['moments']
        self.moments = [Moment(moment) for moment in moments]
        home_players = event['home']['players']
        guest_players = event['visitor']['players']
        players = home_players + guest_players
        player_ids = [player['playerid'] for player in players]
        player_names = [" ".join([player['firstname'],
                                 player['lastname']]) for player in players]
        player_jerseys = [player['jersey'] for player in players]
        values = list(zip(player_names, player_jerseys))
        # Example: 101108: ['Chris Paul', '3']
        self.player_ids_dict = dict(zip(player_ids, values))

    def update_radius(self, i, player_circles, ball_circle, annotations, clock_info):
        moment = self.moments[i]
        
        # プレイヤーの位置更新
        for j, circle in enumerate(player_circles):
            try:
                # プレイヤーデータの数が合わない場合の安全策
                if j < len(moment.players):
                    circle.center = moment.players[j].x, moment.players[j].y
                    annotations[j].set_position(circle.center)
            except IndexError:
                pass

        # --- 時計表示の修正箇所 (None回避) ---
        # ゲームクロックのフォーマット
        if moment.game_clock is None:
            game_clock_str = "--:--"
        else:
            minutes = int(moment.game_clock) % 3600 // 60
            seconds = int(moment.game_clock) % 60
            game_clock_str = "{:02d}:{:02d}".format(minutes, seconds)

        # ショットクロックのフォーマット (ここがエラーの原因でした)
        if moment.shot_clock is None:
            shot_clock_str = "--.-"
        else:
            shot_clock_str = "{:03.1f}".format(moment.shot_clock)

        # テキストの設定
        clock_text = 'Quarter {:d}\n {}\n {}'.format(
            moment.quarter,
            game_clock_str,
            shot_clock_str)
        
        clock_info.set_text(clock_text)
        # ---------------------------------------

        # ボールの位置更新
        ball_circle.center = moment.ball.x, moment.ball.y
        ball_circle.radius = moment.ball.radius / Constant.NORMALIZATION_COEF
        
        return player_circles, ball_circle

    def show(self, save_path=None):
        # Leave some space for inbound passes
        ax = plt.axes(xlim=(Constant.X_MIN,
                            Constant.X_MAX),
                      ylim=(Constant.Y_MIN,
                            Constant.Y_MAX))
        ax.axis('off')
        fig = plt.gcf()
        ax.grid(False)  # Remove grid
        start_moment = self.moments[0]
        player_dict = self.player_ids_dict

        clock_info = ax.annotate('', xy=[Constant.X_CENTER, Constant.Y_CENTER],
                                 color='black', horizontalalignment='center',
                                   verticalalignment='center')

        annotations = [ax.annotate(self.player_ids_dict[player.id][1], xy=[0, 0], color='w',
                                   horizontalalignment='center',
                                   verticalalignment='center', fontweight='bold')
                       for player in start_moment.players]

        # Prepare table
        # チームIDでソートしてホーム/アウェイを分ける
        sorted_players = sorted(start_moment.players, key=lambda player: player.team.id)
        
        # 少なくとも各チーム1人以上いると仮定
        if len(sorted_players) >= 6: # 通常は10人いるはず
            home_player = sorted_players[0]
            guest_player = sorted_players[5] # 後半のグループの最初
        else:
            # 異常系：とりあえず先頭を使う
            home_player = sorted_players[0]
            guest_player = sorted_players[-1]

        column_labels = tuple([home_player.team.name, guest_player.team.name])
        column_colours = tuple([home_player.team.color, guest_player.team.color])
        cell_colours = [column_colours for _ in range(5)]
        
        home_players = [' #'.join([player_dict[player.id][0], player_dict[player.id][1]]) for player in sorted_players[:5]]
        guest_players = [' #'.join([player_dict[player.id][0], player_dict[player.id][1]]) for player in sorted_players[5:]]
        players_data = list(zip(home_players, guest_players))

        table = plt.table(cellText=players_data,
                              colLabels=column_labels,
                              colColours=column_colours,
                              colWidths=[Constant.COL_WIDTH, Constant.COL_WIDTH],
                              loc='bottom',
                              cellColours=cell_colours,
                              fontsize=Constant.FONTSIZE,
                              cellLoc='center')
        table.scale(1, Constant.SCALE)
        table_cells = table.get_celld().values()
        for cell in table_cells:
            cell._text.set_color('white')

        player_circles = [plt.Circle((0, 0), Constant.PLAYER_CIRCLE_SIZE, color=player.color)
                          for player in start_moment.players]
        ball_circle = plt.Circle((0, 0), Constant.PLAYER_CIRCLE_SIZE,
                                   color=start_moment.ball.color)
        for circle in player_circles:
            ax.add_patch(circle)
        ax.add_patch(ball_circle)

        anim = animation.FuncAnimation(
             fig, self.update_radius,
             fargs=(player_circles, ball_circle, annotations, clock_info),
             frames=len(self.moments), interval=Constant.INTERVAL)
        
        # コート画像の読み込み (ファイルパスに注意)
        try:
            court = plt.imread("court.png")
            plt.imshow(court, zorder=0, extent=[Constant.X_MIN, Constant.X_MAX - Constant.DIFF,
                                                Constant.Y_MAX, Constant.Y_MIN])
        except FileNotFoundError:
            print("Warning: 'court.png' not found. Background will be empty.")


        # --- 保存名の決定ロジック ---
        if save_path:
            output_file = save_path
        else:
            # 指定がなければ現在時刻を使う
            now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"play_{now}.gif"
        
        print(f"GIFアニメーションを作成中... ({output_file})")
        print("※時間がかかります。完了するまで閉じないでください...")
        
        # pillowを使ってGIFとして保存
        try:
            anim.save(output_file, writer='pillow', fps=25)
            print(f"保存完了！ファイル名: {output_file}")
        except Exception as e:
            print(f"保存中にエラーが発生しました: {e}")

        # GUIで表示したい場合はコメントアウトを外す（保存のみなら不要）
        # plt.show() 
        plt.close() # メモリ解放のため閉じる