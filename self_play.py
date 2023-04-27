# 必要なライブラリをインポート
import numpy as np
import pickle
from datetime import datetime
from keras.models import load_model
from pathlib import Path
from pv_mcts import boltzman, pv_mcts_scores
from tictactoe import State, popcount

# ゲームの最大数とMCTSの評価回数を設定
MAX_GAME_COUNT = 500  # AlphaZeroでは25000。
MCTS_EVALUATE_COUNT = 20  # AlphaZeroでは1600。
TEMPERATURE = 1.0


# 引き分け以外の場合の結果を返す関数
def first_player_value(ended_state):
    if ended_state.lose:
        return 1 if (popcount(ended_state.pieces) + popcount(ended_state.enemy_pieces)) % 2 == 1 else -1
    return 0


# ゲームをプレイし、データを生成する関数
def play(model):
    states = []
    ys = [[], None]

    state = State()

    while True:
        if state.end:
            break

        scores = pv_mcts_scores(model, MCTS_EVALUATE_COUNT, state)

        policies = [0] * 9
        for action, policy in zip(state.legal_actions, boltzman(scores, 1.0)):
            policies[action] = policy

        states.append(state)
        ys[0].append(policies)

        # The model conducts search using a method called MCTS (Monte Carlo Tree Search)
        # and selects legal moves based on the results.
        # The next line does not simply choose a legal move randomly,
        # but selects a move based on the probabilities determined by the Boltzmann distribution.
        state = state.next(np.random.choice(state.legal_actions, p=boltzman(scores, TEMPERATURE)))

    value = first_player_value(state)
    ys[1] = tuple(value if i % 2 == 0 else -value for i in range(len(ys[0])))

    return states, ys


# 学習データをファイルに保存する関数
def write_data(states, ys, game_count):
    y_policies, y_values = ys
    now = datetime.now()

    for i in range(len(states)):
        with open('./data/{:04}-{:02}-{:02}-{:02}-{:02}-{:02}-{:04}-{:02}.pickle'.format(now.year, now.month, now.day,
                                                                                         now.hour, now.minute,
                                                                                         now.second, game_count, i),
                  mode='wb') as f:
            pickle.dump((states[i], y_policies[i], y_values[i]), f)


# メイン関数
def main():
    # 最新のモデルをロード
    model = load_model(sorted(Path('model/candidates').glob('*.h5'))[-1])

    # ゲームを繰り返しプレイし、学習データを生成
    for i in range(MAX_GAME_COUNT):
        states, ys = play(model)

        print('*** game {:03}/{:03} ended at {} ***'.format(i + 1, MAX_GAME_COUNT, datetime.now()))
        print(states[-1])

        write_data(states, ys, i)


# スクリプトが直接実行された場合に、main関数を呼び出す
if __name__ == '__main__':
    main()
