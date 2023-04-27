import numpy as np

from datetime import datetime
from keras.models import load_model
from pathlib import Path
from pv_mcts import boltzman, pv_mcts_scores
from shutil import copy
from tictactoe import State, popcount
from itertools import cycle

MAX_GAME_COUNT = 100  # AlphaZeroでは400。
MCTS_EVALUATE_COUNT = 20  # AlphaZeroでは1600。
TEMPERATURE = 1.0


def first_player_point(ended_state):
    if ended_state.lose:
        return 1 if (popcount(ended_state.pieces) + popcount(ended_state.enemy_pieces)) % 2 == 1 else 0

    return 0.5


def play(models):
    state = State()

    for model in cycle(models):
        if state.end:
            break;

        state = state.next(np.random.choice(state.legal_actions,
                                            p=boltzman(pv_mcts_scores(model, MCTS_EVALUATE_COUNT, state), TEMPERATURE)))

    return first_player_point(state)


def update_model():
    challenger_path = sorted(Path('model/candidates').glob('*.h5'))
    champion_paths = sorted(Path('./model').glob('*.h5'))

    if not challenger_path:
        print("No challenger model found. Please make sure it has at least one *.h5 file.")
        return

    challenger_path = challenger_path[-1]

    if not champion_paths:
        champion_path = Path('./model').joinpath(challenger_path.name)
        print(f"No champion model found. Creating a new one at {champion_path}")
    else:
        champion_path = champion_paths[-1].with_name(challenger_path.name)
        print(f"Replacing champion model at {champion_path}")

    copy(str(challenger_path), str(champion_path))


def main():
    model_paths = [Path('model/candidates'), Path('./model')]
    models = []

    for path in model_paths:
        sorted_h5_files = sorted(path.glob('*.h5'))
        if sorted_h5_files:
            last_h5_file = sorted_h5_files[-1]
            model = load_model(last_h5_file)
            models.append(model)

    models = tuple(models)
    total_point = 0

    for i in range(MAX_GAME_COUNT):
        if i % 2 == 0:
            total_point += play(models)
        else:
            total_point += 1 - play(tuple(reversed(models)))

        print('*** game {:03}/{:03} ended at {} ***'.format(i + 1, MAX_GAME_COUNT, datetime.now()))
        print(total_point / (i + 1))

    average_point = total_point / MAX_GAME_COUNT
    print(average_point)

    if average_point > 0.5:  # AlphaZeroでは0.55。マルバツだと最善同士で引き分けになるので、ちょっと下げてみました。
        update_model()
    else:
        print('Keep the champion.')


if __name__ == '__main__':
    main()
