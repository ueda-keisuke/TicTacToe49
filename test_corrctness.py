import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KMP_AFFINITY'] = 'disabled'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from keras.models import load_model
from pathlib import Path
from pv_mcts import pv_mcts_next_action_fn
from tictactoe import State, popcount, random_next_action, monte_carlo_tree_search_next_action, nega_alpha_next_action


def silent_load_model(model_path):
    class ClearSession(tf.keras.callbacks.Callback):
        def on_train_begin(self, logs=None):
            tf.keras.backend.clear_session()

    return load_model(model_path, custom_objects={'ClearSession': ClearSession})


def main():
    def test_correctness(next_action):
        return ((next_action(State().next(0)) in (4,)) +
                (next_action(State().next(2)) in (4,)) +
                (next_action(State().next(6)) in (4,)) +
                (next_action(State().next(8)) in (4,)) +
                (next_action(State().next(4)) in (0, 2, 6, 8)) +
                (next_action(State().next(1)) in (0, 2, 4, 7)) +
                (next_action(State().next(3)) in (0, 4, 5, 6)) +
                (next_action(State().next(5)) in (2, 3, 4, 8)) +
                (next_action(State().next(7)) in (1, 4, 6, 8)) +
                (next_action(State().next(0).next(4).next(8)) in (1, 3, 5, 7)) +
                (next_action(State().next(2).next(4).next(6)) in (1, 3, 5, 7)))

    model_name = sorted(Path('./model').glob('*.h5'), reverse=True)[0]
    print(f"Using model {model_name}")

    pv_mcts_next_action = pv_mcts_next_action_fn(
        silent_load_model(model_name))

    nega_alpha_correctness = test_correctness(nega_alpha_next_action)
    print('{:4.1f}/11 = {:.2f} nega_alpha'.format(nega_alpha_correctness, nega_alpha_correctness / 11))

    pv_mcts_correctness = test_correctness(pv_mcts_next_action)
    print('{:4.1f}/11 = {:.2f} pv_mcts'.format(pv_mcts_correctness, pv_mcts_correctness / 11))

    monte_carlo_tree_search_correctness = sum([test_correctness(monte_carlo_tree_search_next_action) for _ in range(100)]) / 100
    print('{:4.1f}/11 = {:.2f} monte_carlo_tree_search'.format(monte_carlo_tree_search_correctness, monte_carlo_tree_search_correctness / 11))

    monte_carlo_tree_search_correctness = sum(
        [test_correctness(monte_carlo_tree_search_next_action) for _ in range(100)]) / 100
    print('{:4.1f}/11 = {:.2f} monte_carlo_tree_search'.format(monte_carlo_tree_search_correctness,
                                                               monte_carlo_tree_search_correctness / 11))


# main関数を呼び出す
main()


