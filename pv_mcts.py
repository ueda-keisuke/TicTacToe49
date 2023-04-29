from math import sqrt
from operator import attrgetter
import numpy as np
from funcy import mapcat


def pv_mcts_scores(model, evaluate_count, state):
    class node:
        def __init__(self, state, p):
            self.state = state
            self.p = p  # 方策
            self.w = 0  # 価値
            self.n = 0  # 試行回数
            self.child_nodes = None  # 子ノード

        def evaluate(self):
            if self.state.end:
                value = -1 if self.state.lose else 0

                self.w += value
                self.n += 1

                return value

            if not self.child_nodes:
                policies, value = predict(model, self.state)

                self.w += value
                self.n += 1

                self.child_nodes = tuple(
                    node(self.state.next(action), policy) for action, policy in zip(self.state.legal_actions, policies))

                return value
            else:
                value = -self.next_child_node().evaluate()

                self.w += value
                self.n += 1

                return value

        def next_child_node(self):
            def pucb_values():
                t = sum(map(attrgetter('n'), self.child_nodes))

                # child_node.nが0の場合に大きな値を設定してしまうと結局全部の手を試してみることになってしまうので、-1から1の中央の0を設定しました。
                C_PUCT = 1.0
                return tuple(
                    (-child_node.w / child_node.n if child_node.n else 0.0) + C_PUCT * child_node.p * sqrt(t) / (
                            1 + child_node.n) for child_node in self.child_nodes)

            return self.child_nodes[np.argmax(pucb_values())]

    root_node = node(state, 0)

    for _ in range(evaluate_count):
        root_node.evaluate()

    return tuple(map(attrgetter('n'), root_node.child_nodes))


def boltzman(xs, temperature):
    xs = [x ** (1 / temperature) for x in xs]

    return [x / sum(xs) for x in xs]


def pv_mcts_next_action_fn(model):
    def pv_mcts_next_action(state):
        return state.legal_actions[np.argmax(pv_mcts_scores(model, 20, state))]

    return pv_mcts_next_action


def to_x(state):
    def pieces_to_x(pieces):
        return (1.0 if pieces & 0b100000000 >> i else 0.0 for i in range(9))

    return np.array(tuple(mapcat(pieces_to_x, (state.pieces, state.enemy_pieces)))).reshape(2, 3, 3).transpose(1, 2, 0)


def predict(model, state):
    x = to_x(state).reshape(1, 3, 3, 2)
    y = model.predict(x, batch_size=1, verpose=0)

    policies = y[0][0][list(state.legal_actions)]
    policies /= sum(policies) if sum(policies) else 1

    return policies, y[1][0][0]
