import copy
from typing import List, Tuple
from player import Player
from random import choices
import yaml
import numpy as np
from numpy.random import default_rng

rng = default_rng()


class Evolution:
    def __init__(self):
        self.game_mode = "Neuroevolution"

    def next_population_selection(self, players: List[Player], num_players):
        """
        Gets list of previous and current players (μ + λ) and returns num_players number of players based on their
        fitness value.

        :param players: list of players in the previous generation
        :param num_players: number of players that we return
        """
        # TODO (Implement top-k algorithm here)
        players.sort(key=lambda player: player.fitness, reverse=True)
        fitnesses = [player.fitness for player in players]

        # TODO (Additional: Implement roulette wheel here)
        self.roulette_wheel(players, num_players)

        # TODO (Additional: Implement SUS here)
        self.sus(players, num_players)

        # TODO (Additional: Learning curve)
        self.update_metric_points(fitnesses)

        return players[: num_players]

    def generate_new_population(self, num_players, prev_players=None):
        """
        Gets survivors and returns a list containing num_players number of children.

        :param num_players: Length of returning list
        :param prev_players: List of survivors
        :return: A list of children
        """
        first_generation = prev_players is None
        if first_generation:
            return [Player(self.game_mode) for _ in range(num_players)]
        else:
            # TODO ( Parent selection and child generation )
            # Defualt
            parents = prev_players

            # Parent Selection with Q Tournament
            parents = self.q_tournament(prev_players, num_players)

            # Parent Selection with Roulette Wheel
            parents = self.roulette_wheel(prev_players, num_players)

            # Parent Selection with SUS
            parents = self.sus(prev_players, num_players)

            parent_pairs = [parents[i:i + 2] for i in range(0, len(parents), 2)]
            childs = []
            for parent_1, parent_2 in parent_pairs:
                if parent_1 is None or parent_2 is None:
                    continue

                child_1, child_2 = self.crossover(parent_1, parent_2)

                child_1 = self.mutate(child_1)
                child_2 = self.mutate(child_2)

                childs.extend(
                    [child_1, child_2]
                )

            childs = prev_players  # TODO DELETE THIS AFTER YOUR IMPLEMENTATION
            return childs

    def clone_player(self, player) -> Player:
        """
        Gets a player as an input and produces a clone of that player.
        """
        new_player = Player(self.game_mode)
        new_player.nn = copy.deepcopy(player.nn)
        new_player.fitness = player.fitness
        return new_player

    def update_metric_points(self, fitnesses):
        fitnesses.sort(reverse=True)

        new_metric_points = {
            'max': fitnesses[0],
            'avg': sum(fitnesses) / len(fitnesses),
            'min': fitnesses[-1]
        }
        with open('points.yml', 'r') as f:
            metric_points = yaml.safe_load(f)
            for metric, points in metric_points.keys():
                metric_points[metric].append(
                    new_metric_points.get(metric)
                )
            yaml.safe_dump(metric_points, f)  # Also note the safe_dump

    @staticmethod
    def sorted_selection(players, num_players):
        players.sort(key=lambda player: player.fitness, reverse=True)
        return players[:num_players]

    @staticmethod
    def q_tournament(players, num_players, q=5):
        winners = []
        for tournament_i in range(num_players):
            tournament_players = choices(players, q)
            tournament_players.sort(key=lambda player: player.fitness, reverse=True)
            winner = tournament_players[0]

            winners.append(winner)

        return winners

    @staticmethod
    def roulette_wheel(players, num_players):
        fitnesses = [player.fitness for player in players]

        sum_fitnesses = sum(fitnesses)
        selection_probs = [player.fitness / sum_fitnesses for player in players]
        return players[rng.choice(num_players, p=selection_probs)]

    @staticmethod
    def sus(players, num_players):
        fitnesses = np.array([player.fitness for player in players])

        fitness_cumsum = fitnesses.cumsum()
        fitness_sum = fitness_cumsum[-1]  # the "roulette wheel"
        step = fitness_sum / num_players  # we'll move by this amount in the wheel
        start = rng.random() * step  # sample a start point in [0, step)
        # get N evenly-spaced points in the wheel
        selectors = np.arange(start, fitness_sum, step)
        selected = np.searchsorted(fitness_cumsum, selectors)
        return selected

    def crossover(self, parent_1: Player, parent_2: Player) -> Tuple[Player, Player]:
        layers = parent_1.nn.layer_size

        child_1 = self.clone_player(parent_1)
        child_2 = self.clone_player(parent_2)

        for layer_number, layer in enumerate(layers):
            cut_off = layer // 2

            child_1.nn.weights[layer_number][:cut_off] = parent_1.nn.weights[layer_number][:cut_off]
            child_1.nn.weights[layer_number][cut_off:] = parent_2.nn.weights[layer_number][cut_off:]

            child_2.nn.weights[layer_number][:cut_off] = parent_2.nn.weights[layer_number][:cut_off]
            child_2.nn.weights[layer_number][cut_off:] = parent_1.nn.weights[layer_number][cut_off:]

            child_1.nn.biases[layer_number][:cut_off] = parent_1.nn.weights[layer_number][:cut_off]
            child_1.nn.biases[layer_number][cut_off:] = parent_2.nn.weights[layer_number][cut_off:]

            child_2.nn.biases[layer_number][:cut_off] = parent_2.nn.biases[layer_number][:cut_off]
            child_2.nn.biases[layer_number][cut_off:] = parent_1.nn.biases[layer_number][cut_off:]

        return child_1, child_2

    def mutate(self, child: Player):
        mutation_prob = 0.3

        mutated_child = self.clone_player(child)

        for index, weight in enumerate(child.nn.weights):
            if np.random.uniform(0, 1, 1)[0] < mutation_prob:
                mutated_child.nn.weights[index] = weight + np.random.randn(*weight.shape)

        for index, bias in enumerate(child.nn.biases):
            if np.random.uniform(0, 1, 1)[0] < mutation_prob:
                mutated_child.nn.biases[index] = bias + np.random.randn(*bias.shape)

        return mutated_child
