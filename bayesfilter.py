# Complete this class for all parts of the project

import numpy as np
import math
from pacman_module import util
from pacman_module.game import Agent


def normal_pdf(x, mean, standard_deviation):
    """
    Normal Probability Distribution Function - Computes the normal probability of x

    Arguments:
    ----------
    - 'x': the variable of the normal propability distribution function
    - 'mean': the mean of the normal distribution
    - 'standard_deviation': the standard deviation of the normal distribution

    Return:
    -------
    - The normal probability of x (in range [0, 1])
    """
    return 1 / math.sqrt(2 * math.pi * (standard_deviation ** 2)) * \
           math.exp(-((x - mean) ** 2) / (2 * (standard_deviation ** 2)))


class BeliefStateAgent(Agent):
    def __init__(self, args):
        """
        Arguments:
        ----------
        - `args`: Namespace of arguments from command-line prompt.
        """
        self.args = args
        """
            Variables to use in 'update_belief_state' method.
            Initialization occurs in 'get_action' method.
        """
        # Current list of belief states over ghost positions
        self.beliefGhostStates = None

        # Grid of walls (assigned with 'state.getWalls()' method)
        self.walls = None

        # Hyper-parameters
        self.ghost_type = self.args.ghostagent
        self.sensor_variance = self.args.sensorvariance

    def update_belief_state(self, evidences, pacman_position):
        """
        Given a list of (noised) distances from pacman to ghosts,
        returns a list of belief states about ghosts positions

        Arguments:
        ----------
        - `evidences`: list of distances between
          pacman and ghosts at state x_{t}
          where 't' is the current time step
        - `pacman_position`: 2D coordinates position
          of pacman at state x_{t}
          where 't' is the current time step

        Return:
        -------
        - A list of Z belief states at state x_{t}
          as N*M numpy mass probability matrices
          where N and M are respectively width and height
          of the maze layout and Z is the number of ghosts.

        N.B. : [0,0] is the bottom left corner of the maze
        """
        
        belief_states = self.beliefGhostStates

        # XXX: Your code here

        nb_ghosts = len(evidences)
        width = len(belief_states[0])
        height = len(belief_states[0][0])
        nb_cells = width * height

        # --------------------------------------------------------- #
        # Computation of the transition matrix P(X_{t + 1} ∣ X_{t}) #
        # --------------------------------------------------------- #

        transition_matrix = np.zeros((nb_ghosts, nb_cells, nb_cells))
        # one dimension for each ghost,
        # one for each previous position of the ghost
        # one for each new position of the ghost

        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # the 4 possible directions for a ghost to move

        # for each ghost
        for g in range(nb_ghosts):

            # for each cell of the maze, i.e. for each position
            for k in range(width):  # x_{t}, the previous position of the ghost
                for l in range(height):

                    sum_row = 0
                    # for each direction of the 2D space
                    for direction in directions:  # X_{t+1}, the new position of the ghost
                        i = k + direction[0]
                        j = l + direction[1]

                        # if the ghost makes a move from and to a legal position,
                        # i.e.: if it moves from and to an existing square with no wall
                        if i >= 0 and j >= 0 and i < width and j < height \
                                and self.walls[i][j] == False and self.walls[k][l] == False:

                            power = 0
                            # if the ghost moves away from pacman
                            if util.manhattanDistance((i, j), pacman_position) >= \
                                    util.manhattanDistance((k, l), pacman_position):

                                # if self.ghost_type == "confused": power = 0
                                if self.ghost_type == "scared":
                                    power = 3
                                elif self.ghost_type == "afraid":
                                    power = 1

                            transition_matrix[g][l + k * height][j + i * height] += 2 ** power

                            sum_row += transition_matrix[g][l + k * height][j + i * height]

                    if sum_row != 0:
                        transition_matrix[g][l + k * height] /= sum_row  # normalize the last completed row of
                        # the transition matrix

        # ------------------------------------------------------- #
        # Computation of the matrix: P(E_{t+1} = e_{t+1} ∣ X_{t}) #
        # ------------------------------------------------------- #

        observation_matrix = np.zeros((nb_ghosts, nb_cells, nb_cells))
        # one dimension for each ghost,
        # one for each new position of the ghost
        # one again for each new position of the ghost
        # (2D diagonal sub-matrix)

        # for each ghost
        for g in range(nb_ghosts):

            # for each cell of the maze, i.e. for each position
            for i in range(width):  # X_{t+1}, the new position of the ghost
                for j in range(height):

                    if not self.walls[i][j]:
                        pacman_ghost_dist = util.manhattanDistance((i, j), pacman_position)
                        observation_matrix[g][j + i * height][j + i * height] = \
                            normal_pdf(pacman_ghost_dist, evidences[g], math.sqrt(self.sensor_variance))

        # --------------------------------------------------------------- #
        # Computation of the forward matrix: P(X_{t + 1} ∣ e_{1 : t + 1}) #
        # --------------------------------------------------------------- #

        forward = np.reshape(np.asarray(belief_states), (nb_ghosts, width * height))
        # one dimension for each ghost,
        # one for each new position of
        # the ghost

        # for each ghost
        for g in range(nb_ghosts):

            forward[g] = np.matmul(np.transpose(transition_matrix[g]), forward[g])
            forward[g] = np.matmul(observation_matrix[g], forward[g])
            sum_row = np.sum(forward[g])
            if sum_row != 0:
                forward[g] /= sum_row  # normalize the forward vector

        belief_states = np.reshape(forward, (nb_ghosts, width, height))  # forward matrix reshaped as a
        # (nb_ghosts * width * height) matrix

        # XXX: End of your code

        self.beliefGhostStates = belief_states

        return belief_states

    def _get_evidence(self, state):
        """
        Computes noisy distances between pacman and ghosts.

        Arguments:
        ----------
        - `state`: The current game state s_t
                   where 't' is the current time step.
                   See FAQ and class `pacman.GameState`.


        Return:
        -------
        - A list of Z noised distances in real numbers
          where Z is the number of ghosts.

        XXX: DO NOT MODIFY THIS FUNCTION !!!
        Doing so will result in a 0 grade.
        """
        positions = state.getGhostPositions()
        pacman_position = state.getPacmanPosition()
        noisy_distances = []

        for p in positions:
            true_distance = util.manhattanDistance(p, pacman_position)
            noisy_distances.append(
                np.random.normal(loc=true_distance,
                                 scale=np.sqrt(self.sensor_variance)))

        return noisy_distances

    def _record_metrics(self, belief_states, state):
        """
        Use this function to record your metrics
        related to true and belief states.
        Won't be part of specification grading.

        Arguments:
        ----------
        - `state`: The current game state s_t
                   where 't' is the current time step.
                   See FAQ and class `pacman.GameState`.
        - `belief_states`: A list of Z
           N*M numpy matrices of probabilities
           where N and M are respectively width and height
           of the maze layout and Z is the number of ghosts.

        N.B. : [0,0] is the bottom left corner of the maze
        """

        nb_ghosts = len(belief_states)
        width = len(belief_states[0])
        height = len(belief_states[0][0])
        nb_cells = width * height

        # --------------------------------------- #
        # Uncertainty Measure of the Belief State #
        # --------------------------------------- #

        threshold = 1e-7
        uncertainty = 0

        # for each ghost
        for g in range(nb_ghosts):

            # for each cell of the maze, i.e. for each position
            for i in range(width):
                for j in range(height):
                    if belief_states[g][i][j] > threshold:

                        uncertainty += (1 - belief_states[g][i][j])
                        # Probability of (i, j) not being the position of the ghost (according to the Bayes filter)

        uncertainty /= nb_cells * nb_ghosts

        print("Uncertainty: ", uncertainty)
        # 0 = no uncertainty, only one possible position for each ghost
        # 1 = no clue on the ghosts' position

        # --------------------------- #
        # Quality of the Belief State #
        # --------------------------- #

        max_manhattan_distance = width + height - 2
        # Max manhattan distance between a ghost and an estimation of its position
        quality = 0

        # for each ghost
        for g in range(nb_ghosts):
            ghost_position = state.getGhostPosition(g + 1)

            # for each cell of the maze, i.e. for each position
            for i in range(width):
                for j in range(height):
                    quality += util.manhattanDistance((i, j), ghost_position) * belief_states[g][i][j]
                    # Compute the mean manhattan distance between the ghost and each estimation of its position
                    # (each has as weight the probability of the estimation)

        quality = 1 - quality / max_manhattan_distance / nb_ghosts

        print("Quality: ", quality) # 0 = worst quality; 1 = best quality

    def get_action(self, state):
        """
        Given a pacman game state, returns a legal move.

        Arguments:
        ----------
        - `state`: the current game state.
                   See FAQ and class `pacman.GameState`.

        Return:
        -------
        - A legal move as defined in `game.Directions`.
        """

        """
           XXX: DO NOT MODIFY THAT FUNCTION !!!
                Doing so will result in a 0 grade.
        """
        # Variables are specified in constructor.
        if self.beliefGhostStates is None:
            self.beliefGhostStates = state.getGhostBeliefStates()
        if self.walls is None:
            self.walls = state.getWalls()

        newBeliefStates = self.update_belief_state(self._get_evidence(state),
                                                   state.getPacmanPosition())
        self._record_metrics(self.beliefGhostStates, state)

        return newBeliefStates
