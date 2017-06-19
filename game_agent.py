#!coding=utf-8
"""
Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player, a=3, b=2, c=1, d=1):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Notes
    -----
    The default parameters of `a`, `b`, `c`, `d` are leanred with grid search.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    a, b, c, d : int
        Learnable scale parameters.

    Returns
    -------
    score: float
        The heuristic value of the current game state to the specified player.

    """
    opp = game.get_opponent(player)

    own_moves = game.get_legal_moves(player)
    opp_moves = game.get_legal_moves(opp)

    num_own_moves = len(own_moves)
    num_opp_moves = len(opp_moves)

    if game.is_loser(player) or num_own_moves == 0:
        return float('-inf')
    if game.is_winner(player) or num_opp_moves == 0:
        return float('inf')

    own_controlled = []
    for move in own_moves:
        own_controlled.extend(game.forecast_move(move).get_legal_moves())
    num_own_controlled = len(set(own_controlled))

    opp_controlled = []
    for move in opp_moves:
        opp_controlled.extend(game.forecast_move(move).get_legal_moves(opp))
    num_opp_controlled = len(set(opp_controlled))

    own_score = num_own_moves * a + num_own_controlled * c
    opp_score = num_opp_moves * b + num_opp_controlled * d

    return float(own_score - opp_score)


def custom_score_2(game, player, a=1, b=1, c=1):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    a, b, c : int
        Learnable parameters.

    Returns
    -------
    score: float
        The heuristic value of the current game state to the specified player.

    """
    opp = game.get_opponent(player)

    own_moves = game.get_legal_moves(player)
    opp_moves = game.get_legal_moves(opp)

    num_own_moves = len(own_moves)
    num_opp_moves = len(opp_moves)

    if game.is_loser(player) or num_own_moves == 0:
        return float('-inf')
    if game.is_winner(player) or num_opp_moves == 0:
        return float('inf')

    num_next_own = sum(
        [len(game.forecast_move(m).get_legal_moves()) for m in own_moves])
    num_next_opp = sum(
        [len(game.forecast_move(m).get_legal_moves(opp)) for m in opp_moves])

    player_score = num_next_own * b + num_own_moves * a
    opp_score = num_next_opp * c
    return float(player_score - opp_score)


def custom_score_3(game, player, a=2, b=3):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This is about 3% better than `improved_score`.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    a, b : int
        Learnable parameters.

    Returns
    -------
    score: float
        The heuristic value of the current game state to the specified player.

    See Also
    --------
    The grid search is implemented at ``tornament.grid_search_custom_fn3_ab``.

    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    num_own_moves = len(game.get_legal_moves(player))
    num_opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    return float(num_own_moves * a - num_opp_moves * b)


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            # Handle any actions required after timeout as needed
            # Just randomly choose a legal move
            legal_moves = game.get_legal_moves(self)
            if len(legal_moves) > 0:
                best_move = random.choice(legal_moves)

        # Return the best move from the last completed search iteration
        return best_move

    def max_value(self, game, depth):
        """
        Return the maximum score at the given state.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        score : float
            The maximum score.

        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        if game.utility(self) != 0:
            return game.utility(self)
        elif depth == 0:
            return self.score(game, self)
        else:
            legal_moves = game.get_legal_moves()
            return max([self.min_value(game.forecast_move(move), depth - 1)
                        for move in legal_moves])

    def min_value(self, game, depth):
        """
        Return the minimum score at the given state.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        score : float
            The minimum score.

        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        if game.utility(self) != 0:
            return game.utility(self)
        elif depth == 0:
            return self.score(game, self)
        else:
            legal_moves = game.get_legal_moves()
            return min([self.max_value(game.forecast_move(move), depth - 1)
                        for move in legal_moves])

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        best_move = (-1, -1)

        legal_moves = game.get_legal_moves(self)
        if len(legal_moves) == 0:
            return best_move

        _, best_move = max(
            [(self.min_value(game.forecast_move(move), depth - 1), move)
             for move in legal_moves])

        return best_move


class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        move: (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.

        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        move = (-1, -1)
        depth = 0

        try:
            while self.time_left() > self.TIMER_THRESHOLD:
                depth += 1
                move = self.alphabeta(game, depth)
        except SearchTimeout:
            pass
        finally:
            return move

    def min_value(self, game, depth, alpha, beta):
        """
        Minimize the opponent.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        score : float
            The score.
        move : (int, int)
            The move corresponding to the `score`.

        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        utility = game.utility(self)
        if utility != 0:
            return utility, (-1, -1)
        elif depth == 0:
            return self.score(game, self), (-1, -1)
        else:
            legal_moves = game.get_legal_moves()
            best_score = float('inf')
            best_move = legal_moves[0]
            for move in legal_moves:
                score, _ = self.max_value(
                    game.forecast_move(move), depth - 1, alpha, beta)
                best_score = min(score, best_score)
                if best_score < beta:
                    beta = best_score
                    best_move = move
                if beta <= alpha:
                    break
            return best_score, best_move

    def max_value(self, game, depth, alpha, beta):
        """
        Maximize the player.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state
        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting
        alpha : float
            Alpha limits the lower bound of search on minimizing layers
        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        score : float
            The score.
        move : (int, int)
            The move corresponding to the `score`.

        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        utility = game.utility(self)
        if utility != 0:
            return utility, (-1, -1)
        elif depth == 0:
            return self.score(game, self), (-1, -1)
        else:
            legal_moves = game.get_legal_moves()
            best_score = float('-inf')
            best_move = legal_moves[0]
            for move in legal_moves:
                score, _ = self.min_value(
                    game.forecast_move(move), depth - 1, alpha, beta)
                best_score = max(score, best_score)
                if best_score > alpha:
                    alpha = best_score
                    best_move = move
                if alpha >= beta:
                    break
            return best_score, best_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        move : (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        move = (-1, -1)
        legal_moves = game.get_legal_moves(self)

        if not legal_moves:
            return float('-inf'), move

        _, move = self.max_value(game, depth, alpha, beta)
        return move
