#!coding=utf-8
"""Implement your own custom search agent using any combination of techniques
you choose.  This agent will compete against other students (and past
champions) in a tournament.

         COMPLETING AND SUBMITTING A COMPETITION AGENT IS OPTIONAL
"""


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
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

    own_score = num_own_moves * 3 + num_own_controlled * 1
    opp_score = num_opp_moves * 2 + num_opp_controlled * 1

    return float(own_score - opp_score)


class CustomPlayer:
    """Game-playing agent to use in the optional player vs player Isolation
    competition.

    You must at least implement the get_move() method and a search function
    to complete this class, but you may use any of the techniques discussed
    in lecture or elsewhere on the web -- opening books, MCTS, etc.

    **************************************************************************
          THIS CLASS IS OPTIONAL -- IT IS ONLY USED IN THE ISOLATION PvP
        COMPETITION.  IT IS NOT REQUIRED FOR THE ISOLATION PROJECT REVIEW.
    **************************************************************************

    Parameters
    ----------
    data : string
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted.  Note that
        the PvP competition uses more accurate timers that are not cross-
        platform compatible, so a limit of 1ms (vs 10ms for the other classes)
        is generally sufficient.
    """

    def __init__(self, data=None, timeout=1.):
        self.score = custom_score
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

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
