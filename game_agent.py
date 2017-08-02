
class MiniMaxSearchTimeout(Exception) :
    def __init__(self, best_move):
        self.best_move = best_move

class SearchTimeout(Exception):
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

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

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    my_moves = len(game.get_legal_moves(player))
    opponent_moves = len(game.get_legal_moves(game.get_opponent(player)))

    # Penalize/reward move count if some moves are in the corner
    return float(my_moves - 2 * opponent_moves)


def custom_score_2(game, player):
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

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    blank_spaces = len(game.get_blank_spaces())
    my_moves = len(game.get_legal_moves(player))

    return float(blank_spaces / 2 + 2 * my_moves)



def custom_score_3(game, player):
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

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    return float(custom_score(game,player)) + (float(custom_score_2(game,player))/13.0)




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


class MinimaxHelper :
    """Class is is a helper for MinimaxPlayer.minimax and ensures timeouts are obeyed

        Parameters
        ----------
        agent : `isolation`
        An instance of `isolation` encoding the current state of the
        game (e.g., player locations and blocked cells).

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
    def __init__(self, agent, game, depth, score_fn):
        self.agent = agent
        self.game = game
        self.depth = depth
        self.best_move = None
        self.score_fn = score_fn

        # Raise SearchTimeout() if threshold is exceeded
        if self.agent.time_left() < self.agent.TIMER_THRESHOLD:
            raise SearchTimeout()

    def run_minimax(self):
        # Raise SearchTimeout() if threshold is exceeded and return best_move
        if self.agent.time_left() < self.agent.TIMER_THRESHOLD:
            raise MiniMaxSearchTimeout(self.best_move)


        # Set max value to the -inf.  This function searches for max value at the end
        best_score = float("-inf")
        # Set best move
        best_move = (-1, -1)

        # Set the the value of best_move to the first possible move in the event of a timeout
        my_moves = self.game.get_legal_moves()
        if len(my_moves) >= 1:
            best_move = my_moves[0]
        else:
            # If no moves remain return (-1,-1)
            return best_move

        # Iterate over possible moves
        for move in my_moves:
            # Forecast game board for specified move.
            possible_game = self.game.forecast_move(move)
            score = self.min_value(possible_game, self.depth - 1)
            # Each iteration, update game_value, and best_move if game_value > current max value
            if score > best_score:
                best_move, best_score = move, score
                self.best_move = best_move
        return best_move


    def result(self, game, move):
        #If timer threshold not reached, return game forecast
        if self.agent.time_left() < self.agent.TIMER_THRESHOLD:
            raise SearchTimeout()
        return game.forecast_move(move)

    def max_value(self, game, depth):
        if self.agent.time_left() < self.agent.TIMER_THRESHOLD:
            raise SearchTimeout()

        # If the maximum depth is reached, then retrieve score and return score
        if depth == 0:
            score = self.score_fn(game,self.agent)
            return score

        # Set score to the lowest value possible in preperation for max
        score = float("-inf")

        # Iterate through each move
        for move in game.get_legal_moves(game.active_player):
            # Retrieve score for each move, determine if it's highest value
            score = max(score, self.min_value(self.result(game,move), depth - 1))
        return score

    def min_value(self, game, depth):
        if self.agent.time_left() < self.agent.TIMER_THRESHOLD:
            raise SearchTimeout()

        # If the maximum depth is reached, then retrieve score and return score
        if depth == 0:
            score = self.score_fn(game,self.agent)
            return score

        # Set score to the highest value possible in preperation for min
        score = float("inf")
        for move in game.get_legal_moves(game.active_player):
            # Retrieve score for each move, determine if it's lowest value
            score = min(score, self.max_value(self.result(game,move), depth - 1))
        return score

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

        # If no legal removes remain, set best_move to (-1,-1)
        best_move = (-1,-1)
        if len(game.get_legal_moves()) == 0 :
            best_move = (-1,-1)
        else :
            best_move = game.get_legal_moves()[0]

        try:
            # Attempt to search [search_depth] levels deep and return best move
            best_move = self.minimax(game, self.search_depth)
            return best_move

        except SearchTimeout:
            return best_move  # If timeout is exceeded, return best_move


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

        # If timer threshold exceeded, raise SearchTimeout() exception
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # Leverage MinimaxHelper to perform minmax.
        mh = MinimaxHelper(self, game, depth, self.score)
        # Runs the algorithm and stores result
        best_move = mh.run_minimax()

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
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left


        best_move = (-1, -1)
        # If legal removes remain, set best_move to the first legal move (In case timeout, is exceeded)
        my_moves = game.get_legal_moves()
        if len(my_moves) >= 1:
            my_moves = game.get_legal_moves()
            best_move = my_moves[0]
        else:
            # If no moves remain, return (-1,-1)
            return best_move

        try:
            # Set depth to 1 to start 1 search level deep
            depth = 1
            while time_left() > self.TIMER_THRESHOLD:
                # While the timer has remaining time, search for best move at set depth
                next_move = self.alphabeta(game, depth)
                # Return if no moves remain
                if not next_move:
                    return best_move
                else:
                    # If tree is successfully searched, update best_move
                    best_move = next_move

                # Increase depth by 1 each iteration
                depth += 1

        except SearchTimeout:
            return best_move  # Return best move if timer threshold exceeded

        # Return best move found
        return best_move

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

        # Set max value to the -inf.  This function searches for max value at the end
        best_score = float("-inf")
        # Set best move
        best_move = None
        my_moves = game.get_legal_moves()

        # Iterate over all possible candidate moves
        for move in my_moves:
            # Forecast game board for specified move.
            possible_game = game.forecast_move(move)
            score = self.min_value(possible_game, depth-1, alpha, beta)
            # Each iteration, update game_value, and best_move if game_value > current max value
            if score > best_score:
                best_move, best_score = move, score
            # If best_score > beta, break the function.  Alpha > Beta equiv
            if best_score >= beta:
                break

            # Update Alpha
            alpha = max(alpha, best_score)
        return best_move

    def max_value(self, game, depth, alpha, beta):
        """Return max value move of candidate game state.
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
        int
            The max value move of candidate game state
        """

        #If timer threshold reached, raise SearchTimeout()
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # If the maximum depth is reached, then retrieve score and return score
        if depth == 0:
            return self.score(game, self)

        # Set score to the lowest value possible in preperation for max
        score = float("-inf")

        # Iterate through each move
        for move in game.get_legal_moves():
            # Retrieve score for each move, determine if it's highest value
            score = max(score, self.min_value(game.forecast_move(move), depth-1, alpha, beta))

            # Update alpha to highest score
            alpha = max(alpha, score)

            # If alpha >= beta, return score
            if score >= beta:
                return score
        return score

    def min_value(self, game, depth, alpha, beta):
        """Return minimum value move of candidate game state.
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
        int
            The minimum value move of candidate game state
        """

        # If timer threshold reached, raise SearchTimeout()
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # If the maximum depth is reached, then retrieve score and return score
        if depth == 0:
            return self.score(game, self)

        # Set score to the highest value possible in preparation for min function
        score = float("inf")

        for move in game.get_legal_moves():
            score = min(score, self.max_value(game.forecast_move(move), depth-1, alpha, beta))

            # If alpha >= beta, return score
            if score <= alpha:
                return score # Found candidate lower value

            # Update beta to the lowest score
            beta = min(beta, score)
        return score
