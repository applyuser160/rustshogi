API Reference
=============

This is the complete API reference for rustshogi.

.. toctree::
   :maxdepth: 2

   ../reference/rustshogi

Module Overview
===============

rustshogi consists of the following main classes and enumerations:

* :doc:`../reference/rustshogi` - Main module (basic functions for shogi board, moves, pieces, etc.)

Basic Types
===========

.. py:class:: Address
   :module: rustshogi

   A class that represents coordinates on the shogi board. It includes column and row information.

.. py:class:: ColorType
   :module: rustshogi

   An enumeration that represents the players. It has values for Black (Sente) and White (Gote).

.. py:class:: PieceType
   :module: rustshogi

   An enumeration that represents the types of shogi pieces. It includes King, Gold, Rook, Bishop, Silver, Knight, Lance, Pawn, and their promoted versions.

.. py:class:: Piece
   :module: rustshogi

   A class that represents a shogi piece. It includes information about the piece type (PieceType) and its color (ColorType).

.. py:class:: Move
   :module: rustshogi

   A class that represents a shogi move. It includes information such as the source, destination, piece type, and promotion status.

.. py:class:: Hand
   :module: rustshogi

   A class that represents the pieces in a player's hand.

.. py:class:: Board
   :module: rustshogi

   A class that represents the shogi board. It provides functions for managing the position, generating legal moves, and executing moves.

.. py:class:: Game
   :module: rustshogi

   A class that manages the overall game. It handles game progression, win/loss determination, and random games.

Evaluation Functions
====================

rustshogi provides multiple evaluation functions to evaluate positions.

.. py:class:: SimpleEvaluator
   :module: rustshogi

   A simple evaluator. It evaluates the position using only the value of the pieces.

   .. py:method:: evaluate(board: Board, color: ColorType) -> float

      Evaluates the board.

      :param board: The board to evaluate
      :param color: The color of the player to evaluate for (Black or White)
      :returns: The evaluation score (from the perspective of `color`, higher is better)

.. py:class:: NeuralEvaluator
   :module: rustshogi

   A neural network evaluator. It uses a machine learning model to predict the win rate of a position and calculate an evaluation score.

   .. py:method:: __init__(db_type_str: Optional[str] = None, connection_string: Optional[str] = None, model_path: Optional[str] = None)

      Initializes the evaluator.

      :param db_type_str: Database type ("sqlite" or "postgres")
      :param connection_string: Database connection string
      :param model_path: Path to the model file

   .. py:method:: init_database() -> None

      Initializes the database tables.

   .. py:method:: evaluate(board: Board, color: ColorType) -> float

      Evaluates the board.

      :param board: The board to evaluate
      :param color: The color of the player to evaluate for
      :returns: The evaluation score

   .. py:method:: evaluate_position(board: Board, model_path: Optional[str] = None) -> Tuple[float, float, float]

      Predicts the win rate for a specific position.

      :param board: The board to evaluate
      :param model_path: Path to the model file (optional)
      :returns: A tuple of (white win rate, black win rate, draw rate)

   .. py:method:: generate_and_save_random_boards(count: int) -> int

      Generates random boards and saves them to the database.

      :param count: The number of boards to generate
      :returns: The number of boards saved

Search
======

rustshogi provides a search engine and search algorithms to find the best move.

.. py:class:: SearchEngine
   :module: rustshogi

   The search engine. It combines a search algorithm and an evaluation function to find the best move.

   .. py:method:: __init__(algorithm: str = "minmax", max_nodes: int = 1000000, evaluator: Optional[Evaluator] = None)

      Initializes the search engine.

      :param algorithm: Search algorithm ("minmax" or "alphabeta")
      :param max_nodes: Maximum number of nodes to search
      :param evaluator: Evaluator (optional, defaults to SimpleEvaluator)

   .. py:method:: search(board: Board, color: ColorType, depth: int) -> EvaluationResult

      Executes a search to find the best move.

      :param board: The current board
      :param color: The color of the player to move
      :param depth: The search depth
      :returns: The evaluation result (includes score, best move, and number of nodes searched)

.. py:class:: EvaluationResult
   :module: rustshogi

   A class representing the search result.

   .. py:attribute:: score

      The evaluation score (float)

   .. py:attribute:: best_move

      The best move (Optional[Move])

   .. py:attribute:: nodes_searched

      The number of nodes searched (int)

.. py:class:: MinMaxSearchStrategy
   :module: rustshogi

   The Minimax search algorithm. It explores all possible moves to select the optimal one.

   .. py:method:: __init__(max_nodes: int = 100)

      Initializes the Minimax search strategy.

      :param max_nodes: Maximum number of nodes to search

.. py:class:: AlphaBetaSearchStrategy
   :module: rustshogi

   The Alpha-Beta search algorithm. It is an optimized version of Minimax that prunes branches of the search tree that are not worth exploring.

   .. py:method:: __init__(max_nodes: int = 100)

      Initializes the Alpha-Beta search strategy.

      :param max_nodes: Maximum number of nodes to search
