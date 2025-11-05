Examples
========

This section introduces practical examples using rustshogi.

Basic Game
==========

.. code-block:: python

   from rustshogi import Board, ColorType, Move, Address
   import random

   def random_game():
       """Executes a random game"""
       board = Board("startpos")

       while True:
           is_finished, winner = board.is_finished()
           if is_finished:
               print(f"Game over: Winner {winner}")
               break

           # Determine the current turn
           current_color = ColorType.Black if board.move_count % 2 == 0 else ColorType.White
           legal_moves = board.search_moves(current_color)

           if not legal_moves:
               print("No legal moves available")
               break

           # Choose a move randomly
           move = random.choice(legal_moves)
           board.execute_move(move)

           print(f"Move {board.move_count}: {move}")
           print(board)
           print("-" * 40)

       return board

Piece Placement and Movement
============================

.. code-block:: python

   from rustshogi import Board, Address, PieceType, ColorType

   def piece_placement_example():
       """Example of piece placement and movement"""
       board = Board("startpos")

       # Place a piece at a specific position
       address = Address(5, 5)  # Position 5e
       board.deploy(address, PieceType.Pawn, ColorType.Black)

       # Check the placed piece
       piece = board.get_piece(address)
       print(f"Piece at 5e: {piece}")

       # Move a piece
       legal_moves = board.search_moves(ColorType.Black)
       if legal_moves:
           move = legal_moves[0]
           print(f"Executing move: {move}")
           board.execute_move(move)

       return board

Position Analysis
=================

.. code-block:: python

   from rustshogi import Board, ColorType, Address

   def analyze_position():
       """Detailed analysis of a position"""
       board = Board("startpos")

       print("=== Position Analysis ===")
       print(f"Board state:")
       print(board)

       # Compare legal moves for Black and White
       black_moves = board.search_moves(ColorType.Black)
       white_moves = board.search_moves(ColorType.White)

       print(f"Number of legal moves for Black: {len(black_moves)}")
       print(f"Number of legal moves for White: {len(white_moves)}")

       # Details of each legal move
       print("\n=== Legal moves for Black ===")
       for i, move in enumerate(black_moves[:5]):  # First 5 moves only
           print(f"{i+1}. {move}")
           print(f"   From: {move.get_from()}")
           print(f"   To: {move.get_to()}")
           print(f"   Piece: {move.get_piece()}")
           print(f"   Is Promote: {move.is_promote()}")
           print(f"   Is Drop: {move.is_drop()}")

       # Check for game end
       is_finished, winner = board.is_finished()
       if is_finished:
           print(f"\nGame over: Winner {winner}")
       else:
           print("\nGame in progress")

Hand Management
===============

.. code-block:: python

   from rustshogi import Hand, Piece, ColorType, PieceType

   def hand_management_example():
       """Example of hand management"""
       # Create an empty hand
       hand = Hand([], [])

       # Add a piece
       hand.add_piece(ColorType.Black, PieceType.Pawn)
       hand.add_pieces(ColorType.Black, PieceType.Pawn, 3)  # Add 3 Pawns

       # Check the hand
       black_pieces = hand.get_player_pieces(ColorType.Black)
       print(f"Black's hand: {black_pieces}")

       # Remove a piece
       hand.decrease_piece(ColorType.Black, PieceType.Pawn)

       # Check the updated hand
       black_pieces = hand.get_player_pieces(ColorType.Black)
       print(f"Updated Black's hand: {black_pieces}")

       return hand

Using the Game Class
====================

.. code-block:: python

   from rustshogi import Game, Board, ColorType, Move

   def game_management_example():
       """Example of game management using the Game class"""
       # Create a game with an initial position
       board = Board("startpos")
       game = Game(board=board, move_number=1, turn=ColorType.Black)

       print("=== Game Start ===")
       print(f"Move number: {game.move_number}")
       print(f"Turn: {game.turn}")

       # Execute a move
       legal_moves = board.search_moves(ColorType.Black)
       if legal_moves:
           move = legal_moves[0]
           game.execute_move(move)
           print(f"Executed move: {move}")

       # Check for game end
       is_finished, winner = game.is_finished()
       if is_finished:
           print(f"Game over: Winner {winner}")
       else:
           print("Game in progress")

       # Run a random game
       random_game_result = game.random_play()
       print(f"Result of random game: {random_game_result}")

       return game

Using Evaluation Functions
==========================

SimpleEvaluator
---------------

.. code-block:: python

   from rustshogi import Board, ColorType, SimpleEvaluator

   def simple_evaluator_example():
       """Example of position evaluation using SimpleEvaluator"""
       board = Board("startpos")
       evaluator = SimpleEvaluator()

       # Evaluate the position from Black's perspective
       score = evaluator.evaluate(board, ColorType.Black)
       print(f"Evaluation for Black: {score}")

       # Evaluate the position from White's perspective
       score = evaluator.evaluate(board, ColorType.White)
       print(f"Evaluation for White: {score}")

       return evaluator

NeuralEvaluator (Neural Network Evaluator)
------------------------------------------

.. code-block:: python

   from rustshogi import Board, ColorType, NeuralEvaluator

   def neural_evaluator_example():
       """Example of position evaluation using NeuralEvaluator"""
       # Create a NeuralEvaluator with database and model path
       evaluator = NeuralEvaluator(
           db_type_str="sqlite",
           connection_string="training.db",
           model_path="model.mpk"
       )

       # Initialize the database (only on the first run)
       evaluator.init_database()

       # Generate and save random boards to the database
       evaluator.generate_and_save_random_boards(1000)

       # Evaluate a position
       board = Board("startpos")
       score = evaluator.evaluate(board, ColorType.Black)
       print(f"Evaluation: {score}")

       # Predict the win rate for a specific position
       white_win_rate, black_win_rate, draw_rate = evaluator.evaluate_position(board)
       print(f"White win rate: {white_win_rate:.2%}")
       print(f"Black win rate: {black_win_rate:.2%}")
       print(f"Draw rate: {draw_rate:.2%}")

       return evaluator

Using the Search Engine
=======================

Basic Search
------------

.. code-block:: python

   from rustshogi import Board, ColorType, SearchEngine, SimpleEvaluator

   def basic_search_example():
       """Example of a basic search"""
       board = Board("startpos")
       
       # Create a default search engine (uses MinMax and SimpleEvaluator)
       engine = SearchEngine()

       # Execute a search with depth 3
       result = engine.search(board, ColorType.Black, depth=3)
       
       print(f"Evaluation: {result.score}")
       print(f"Best move: {result.best_move}")
       print(f"Nodes searched: {result.nodes_searched}")

       # Execute the best move
       if result.best_move:
           board.execute_move(result.best_move)
           print(f"Executed move: {result.best_move}")

       return engine

Using AlphaBeta Search
----------------------

.. code-block:: python

   from rustshogi import Board, ColorType, SearchEngine

   def alphabeta_search_example():
       """Example using AlphaBeta search"""
       board = Board("startpos")

       # Create a search engine using AlphaBeta search
       engine = SearchEngine(
           algorithm="alphabeta",
           max_nodes=1000000  # Maximum number of nodes to search
       )

       # Execute a search with depth 4
       result = engine.search(board, ColorType.Black, depth=4)
       
       print(f"Evaluation: {result.score}")
       print(f"Best move: {result.best_move}")
       print(f"Nodes searched: {result.nodes_searched}")

       return engine

Search with a Custom Evaluator
------------------------------

.. code-block:: python

   from rustshogi import Board, ColorType, SearchEngine, SimpleEvaluator, NeuralEvaluator

   def custom_evaluator_search_example():
       """Example of search with a custom evaluator"""
       board = Board("startpos")

       # Using SimpleEvaluator
       simple_evaluator = SimpleEvaluator()
       engine = SearchEngine(
           algorithm="alphabeta",
           max_nodes=1000000,
           evaluator=simple_evaluator
       )

       result = engine.search(board, ColorType.Black, depth=3)
       print(f"Evaluation with SimpleEvaluator: {result.score}")

       # Using NeuralEvaluator (if model exists)
       neural_evaluator = NeuralEvaluator(
           db_type_str="sqlite",
           connection_string="training.db",
           model_path="model.mpk"
       )
       engine_neural = SearchEngine(
           algorithm="alphabeta",
           max_nodes=1000000,
           evaluator=neural_evaluator
       )

       result_neural = engine_neural.search(board, ColorType.Black, depth=3)
       print(f"Evaluation with NeuralEvaluator: {result_neural.score}")

       return engine

Automated Game System
=====================

Automated Game using Evaluation and Search
------------------------------------------

.. code-block:: python

   from rustshogi import Board, ColorType, SearchEngine, SimpleEvaluator

   def auto_game_example():
       """Example of an automated game using evaluation and search"""
       board = Board("startpos")
       engine = SearchEngine(
           algorithm="alphabeta",
           max_nodes=500000,
           evaluator=SimpleEvaluator()
       )

       current_color = ColorType.Black
       move_count = 0
       max_moves = 50

       while move_count < max_moves:
           # Check for game end
           is_finished, winner = board.is_finished()
           if is_finished:
               print(f"Game over: Winner {winner}")
               break

           # Execute search to get the best move
           result = engine.search(board, current_color, depth=3)
           
           if result.best_move:
               board.execute_move(result.best_move)
               move_count += 1
               print(f"Move {move_count}: {result.best_move} (Evaluation: {result.score:.2f})")
               print(board)
               print("-" * 40)
           else:
               print("No legal moves available")
               break

           # Switch turns
           current_color = ColorType.White if current_color == ColorType.Black else ColorType.Black

       return board
