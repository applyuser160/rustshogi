Quickstart
==========

This guide explains the basic usage of rustshogi.

Basic Usage
===========

First, import rustshogi and create an initial position:

.. code-block:: python

   from rustshogi import Board, ColorType, Move, Address, PieceType

   # Create an initial position
   board = Board("startpos")
   print(board)

Displaying the Board
--------------------

You can check the current state of the board:

.. code-block:: python

   # Display the board as a string
   print(str(board))

   # Check the piece at a specific position
   address = Address(5, 5)  # Position 5e
   piece = board.get_piece(address)
   print(f"Piece at 5e: {piece}")

Searching for Legal Moves
=========================

Search for legal moves in the current position:

.. code-block:: python

   # Search for Black's legal moves
   legal_moves = board.search_moves(ColorType.Black)
   print(f"Number of legal moves for Black: {len(legal_moves)}")

   # Display the first legal move
   if legal_moves:
       print(f"First legal move: {legal_moves[0]}")

Executing a Move
================

Execute a legal move to advance the position:

.. code-block:: python

   # Execute a move
   if legal_moves:
       move = legal_moves[0]
       board.execute_move(move)
       print(f"Executed move: {move}")
       print(board)

Move Representation
===================

Moves are represented in the following format:

.. code-block:: python

   # Detailed information of a move
   move = legal_moves[0]
   print(f"From: {move.get_from()}")
   print(f"To: {move.get_to()}")
   print(f"Piece: {move.get_piece()}")
   print(f"Is promote: {move.is_promote()}")
   print(f"Is drop: {move.is_drop()}")

Checking for Game End
=====================

Check if the game has ended:

.. code-block:: python

   # Check for game end
   is_finished, winner = board.is_finished()
   if is_finished:
       print("Game over")
       print(f"Winner: {winner}")
   else:
       print("Game in progress")

Complete Example
----------------

Here is an example of a simple game:

.. code-block:: python

   from rustshogi import Board, ColorType, Game, Move

   # Initial position
   board = Board("startpos")

   # Play up to 10 moves
   for i in range(10):
       is_finished, winner = board.is_finished()
       if is_finished:
           print(f"Game over: Winner {winner}")
           break

       # Determine the current turn (alternating)
       current_color = ColorType.Black if i % 2 == 0 else ColorType.White
       legal_moves = board.search_moves(current_color)

       if not legal_moves:
           print("No legal moves")
           break

       # Play the first legal move
       move = legal_moves[0]
       board.execute_move(move)

       print(f"Move {i+1}: {move}")
       print(board)
       print("-" * 40)

   print("Game finished")

Using the Evaluation Function
=============================

rustshogi provides an evaluation function to evaluate the position. The most basic evaluation function is SimpleEvaluator:

.. code-block:: python

   from rustshogi import Board, ColorType, SimpleEvaluator

   board = Board("startpos")
   evaluator = SimpleEvaluator()

   # Evaluate the position from Black's perspective
   score = evaluator.evaluate(board, ColorType.Black)
   print(f"Evaluation for Black: {score}")

Using the Search Engine
=======================

You can use the search engine to automatically find the best move:

.. code-block:: python

   from rustshogi import Board, ColorType, SearchEngine

   board = Board("startpos")
   
   # Create a search engine (default is MinMax search)
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

For more details, please refer to :doc:`examples`.
