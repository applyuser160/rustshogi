rustshogi Documentation
========================

rustshogi is a high-performance shogi library implemented in Rust.
It can be used from Python applications through Python bindings.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api/index
   examples
   changelog

Features
--------

* Fast shogi board representation and manipulation
* Legal move generation and search
* Piece placement and movement
* Hand piece management
* Game state management
* Python bindings

Installation
============

.. code-block:: bash

   pip install rustshogi

Quickstart
==========

.. code-block:: python

   from rustshogi import Board, ColorType, Move, Address

   # Create an initial position
   board = Board("startpos")

   # Search for legal moves
   legal_moves = board.search_moves(ColorType.Black)

   # Execute a move
   if legal_moves:
       board.execute_move(legal_moves[0])

For detailed usage, please refer to :doc:`quickstart`.

API Reference
===============

The complete API reference can be found at :doc:`reference/rustshogi`.

.. toctree::
   :maxdepth: 1
   :caption: API Reference:

   reference/rustshogi

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
