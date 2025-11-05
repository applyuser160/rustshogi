Changelog
=========

This document records the changes for each version of rustshogi.

Version 0.1.0 (2025-01-XX)
=============================

Initial release

New Features
~~~~~~~~~~~~

*   Basic shogi board representation and operations (`Board` class)
*   Generation and search of legal moves (`search_moves` method)
*   Execution of moves (`execute_move` method)
*   Piece placement (`deploy` method)
*   Hand management (`Hand` class)
*   Game state management (`Game` class)
*   Coordinate representation (`Address` class)
*   Representation of piece types and colors (`PieceType`, `ColorType` enums)
*   Python bindings

API
~~~

*   :py:class:`Address` - Represents coordinates
*   :py:class:`ColorType` - Represents the players (Black/White)
*   :py:class:`PieceType` - Represents the type of a piece
*   :py:class:`Piece` - Represents a piece
*   :py:class:`Move` - Represents a move
*   :py:class:`Hand` - Manages pieces in hand
*   :py:class:`Board` - Represents the shogi board
*   :py:class:`Game` - Manages the game
