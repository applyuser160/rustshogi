クイックスタート
===============

このガイドでは、rustshogiの基本的な使用方法を説明します。

基本的な使用方法
==============

まず、rustshogiをインポートして初期局面を作成します：

.. code-block:: python

   from rustshogi import Board, ColorType, Move, Address, PieceType

   # 初期局面を作成
   board = Board("startpos")
   print(board)

盤面の表示
----------

盤面の現在の状態を確認できます：

.. code-block:: python

   # 盤面を文字列として表示
   print(str(board))

   # 特定の位置の駒を確認
   address = Address(5, 5)  # 5五の位置
   piece = board.get_piece(address)
   print(f"5五の駒: {piece}")

合法手の検索
============

現在の局面での合法手を検索：

.. code-block:: python

   # 先手の合法手を検索
   legal_moves = board.search_moves(ColorType.Black)
   print(f"先手の合法手数: {len(legal_moves)}")

   # 最初の合法手を表示
   if legal_moves:
       print(f"最初の合法手: {legal_moves[0]}")

手を実行
========

合法手を実行して局面を進めます：

.. code-block:: python

   # 手を実行
   if legal_moves:
       move = legal_moves[0]
       board.execute_move(move)
       print(f"手を実行しました: {move}")
       print(board)

手の表現
========

手は以下のような形式で表現されます：

.. code-block:: python

   # 手の詳細情報
   move = legal_moves[0]
   print(f"移動元: {move.get_from()}")
   print(f"移動先: {move.get_to()}")
   print(f"駒: {move.get_piece()}")
   print(f"成り: {move.is_promote()}")
   print(f"打ち駒: {move.is_drop()}")

ゲームの終了判定
==============

ゲームが終了したかどうかを確認：

.. code-block:: python

   # ゲーム終了判定
   is_finished, winner = board.is_finished()
   if is_finished:
       print("ゲーム終了")
       print(f"勝者: {winner}")
   else:
       print("ゲーム継続中")

完全な例
--------

以下は簡単な対局の例です：

.. code-block:: python

   from rustshogi import Board, ColorType, Game, Move

   # 初期局面
   board = Board("startpos")

   # 10手まで対局
   for i in range(10):
       is_finished, winner = board.is_finished()
       if is_finished:
           print(f"ゲーム終了: 勝者 {winner}")
           break

       # 現在の手番を決定（交互に指す）
       current_color = ColorType.Black if i % 2 == 0 else ColorType.White
       legal_moves = board.search_moves(current_color)

       if not legal_moves:
           print("合法手がありません")
           break

       # 最初の合法手を指す
       move = legal_moves[0]
       board.execute_move(move)

       print(f"手数 {i+1}: {move}")
       print(board)
       print("-" * 40)

   print("対局終了")

評価関数の使用
==============

rustshogiは、局面を評価するための評価関数を提供します。最も基本的な評価関数はSimpleEvaluatorです：

.. code-block:: python

   from rustshogi import Board, ColorType, SimpleEvaluator

   board = Board("startpos")
   evaluator = SimpleEvaluator()

   # 先手の視点で局面を評価
   score = evaluator.evaluate(board, ColorType.Black)
   print(f"先手の評価値: {score}")

探索エンジンの使用
================

探索エンジンを使用すると、最善手を自動的に見つけることができます：

.. code-block:: python

   from rustshogi import Board, ColorType, SearchEngine

   board = Board("startpos")
   
   # 探索エンジンを作成（デフォルトはMinMax探索）
   engine = SearchEngine()

   # 深度3で探索を実行
   result = engine.search(board, ColorType.Black, depth=3)
   
   print(f"評価値: {result.score}")
   print(f"最善手: {result.best_move}")
   print(f"探索ノード数: {result.nodes_searched}")

   # 最善手を実行
   if result.best_move:
       board.execute_move(result.best_move)
       print(f"実行した手: {result.best_move}")

詳細については、:doc:`examples` を参照してください。
