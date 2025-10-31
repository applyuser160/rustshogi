例
==

このセクションでは、rustshogiを使用した実用的な例を紹介します。

基本的な対局
===========

.. code-block:: python

   from rustshogi import Board, ColorType, Move, Address
   import random

   def random_game():
       """ランダムな対局を実行"""
       board = Board("startpos")

       while True:
           is_finished, winner = board.is_finished()
           if is_finished:
               print(f"ゲーム終了: 勝者 {winner}")
               break

           # 現在の手番を決定
           current_color = ColorType.Black if board.move_count % 2 == 0 else ColorType.White
           legal_moves = board.search_moves(current_color)

           if not legal_moves:
               print("合法手がありません")
               break

           # ランダムに手を選択
           move = random.choice(legal_moves)
           board.execute_move(move)

           print(f"手数 {board.move_count}: {move}")
           print(board)
           print("-" * 40)

       return board

駒の配置と移動
=============

.. code-block:: python

   from rustshogi import Board, Address, PieceType, ColorType

   def piece_placement_example():
       """駒の配置と移動の例"""
       board = Board("startpos")

       # 特定の位置に駒を配置
       address = Address(5, 5)  # 5五の位置
       board.deploy(address, PieceType.Pawn, ColorType.Black)

       # 配置された駒を確認
       piece = board.get_piece(address)
       print(f"5五の駒: {piece}")

       # 駒の移動
       legal_moves = board.search_moves(ColorType.Black)
       if legal_moves:
           move = legal_moves[0]
           print(f"実行する手: {move}")
           board.execute_move(move)

       return board

局面の解析
==========

.. code-block:: python

   from rustshogi import Board, ColorType, Address

   def analyze_position():
       """局面の詳細な解析"""
       board = Board("startpos")

       print("=== 局面解析 ===")
       print(f"盤面状態:")
       print(board)

       # 先手と後手の合法手を比較
       black_moves = board.search_moves(ColorType.Black)
       white_moves = board.search_moves(ColorType.White)

       print(f"先手の合法手数: {len(black_moves)}")
       print(f"後手の合法手数: {len(white_moves)}")

       # 各合法手の詳細
       print("\n=== 先手の合法手 ===")
       for i, move in enumerate(black_moves[:5]):  # 最初の5手のみ
           print(f"{i+1}. {move}")
           print(f"   移動元: {move.get_from()}")
           print(f"   移動先: {move.get_to()}")
           print(f"   駒: {move.get_piece()}")
           print(f"   成り: {move.is_promote()}")
           print(f"   打ち駒: {move.is_drop()}")

       # ゲーム終了判定
       is_finished, winner = board.is_finished()
       if is_finished:
           print(f"\nゲーム終了: 勝者 {winner}")
       else:
           print("\nゲーム継続中")

持ち駒の管理
===========

.. code-block:: python

   from rustshogi import Hand, Piece, ColorType, PieceType

   def hand_management_example():
       """持ち駒の管理例"""
       # 空の持ち駒を作成
       hand = Hand([], [])

       # 駒を追加
       hand.add_piece(ColorType.Black, PieceType.Pawn)
       hand.add_pieces(ColorType.Black, PieceType.Pawn, 3)  # 歩を3枚追加

       # 持ち駒を確認
       black_pieces = hand.get_player_pieces(ColorType.Black)
       print(f"先手の持ち駒: {black_pieces}")

       # 駒を減らす
       hand.decrease_piece(ColorType.Black, PieceType.Pawn)

       # 更新後の持ち駒を確認
       black_pieces = hand.get_player_pieces(ColorType.Black)
       print(f"更新後の先手の持ち駒: {black_pieces}")

       return hand

Gameクラスの使用
==============

.. code-block:: python

   from rustshogi import Game, Board, ColorType, Move

   def game_management_example():
       """Gameクラスを使用した対局管理例"""
       # 初期局面でゲームを作成
       board = Board("startpos")
       game = Game(board=board, move_number=1, turn=ColorType.Black)

       print("=== ゲーム開始 ===")
       print(f"手数: {game.move_number}")
       print(f"手番: {game.turn}")

       # 手を実行
       legal_moves = board.search_moves(ColorType.Black)
       if legal_moves:
           move = legal_moves[0]
           game.execute_move(move)
           print(f"実行した手: {move}")

       # ゲーム終了判定
       is_finished, winner = game.is_finished()
       if is_finished:
           print(f"ゲーム終了: 勝者 {winner}")
       else:
           print("ゲーム継続中")

       # ランダム対局の実行
       random_game = game.random_play()
       print(f"ランダム対局の結果: {random_game}")

       return game

評価関数の使用
=============

SimpleEvaluator（簡易評価関数）
--------------------------------

.. code-block:: python

   from rustshogi import Board, ColorType, SimpleEvaluator

   def simple_evaluator_example():
       """SimpleEvaluatorを使用した局面評価の例"""
       board = Board("startpos")
       evaluator = SimpleEvaluator()

       # 先手の視点で局面を評価
       score = evaluator.evaluate(board, ColorType.Black)
       print(f"先手の評価値: {score}")

       # 後手の視点で局面を評価
       score = evaluator.evaluate(board, ColorType.White)
       print(f"後手の評価値: {score}")

       return evaluator

NeuralEvaluator（ニューラルネットワーク評価関数）
------------------------------------------------

.. code-block:: python

   from rustshogi import Board, ColorType, NeuralEvaluator

   def neural_evaluator_example():
       """NeuralEvaluatorを使用した局面評価の例"""
       # データベースとモデルパスを指定してNeuralEvaluatorを作成
       evaluator = NeuralEvaluator(
           db_type_str="sqlite",
           connection_string="training.db",
           model_path="model.mpk"
       )

       # データベースを初期化（初回のみ）
       evaluator.init_database()

       # ランダム盤面を生成してデータベースに保存
       evaluator.generate_and_save_random_boards(1000)

       # 盤面を評価
       board = Board("startpos")
       score = evaluator.evaluate(board, ColorType.Black)
       print(f"評価値: {score}")

       # 特定の局面の勝率を予測
       white_win_rate, black_win_rate, draw_rate = evaluator.evaluate_position(board)
       print(f"白勝率: {white_win_rate:.2%}")
       print(f"黒勝率: {black_win_rate:.2%}")
       print(f"引き分け率: {draw_rate:.2%}")

       return evaluator

探索エンジンの使用
================

基本的な探索
-----------

.. code-block:: python

   from rustshogi import Board, ColorType, SearchEngine, SimpleEvaluator

   def basic_search_example():
       """基本的な探索の例"""
       board = Board("startpos")
       
       # デフォルトの探索エンジンを作成（MinMax、SimpleEvaluatorを使用）
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

       return engine

AlphaBeta探索の使用
-----------------

.. code-block:: python

   from rustshogi import Board, ColorType, SearchEngine

   def alphabeta_search_example():
       """AlphaBeta探索を使用した例"""
       board = Board("startpos")

       # AlphaBeta探索を使用して探索エンジンを作成
       engine = SearchEngine(
           algorithm="alphabeta",
           max_nodes=1000000  # 最大探索ノード数
       )

       # 深度4で探索を実行
       result = engine.search(board, ColorType.Black, depth=4)
       
       print(f"評価値: {result.score}")
       print(f"最善手: {result.best_move}")
       print(f"探索ノード数: {result.nodes_searched}")

       return engine

カスタム評価関数を使用した探索
-----------------------------

.. code-block:: python

   from rustshogi import Board, ColorType, SearchEngine, SimpleEvaluator, NeuralEvaluator

   def custom_evaluator_search_example():
       """カスタム評価関数を使用した探索の例"""
       board = Board("startpos")

       # SimpleEvaluatorを使用
       simple_evaluator = SimpleEvaluator()
       engine = SearchEngine(
           algorithm="alphabeta",
           max_nodes=1000000,
           evaluator=simple_evaluator
       )

       result = engine.search(board, ColorType.Black, depth=3)
       print(f"SimpleEvaluatorでの評価値: {result.score}")

       # NeuralEvaluatorを使用（モデルが存在する場合）
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
       print(f"NeuralEvaluatorでの評価値: {result_neural.score}")

       return engine

自動対局システム
==============

評価関数と探索を使用した自動対局
--------------------------------

.. code-block:: python

   from rustshogi import Board, ColorType, SearchEngine, SimpleEvaluator

   def auto_game_example():
       """評価関数と探索を使用した自動対局の例"""
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
           # ゲーム終了判定
           is_finished, winner = board.is_finished()
           if is_finished:
               print(f"ゲーム終了: 勝者 {winner}")
               break

           # 探索を実行して最善手を取得
           result = engine.search(board, current_color, depth=3)
           
           if result.best_move:
               board.execute_move(result.best_move)
               move_count += 1
               print(f"手数 {move_count}: {result.best_move} (評価値: {result.score:.2f})")
               print(board)
               print("-" * 40)
           else:
               print("合法手がありません")
               break

           # 手番を交代
           current_color = ColorType.White if current_color == ColorType.Black else ColorType.Black

       return board
