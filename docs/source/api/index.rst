API リファレンス
================

rustshogiの完全なAPIリファレンスです。

.. toctree::
   :maxdepth: 2

   ../reference/rustshogi

モジュール概要
=============

rustshogiは以下の主要なクラスと列挙型で構成されています：

* :doc:`../reference/rustshogi` - メインモジュール（将棋盤、手、駒などの基本機能）

基本的な型
==========

.. py:class:: Address
   :module: rustshogi

   将棋盤上の座標を表現するクラス。列（column）と行（row）の情報を含みます。

.. py:class:: ColorType
   :module: rustshogi

   先手・後手を表現する列挙型。Black（先手）とWhite（後手）の値を持ちます。

.. py:class:: PieceType
   :module: rustshogi

   将棋の駒の種類を表現する列挙型。King、Gold、Rook、Bishop、Silver、Knight、Lance、Pawn、および成り駒の種類を含みます。

.. py:class:: Piece
   :module: rustshogi

   将棋の駒を表現するクラス。駒の種類（PieceType）と色（ColorType）の情報を含みます。

.. py:class:: Move
   :module: rustshogi

   将棋の手を表現するクラス。移動元、移動先、駒の種類、成りなどの情報を含みます。

.. py:class:: Hand
   :module: rustshogi.. py:class:: Board
   :module: rustshogi

   将棋盤を表現するクラス。局面の状態、合法手の生成、手の実行などの機能を提供します。

.. py:class:: Game
   :module: rustshogi

   ゲーム全体を管理するクラス。対局の進行、勝敗判定、ランダム対局などを担当します。

評価関数
========

rustshogiは、局面を評価するための複数の評価関数を提供します。

.. py:class:: SimpleEvaluator
   :module: rustshogi

   簡易評価関数。駒の価値のみを使用して局面を評価します。

   .. py:method:: evaluate(board: Board, color: ColorType) -> float

      盤面を評価します。

      :param board: 評価する盤面
      :param color: 評価するプレイヤーの色（先手または後手）
      :returns: 評価値（colorの視点での評価、大きい方が有利）

.. py:class:: NeuralEvaluator
   :module: rustshogi

   ニューラルネットワーク評価関数。機械学習モデルを使用して局面の勝率を予測し、評価値を計算します。

   .. py:method:: __init__(db_type_str: Optional[str] = None, connection_string: Optional[str] = None, model_path: Optional[str] = None)

      評価関数を初期化します。

      :param db_type_str: データベースタイプ（"sqlite" または "postgres"）
      :param connection_string: データベース接続文字列
      :param model_path: モデルファイルのパス

   .. py:method:: init_database() -> None

      データベーステーブルを初期化します。

   .. py:method:: evaluate(board: Board, color: ColorType) -> float

      盤面を評価します。

      :param board: 評価する盤面
      :param color: 評価するプレイヤーの色
      :returns: 評価値

   .. py:method:: evaluate_position(board: Board, model_path: Optional[str] = None) -> Tuple[float, float, float]

      特定の局面の勝率を予測します。

      :param board: 評価する盤面
      :param model_path: モデルファイルのパス（オプション）
      :returns: (白勝率, 黒勝率, 引き分け率) のタプル

   .. py:method:: generate_and_save_random_boards(count: int) -> int

      ランダム盤面を生成してデータベースに保存します。

      :param count: 生成する盤面の数
      :returns: 保存された盤面の数

探索
====

rustshogiは、最善手を探索するための探索エンジンと探索アルゴリズムを提供します。

.. py:class:: SearchEngine
   :module: rustshogi

   探索エンジン。探索アルゴリズムと評価関数を組み合わせて最善手を探索します。

   .. py:method:: __init__(algorithm: str = "minmax", max_nodes: int = 1000000, evaluator: Optional[Evaluator] = None)

      探索エンジンを初期化します。

      :param algorithm: 探索アルゴリズム（"minmax" または "alphabeta"）
      :param max_nodes: 最大探索ノード数
      :param evaluator: 評価関数（オプション、デフォルトはSimpleEvaluator）

   .. py:method:: search(board: Board, color: ColorType, depth: int) -> EvaluationResult

      探索を実行して最善手を見つけます。

      :param board: 現在の盤面
      :param color: 手番の色
      :param depth: 探索深度
      :returns: 評価結果（評価値、最善手、探索ノード数を含む）

.. py:class:: EvaluationResult
   :module: rustshogi

   探索結果を表すクラス。

   .. py:attribute:: score

      評価値（float型）

   .. py:attribute:: best_move

      最善手（Optional[Move]型）

   .. py:attribute:: nodes_searched

      探索されたノード数（int型）

.. py:class:: MinMaxSearchStrategy
   :module: rustshogi

   ミニマックス探索アルゴリズム。すべての可能な手を探索し、最適な手を選択します。

   .. py:method:: __init__(max_nodes: int = 100)

      ミニマックス探索.. py:class:: AlphaBetaSearchStrategy
   :module: rustshogi

   アルファベータ探索アルゴリズム。ミニマックス探索を最適化した探索アルゴリズムで、不要なノードの探索をスキップします。

   .. py:method:: __init__(max_nodes: int = 100)

      アルファベータ探索戦略を初期化します。

      :param max_nodes: 最大探索ノード数i

   ゲーム全体を管理するクラス。対局の進行、勝敗判定、ランダム対局などを担当します。
