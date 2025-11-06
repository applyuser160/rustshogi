# RustShogi プロジェクトコンテキスト

## プロジェクト概要
RustShogiはRustで実装された高性能な将棋ライブラリです。Pythonバインディングも提供しており、Pythonからも利用可能です。

## 技術スタック
- **メイン言語**: Rust (Edition 2021)
- **バインディング**: Python (PyO3使用)
- **ビルドシステム**: Cargo (Rust), Maturin (Python)

## 主要機能
- ⚡ 高性能なRust実装
- 🐍 Pythonバインディング対応
- 💾 メモリ効率の良いデータ構造
- ✅ 完全な将棋ルール実装
- 🧠 ニューラルネットワーク評価機能
- 🔍 アルゴリズム探索（Minimax, AlphaBetaなど）

## パフォーマンス最適化
- ビットボードを使用した効率的な盤面表現
- メモリ効率の良いデータ構造（u16 Move表現など）
- SIMD命令の活用
- ゼロコスト抽象化

## プロジェクト構成
```
rustshogi/
├── src/
│   ├── rustshogi/
│   │   ├── address.rs      # 座標操作
│   │   ├── bitboard.rs     # ビットボード実装
│   │   ├── board.rs        # 盤面管理
│   │   ├── color.rs        # 色定義
│   │   ├── piece.rs        # 駒の定義と操作
│   │   ├── moves.rs        # 指手の表現と操作
│   │   ├── game.rs         # ゲーム進行管理
│   │   ├── hand.rs         # 持ち駒管理
│   │   ├── search/         # 探索アルゴリズム
│   │   │   ├── engine.rs   # 探索エンジン
│   │   │   ├── alphabeta.rs # AlphaBeta探索
│   │   │   └── minmax.rs   # Minimax探索
│   │   └── evaluator/      # 評価関数
│   │       ├── neural.rs   # ニューラルネット評価
│   │       ├── simple.rs   # シンプル評価
│   │       └── database.rs # データベース評価
│   └── lib.rs              # Pythonモジュール定義
├── tests/                  # テストコード
├── benches/                # ベンチマーク
├── docs/                   # ドキュメント
├── Cargo.toml             # Rust依存関係
└── pyproject.toml         # Python設定
```

## 主要依存関係
- **rand**: 乱数生成
- **pyo3**: Pythonバインディング
- **rayon**: 並列処理
- **burn**: 機械学習フレームワーク
- **ndarray/nalgebra**: 数値計算
- **serde**: シリアライズ
- **rusqlite/tokio-postgres**: データベース接続

## 開発環境
- Python 3.8+
- Rust 2021 Edition
- Maturin (Pythonビルド)
- Sphinx (ドキュメント生成)

## ライセンス
MIT License

## ドキュメント
詳細なドキュメント: https://applyuser160.github.io/rustshogi/

## PyPI
パッケージ名: `rustshogi`
