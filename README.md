# 🎌 RustShogi

<p align="center">
  <a href="https://github.com/applyuser160/rustshogi/actions/workflows/CI.yml">
    <img src="https://github.com/applyuser160/rustshogi/actions/workflows/CI.yml/badge.svg" alt="CI">
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT">
  </a>
  <img src="https://img.shields.io/badge/Rust-2021-orange.svg" alt="Rust 2021">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python 3.8+">
</p>

<div align="center">

```text
      ☗
┌─────────┐
│   玉    │
│         │
└─────────┘
```

</div>

> Rustで実装された高速な将棋ライブラリ

---

## 🚀 特徴

- ⚡ 高速なRust実装
- 🐍 Pythonバインディング対応
- 💾 メモリ効率的なデータ構造
- ✅ 完全な将棋ルール実装

---

## 📦 インストール

```bash
pip install rustshogi
```

---

## 💻 クイックスタート

```python
import rustshogi

# Create a new game
game = rustshogi.Game()

# Display board state
print(game.board)

# Search for possible moves
moves = game.board.search_moves(rustshogi.ColorType.Black)
print(f"Number of possible moves: {len(moves)}")

# Execute a move
if moves:
    game.execute_move(moves[0])
    print("Move executed")
    print(game.board)

# Check game end
is_finished, winner = game.is_finished()
if is_finished:
    print(f"Game ended: Winner = {winner}")
```

<details>
<summary>📚 詳細な使い方を見る</summary>

### アドレス操作

```python
# Create an address
address = rustshogi.Address(3, 4)  # Column 3, Row 4
print(f"Address: {address}")

# Create address from string
address = rustshogi.Address.from_string("3d")
print(f"Address: {address}")

# Convert to index
index = address.to_index()
print(f"Index: {index}")
```

### 駒の操作

```python
# Create a piece
piece = rustshogi.Piece(rustshogi.ColorType.Black, rustshogi.PieceType.King)
print(f"Piece: {piece}")

# Create piece from character
piece = rustshogi.Piece.from_char('K')  # Black King
print(f"Piece: {piece}")

piece = rustshogi.Piece.from_char('p')  # White Pawn
print(f"Piece: {piece}")
```

### 指し手操作

```python
# Create a normal move
from_addr = rustshogi.Address(3, 3)
to_addr = rustshogi.Address(3, 4)
move = rustshogi.Move(from_address=from_addr, to_address=to_addr, promote=False)
print(f"Move: {move}")

# Create a drop move
piece = rustshogi.Piece.from_char('p')
to_addr = rustshogi.Address(3, 4)
drop_move = rustshogi.Move(piece=piece, to_address=to_addr)
print(f"Drop move: {drop_move}")

# Create move from CSA format
csa_move = rustshogi.Move(csa="3c3d")
print(f"CSA move: {csa_move}")
```

### 盤面操作

```python
# Create a new board
board = rustshogi.Board()

# Set initial position
board.startpos()
print("Initial position:")
print(board)

# Deploy a piece
piece = rustshogi.Piece(rustshogi.ColorType.Black, rustshogi.PieceType.King)
address = rustshogi.Address(5, 1)
board.deploy(address, piece.piece_type, piece.owner)

# Get piece at specific position
piece_at_pos = board.get_piece(address)
print(f"Piece at {address}: {piece_at_pos}")

# Search for possible moves
moves = board.search_moves(rustshogi.ColorType.Black)
print(f"Black's possible moves: {len(moves)} moves")

# Execute a move
if moves:
    board.execute_move(moves[0])
    print("Board after move execution:")
    print(board)
```

### ゲーム進行

```python
# Create a game
game = rustshogi.Game()

# Random play
result = game.random_play()
print(f"Random play result: Winner = {result.winner}")

# Manual game progression
game = rustshogi.Game()
while not game.is_finished()[0]:
    moves = game.board.search_moves(game.turn)
    if moves:
        # Select the first move
        game.execute_move(moves[0])
    else:
        break

is_finished, winner = game.is_finished()
print(f"Game ended: Winner = {winner}")
```

### データ構造

#### ColorType

```python
rustshogi.ColorType.Black    # Sente (First player)
rustshogi.ColorType.White    # Gote (Second player)
rustshogi.ColorType.None     # None
```

#### PieceType

```python
rustshogi.PieceType.King      # King
rustshogi.PieceType.Gold      # Gold General
rustshogi.PieceType.Rook      # Rook
rustshogi.PieceType.Bichop    # Bishop
rustshogi.PieceType.Silver    # Silver General
rustshogi.PieceType.Knight    # Knight
rustshogi.PieceType.Lance     # Lance
rustshogi.PieceType.Pawn      # Pawn
# Promoted pieces
rustshogi.PieceType.Dragon    # Dragon King
rustshogi.PieceType.Horse     # Dragon Horse
rustshogi.PieceType.ProSilver # Promoted Silver
rustshogi.PieceType.ProKnight # Promoted Knight
rustshogi.PieceType.ProLance  # Promoted Lance
rustshogi.PieceType.ProPawn   # Tokin
```

</details>

---

## 📖 ドキュメント

詳細なドキュメントはこちら: https://applyuser160.github.io/rustshogi/

---

## ⚡ パフォーマンス

このライブラリは、以下の最適化により高速な処理を実現しています。

- ビットボードによる効率的な盤面表現
- メモリ効率の良いデータ構造（u16での指し手表現など）
- SIMD命令の活用
- ゼロコスト抽象化

---

## 📄 ライセンス

このプロジェクトはMITライセンスの下で公開されています。

---

## 🤝 貢献

プルリクエストやイシュー報告を歓迎します。開発に参加したい場合は、まずイシューを作成してご連絡ください。
