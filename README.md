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
  <a href="https://pepy.tech/projects/rustshogi"><img src="https://static.pepy.tech/personalized-badge/rustshogi?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=BLUE&left_text=downloads" alt="PyPI Downloads"></a>
</p>

<img src="logo.png">

> A high-performance Shogi library implemented in Rust.

---

## 🚀 Features

- ⚡ High-performance Rust implementation
- 🐍 Python bindings support
- 💾 Memory-efficient data structures
- ✅ Complete Shogi rule implementation

---

## 📦 Installation

```bash
pip install rustshogi
```

---

## 💻 Quick Start

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
<summary>📚 See Detailed Usage</summary>

### Address Operations

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

### Piece Operations

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

### Move Operations

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

### Board Operations

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

### Game Progression

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

### Data Structures

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
rustshogi.PieceType.Bishop    # Bishop
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

## 📖 Documentation

See the detailed documentation here: https://applyuser160.github.io/rustshogi/

---

## ⚡ Performance

This library achieves high-speed processing through the following optimizations:

- Efficient board representation using bitboards
- Memory-efficient data structures (e.g., u16 Move representation)
- Utilization of SIMD instructions
- Zero-cost abstractions

---

## 📄 License

This project is released under the MIT License.

---

## 🤝 Contributing

Pull requests and issue reports are welcome. If you want to participate in development, please create an issue first and contact us.
