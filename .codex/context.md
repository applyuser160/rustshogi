# RustShogi - Context for Codex

## Project Overview
RustShogi is a high-performance Shogi (Japanese chess) library implemented in Rust with Python bindings. It provides efficient board representation, move generation, game state management, and AI search capabilities.

## Architecture
- **Core Library**: Rust implementation with optimized bitboard representation
- **Python Bindings**: PyO3-based Python interface
- **Search Engine**: Multiple algorithms (Minimax, AlphaBeta)
- **Evaluation System**: Neural network and simple evaluators
- **Database Integration**: SQLite and PostgreSQL support

## Key Components

### Board Representation
- Uses bitboards for efficient piece position tracking
- 9x9 board with standard Shogi pieces
- Support for piece promotion and drops

### Move System
- Move representation using u16 for memory efficiency
- Support for normal moves, promotions, and piece drops
- CSA protocol compatibility

### Search Algorithms
- Minimax with alpha-beta pruning
- Parallel search using Rayon
- Configurable search depth and time limits

### Evaluation Functions
- Neural network evaluator with ML models
- Simple material-based evaluation
- Database-driven evaluation with position lookup

## File Structure
```
src/rustshogi/
├── lib.rs              # Main Python module exports
├── board.rs            # Board state and operations
├── moves.rs            # Move representation and validation
├── game.rs             # Game flow and state management
├── search/             # Search algorithms
│   ├── engine.rs       # Search engine core
│   ├── alphabeta.rs    # Alpha-beta implementation
│   └── minmax.rs       # Minimax implementation
└── evaluator/          # Position evaluation
    ├── neural.rs       # Neural network evaluator
    ├── simple.rs       # Simple material evaluator
    └── database.rs     # Database-based evaluator
```

## Dependencies
- **pyo3**: Python bindings
- **rayon**: Parallel processing
- **burn**: ML framework for neural evaluation
- **ndarray/nalgebra**: Numerical computing
- **serde**: Serialization
- **rusqlite/tokio-postgres**: Database connectivity

## Usage Patterns
- Game creation and management
- Move generation and validation
- Position evaluation
- AI opponent implementation
- Game analysis and study

## Performance Considerations
- Bitboard operations for fast move generation
- Memory-efficient move encoding
- Parallel search capabilities
- SIMD optimizations where applicable
