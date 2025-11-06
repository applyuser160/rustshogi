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

## Coding Guidelines

### Naming (Rust)
- Types/Traits: PascalCase (e.g., `Board`, `ColorType`, `NeuralEvaluator`)
- Functions/Methods/Vars: snake_case (e.g., `execute_move`, `search_moves`)
- Constants: UPPER_SNAKE_CASE (e.g., `MOVE_CACHE`, `CACHE_SIZE`)
- Type aliases: PascalCase

### Python Bindings
- Use `#[pyclass]` for classes and `#[pymethods]` for methods
- Expose properties with `#[pyo3(get, set)]`
- Adjust Python names via `#[pyo3(name = "...")]` when needed

### Module/Imports Order
1. Relative (`crate::`, `super::`)
2. External crates
3. Standard library
4. `pyo3::prelude::*` (when exposing to Python)

### Documentation
- Public API uses `///`, module-level uses `//!`
- Describe purpose, params, returns, and errors concisely

### Error Handling
- Internal: `Result<T, Box<dyn std::error::Error + Send + Sync>>`
- Python: `PyResult<T>` / `Result<T, PyErr>`
- Provide clear, contextual messages

### Performance
- Use global/LRU caches (e.g., `const CACHE_SIZE: usize = 70000;`, `MOVE_CACHE`)
- Parallelize with rayon (`par_iter`), use `num_cpus::get()` for threads
- Optimize memory (`u16` move encoding), minimize clones, prefer bit ops

### Formatting & Linting
- Run `cargo fmt` and `cargo clippy`; pre-commit hooks include compile/fmt/clippy/trailing-whitespace/EOF-fixer

### Type Safety
- Prefer explicit conversions (`From`/`Into`/`TryFrom`); enums may use `#[repr(usize)]` and `strum`

### Tests & Benchmarks
- Unit tests under `tests/` as `test_<module>.rs`
- Benchmarks under `benches/` with `criterion`

### Comments & Organization
- Add inline comments for complex logic/bit ops
- Replace magic numbers with constants (e.g., `PROMOTE`, `PIECE_TYPE_NUMBER`)
- Prefer exhaustive `match`; use references to avoid unnecessary moves/clones
