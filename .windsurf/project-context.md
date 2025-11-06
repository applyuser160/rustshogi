# RustShogi Project Context

## Project Overview
RustShogi is a high-performance Shogi library implemented in Rust. It also provides Python bindings and can be used from Python.

## Technology Stack
- **Main Language**: Rust (Edition 2021)
- **Bindings**: Python (using PyO3)
- **Build System**: Cargo (Rust), Maturin (Python)

## Key Features
- ⚡ High-performance Rust implementation
- 🐍 Python bindings support
- 💾 Memory-efficient data structures
- ✅ Complete Shogi rule implementation
- 🧠 Neural network evaluation functionality
- 🔍 Search algorithms (Minimax, AlphaBeta, etc.)

## Performance Optimizations
- Efficient board representation using bitboards
- Memory-efficient data structures (e.g., u16 Move representation)
- Utilization of SIMD instructions
- Zero-cost abstractions

## Project Structure
```
rustshogi/
├── src/
│   ├── rustshogi/
│   │   ├── address.rs      # Coordinate operations
│   │   ├── bitboard.rs     # Bitboard implementation
│   │   ├── board.rs        # Board management
│   │   ├── color.rs        # Color definitions
│   │   ├── piece.rs        # Piece definitions and operations
│   │   ├── moves.rs        # Move representation and operations
│   │   ├── game.rs         # Game progression management
│   │   ├── hand.rs         # Hand piece management
│   │   ├── search/         # Search algorithms
│   │   │   ├── engine.rs   # Search engine
│   │   │   ├── alphabeta.rs # AlphaBeta search
│   │   │   └── minmax.rs   # Minimax search
│   │   └── evaluator/      # Evaluation functions
│   │       ├── neural.rs   # Neural network evaluation
│   │       ├── simple.rs   # Simple evaluation
│   │       └── database.rs # Database evaluation
│   └── lib.rs              # Python module definition
├── tests/                  # Test code
├── benches/                # Benchmarks
├── docs/                   # Documentation
├── Cargo.toml             # Rust dependencies
└── pyproject.toml         # Python configuration
```

## Key Dependencies
- **rand**: Random number generation
- **pyo3**: Python bindings
- **rayon**: Parallel processing
- **burn**: Machine learning framework
- **ndarray/nalgebra**: Numerical computing
- **serde**: Serialization
- **rusqlite/tokio-postgres**: Database connections

## Development Environment
- Python 3.8+
- Rust 2021 Edition
- Maturin (Python build)
- Sphinx (Documentation generation)

## License
MIT License

## Documentation
Detailed documentation: https://applyuser160.github.io/rustshogi/

## PyPI
Package name: `rustshogi`

## Coding Conventions

### Naming Conventions (Rust)
- Structs, Enums, Traits: PascalCase (e.g., `Board`, `ColorType`, `NeuralEvaluator`)
- Functions, Methods, Variables: snake_case (e.g., `execute_move`, `search_moves`)
- Constants: UPPER_SNAKE_CASE (e.g., `MOVE_CACHE`, `CACHE_SIZE`)
- Type Aliases: PascalCase

### Python Bindings
- Classes exposed with `#[pyclass]`, methods with `#[pymethods]`
- Use `#[pyo3(name = "...")]` to adjust Python public names
- Provide property access with `#[pyo3(get, set)]`

### Module Structure and Import Order
1. Relative imports from `crate::`/`super::`
2. External crates
3. Standard library
4. `pyo3::prelude::*`

### Documentation Comments
- Use `///` for public APIs, `//!` for modules
- Briefly describe purpose, parameters, return values, and errors

### Error Handling
- Internal: `Result<T, Box<dyn std::error::Error + Send + Sync>>`
- Python: `PyResult<T>` / `Result<T, PyErr>`
- Provide context-aware error messages

### Performance Optimization
- Use global/LRU caches (e.g., `const CACHE_SIZE: usize = 70000;`, `MOVE_CACHE`)
- Parallelization: rayon (`par_iter`), thread count via `num_cpus::get()`
- Memory efficiency: `u16` for Move representation, avoid unnecessary clones, leverage bit operations

### Formatting and Linting
- Auto-format with `cargo fmt`, lint with `cargo clippy`
- Pre-commit: compile check / fmt / clippy / trailing-whitespace / end-of-file-fixer

### Type Safety
- Leverage explicit conversions (`From`/`Into`/`TryFrom`)
- Enumerations provide utilities with `#[repr(usize)]` and `strum`

### Tests/Benchmarks
- Unit tests in `tests/test_<module>.rs`
- Benchmarks in `benches/` using `criterion`

### Comments and Code Organization
- Inline comments for complex logic and bit operations
- Convert magic numbers to constants (e.g., `PROMOTE`, `PIECE_TYPE_NUMBER`)
- Use `match` for exhaustive branching, prefer references over ownership/borrowing
