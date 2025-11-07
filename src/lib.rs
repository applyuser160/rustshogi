#[path = "rustshogi/address.rs"]
pub mod address;
#[path = "rustshogi/bitboard.rs"]
pub mod bitboard;
#[path = "rustshogi/board.rs"]
pub mod board;
#[path = "rustshogi/color.rs"]
pub mod color;
#[path = "rustshogi/common.rs"]
pub mod common;
#[path = "rustshogi/direction.rs"]
pub mod direction;
#[path = "rustshogi/evaluator/mod.rs"]
pub mod evaluator;
#[path = "rustshogi/game.rs"]
pub mod game;
#[path = "rustshogi/hand.rs"]
pub mod hand;
#[path = "rustshogi/mctsresult.rs"]
pub mod mctsresult;
#[path = "rustshogi/move_pattern.rs"]
pub mod move_pattern;
#[path = "rustshogi/moves.rs"]
pub mod moves;
#[path = "rustshogi/pca.rs"]
pub mod pca;
#[path = "rustshogi/piece.rs"]
pub mod piece;
#[path = "rustshogi/random.rs"]
pub mod random;
#[path = "rustshogi/search/mod.rs"]
pub mod search;

use pyo3::prelude::*;

#[pymodule]
fn rustshogi(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<address::Address>()?;
    m.add_class::<color::ColorType>()?;
    m.add_class::<piece::PieceType>()?;
    m.add_class::<piece::Piece>()?;
    m.add_class::<moves::Move>()?;
    m.add_class::<hand::Hand>()?;
    m.add_class::<board::Board>()?;
    m.add_class::<game::Game>()?;
    m.add_class::<mctsresult::MctsResult>()?;
    m.add_class::<evaluator::neural::NeuralEvaluator>()?;
    m.add_class::<evaluator::simple::SimpleEvaluator>()?;
    m.add_class::<search::engine::SearchEngine>()?;
    m.add_class::<search::search_strategy::EvaluationResult>()?;
    m.add_class::<search::alphabeta::AlphaBetaSearchStrategy>()?;
    m.add_class::<search::minmax::MinMaxSearchStrategy>()?;
    Ok(())
}
