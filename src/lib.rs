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
#[path = "rustshogi/nn_model.rs"]
pub mod nn_model;
#[path = "rustshogi/pca.rs"]
pub mod pca;
#[path = "rustshogi/piece.rs"]
pub mod piece;
#[path = "rustshogi/random.rs"]
pub mod random;
#[path = "rustshogi/search/mod.rs"]
pub mod search;

#[cfg(test)]
#[path = "rustshogi/test_address.rs"]
pub mod test_address;
#[cfg(test)]
#[path = "rustshogi/test_bitboard.rs"]
pub mod test_bitboard;
#[cfg(test)]
#[path = "rustshogi/test_board.rs"]
pub mod test_board;
#[cfg(test)]
#[path = "rustshogi/test_board_pca.rs"]
pub mod test_board_pca;
#[cfg(test)]
#[path = "rustshogi/test_color.rs"]
pub mod test_color;
#[cfg(test)]
#[path = "rustshogi/test_common.rs"]
pub mod test_common;
#[cfg(test)]
#[path = "rustshogi/test_direction.rs"]
pub mod test_direction;
#[cfg(test)]
#[path = "rustshogi/test_evaluator.rs"]
pub mod test_evaluator;
#[cfg(test)]
#[path = "rustshogi/test_game.rs"]
pub mod test_game;
#[cfg(test)]
#[path = "rustshogi/test_hand.rs"]
pub mod test_hand;
#[cfg(test)]
#[path = "rustshogi/test_moves.rs"]
pub mod test_moves;
#[cfg(test)]
#[path = "rustshogi/test_nn_model.rs"]
pub mod test_nn_model;
#[cfg(test)]
#[path = "rustshogi/test_pca.rs"]
pub mod test_pca;
#[cfg(test)]
#[path = "rustshogi/test_piece.rs"]
pub mod test_piece;
#[cfg(test)]
#[path = "rustshogi/test_random.rs"]
pub mod test_random;
#[cfg(test)]
#[path = "rustshogi/test_search.rs"]
pub mod test_search;

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
    m.add_class::<search::engine::SearchEngine>()?;
    Ok(())
}
