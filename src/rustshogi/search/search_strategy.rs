use super::super::board::Board;
use super::super::color::ColorType;
use super::super::evaluator::abst::Evaluator;
use super::super::evaluator::simple::SimpleEvaluator;
use super::super::moves::Move;
use pyo3::prelude::*;

/// Structure representing the evaluation result
#[derive(Debug, Clone)]
#[pyclass]
pub struct EvaluationResult {
    #[pyo3(get, set)]
    pub score: f32,
    #[pyo3(get, set)]
    pub best_move: Option<super::super::moves::Move>,
    #[pyo3(get, set)]
    pub nodes_searched: u64,
}

/// Trait for search strategies
pub trait SearchStrategy {
    /// Execute a search
    ///
    /// # Arguments
    /// * `board` - The current board state
    /// * `color` - The color of the current player
    /// * `depth` - The search depth
    /// * `evaluator` - The evaluation function (optional)
    ///
    /// # Returns
    /// The evaluation result
    fn search(
        &self,
        board: &Board,
        color: ColorType,
        depth: u8,
        evaluator: Option<&dyn Evaluator>,
    ) -> EvaluationResult;
}

/// Get the default evaluation function
pub fn get_default_evaluator() -> SimpleEvaluator {
    SimpleEvaluator::new()
}

/// Handle the case where there are no moves
pub fn handle_no_moves(
    evaluator: &dyn Evaluator,
    board: &Board,
    color: ColorType,
) -> EvaluationResult {
    EvaluationResult {
        score: evaluator.evaluate(board, color),
        best_move: None,
        nodes_searched: 1,
    }
}

/// Common initialization process for search
/// A helper function to prepare the evaluation function, get moves, initialize, and execute the search
pub fn search_helper<F>(
    board: &Board,
    color: ColorType,
    evaluator: Option<&dyn Evaluator>,
    search_fn: F,
) -> EvaluationResult
where
    F: FnOnce(&Board, ColorType, &dyn Evaluator, Vec<Move>) -> EvaluationResult,
{
    let default_evaluator;
    let evaluator_ref = match evaluator {
        Some(e) => e,
        None => {
            default_evaluator = get_default_evaluator();
            &default_evaluator as &dyn Evaluator
        }
    };

    let moves = board.search_moves(color, true);
    if moves.is_empty() {
        return handle_no_moves(evaluator_ref, board, color);
    }

    search_fn(board, color, evaluator_ref, moves)
}
