use super::super::board::Board;
use super::super::color::ColorType;

/// Trait for evaluation functions
/// Each evaluation function can be used in the search engine by implementing this trait
pub trait Evaluator: Send + Sync {
    /// Evaluates a board position
    ///
    /// # Arguments
    /// * `board` - The board position to evaluate
    /// * `color` - The color of the player to evaluate for
    ///
    /// # Returns
    /// Evaluation value (from color's perspective, higher is better)
    fn evaluate(&self, board: &Board, color: ColorType) -> f32;
}
