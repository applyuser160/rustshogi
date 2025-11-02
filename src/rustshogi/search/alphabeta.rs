use super::super::board::Board;
use super::super::color::{get_reverse_color, ColorType};
use super::super::evaluator::abst::Evaluator;
use super::search_strategy::EvaluationResult;
use super::search_strategy::SearchStrategy;
use pyo3::prelude::*;

/// AlphaBeta探索アルゴリズム
#[pyclass]
pub struct AlphaBetaSearchStrategy {
    #[pyo3(get, set)]
    pub max_nodes: u64,
}

impl AlphaBetaSearchStrategy {
    pub fn new(max_nodes: u64) -> Self {
        Self { max_nodes }
    }

    fn alphabeta(
        &self,
        board: &Board,
        color: ColorType,
        depth: u8,
        alpha: f32,
        beta: f32,
        nodes: &mut u64,
        evaluator: &dyn Evaluator,
    ) -> f32 {
        *nodes += 1;

        if depth == 0 || *nodes > self.max_nodes {
            return evaluator.evaluate(board, color);
        }

        let moves = board.search_moves(color, true);
        if moves.is_empty() {
            return evaluator.evaluate(board, color);
        }

        let mut alpha = alpha;
        let mut best_score = f32::NEG_INFINITY;

        for mv in &moves {
            let mut new_board = board.clone();
            new_board.execute_move(mv);
            let score = -self.alphabeta(
                &new_board,
                get_reverse_color(color),
                depth - 1,
                -beta,
                -alpha,
                nodes,
                evaluator,
            );

            best_score = best_score.max(score);
            alpha = alpha.max(score);

            if beta <= alpha {
                break; // Beta cutoff
            }
        }

        best_score
    }
}

#[pymethods]
impl AlphaBetaSearchStrategy {
    #[new]
    #[pyo3(signature = (max_nodes=100))]
    pub fn new_for_python(max_nodes: u64) -> Self {
        Self::new(max_nodes)
    }
}

impl SearchStrategy for AlphaBetaSearchStrategy {
    fn search(
        &self,
        board: &Board,
        color: ColorType,
        depth: u8,
        evaluator: Option<&dyn Evaluator>,
    ) -> EvaluationResult {
        let default_evaluator = super::super::evaluator::simple::SimpleEvaluator::new();
        let evaluator = evaluator.unwrap_or(&default_evaluator as &dyn Evaluator);

        let moves = board.search_moves(color, true);
        if moves.is_empty() {
            return EvaluationResult {
                score: evaluator.evaluate(board, color),
                best_move: None,
                nodes_searched: 1,
            };
        }

        let mut best_score = f32::NEG_INFINITY;
        let mut best_move = moves[0].clone();
        let mut nodes = 0u64;
        let mut alpha = f32::NEG_INFINITY;
        let beta = f32::INFINITY;

        for mv in &moves {
            let mut new_board = board.clone();
            new_board.execute_move(mv);
            let score = -self.alphabeta(
                &new_board,
                get_reverse_color(color),
                depth.saturating_sub(1),
                -beta,
                -alpha,
                &mut nodes,
                evaluator,
            );

            if score > best_score {
                best_score = score;
                best_move = mv.clone();
            }

            alpha = alpha.max(score);
            if beta <= alpha {
                break;
            }
        }

        EvaluationResult {
            score: best_score,
            best_move: Some(best_move),
            nodes_searched: nodes,
        }
    }
}
