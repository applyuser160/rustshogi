use super::super::board::Board;
use super::super::color::{get_reverse_color, ColorType};
use super::super::evaluator::abst::Evaluator;
use super::super::moves::Move;
use super::search_strategy::{self, EvaluationResult, SearchStrategy};
use pyo3::prelude::*;
use std::cmp::Ordering;

/// AlphaBeta search algorithm
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

        let moves: Vec<Move> = board.search_moves(color, true);
        if moves.is_empty() {
            return evaluator.evaluate(board, color);
        }

        let mut alpha: f32 = alpha;
        let mut best_score: f32 = f32::NEG_INFINITY;

        for mv in &moves {
            let mut new_board = board.clone();
            new_board.execute_move(mv);
            let score: f32 = -self.alphabeta(
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
        limit: Option<usize>,
        evaluator: Option<&dyn Evaluator>,
    ) -> Vec<EvaluationResult> {
        let strategy: &AlphaBetaSearchStrategy = self;
        let search_depth: u8 = depth;
        search_strategy::search_helper(
            board,
            color,
            evaluator,
            limit,
            move |board, color, evaluator, moves, limit| {
                let mut nodes: u64 = 0u64;
                let mut alpha: f32 = f32::NEG_INFINITY;
                let beta: f32 = f32::INFINITY;
                let mut evaluations: Vec<EvaluationResult> = Vec::with_capacity(moves.len());

                for mv in &moves {
                    let mut new_board: Board = board.clone();
                    new_board.execute_move(mv);
                    let nodes_before: u64 = nodes;
                    let score: f32 = -strategy.alphabeta(
                        &new_board,
                        get_reverse_color(color),
                        search_depth.saturating_sub(1),
                        -beta,
                        -alpha,
                        &mut nodes,
                        evaluator,
                    );

                    let nodes_searched: u64 = nodes.saturating_sub(nodes_before);
                    evaluations.push(EvaluationResult {
                        score,
                        best_move: Some(mv.clone()),
                        nodes_searched,
                    });

                    alpha = alpha.max(score);
                    if beta <= alpha {
                        break; // Beta cutoff
                    }
                }

                evaluations
                    .sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));

                if let Some(limit) = limit {
                    evaluations.truncate(limit.min(evaluations.len()));
                }

                evaluations
            },
        )
    }
}
