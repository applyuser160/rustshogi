use super::super::board::Board;
use super::super::color::{get_reverse_color, ColorType};
use super::super::evaluator::abst::Evaluator;
use super::search_strategy::{self, EvaluationResult, SearchStrategy};
use pyo3::prelude::*;
use std::cmp::Ordering;

/// MinMax search algorithm
#[pyclass]
pub struct MinMaxSearchStrategy {
    max_nodes: u64,
}

impl MinMaxSearchStrategy {
    pub fn new(max_nodes: u64) -> Self {
        Self { max_nodes }
    }

    fn minmax(
        &self,
        board: &Board,
        color: ColorType,
        depth: u8,
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

        let mut best_score = f32::NEG_INFINITY;

        for mv in &moves {
            let mut new_board = board.clone();
            new_board.execute_move(mv);
            let score = -self.minmax(
                &new_board,
                get_reverse_color(color),
                depth - 1,
                nodes,
                evaluator,
            );
            best_score = best_score.max(score);
        }

        best_score
    }
}

#[pymethods]
impl MinMaxSearchStrategy {
    #[new]
    pub fn new_for_python(max_nodes: u64) -> Self {
        Self::new(max_nodes)
    }
}

impl SearchStrategy for MinMaxSearchStrategy {
    fn search(
        &self,
        board: &Board,
        color: ColorType,
        depth: u8,
        limit: Option<usize>,
        evaluator: Option<&dyn Evaluator>,
    ) -> Vec<EvaluationResult> {
        let strategy = self;
        let search_depth = depth;
        search_strategy::search_helper(
            board,
            color,
            evaluator,
            limit,
            move |board, color, evaluator, moves, limit| {
                let mut nodes = 0u64;
                let mut evaluations: Vec<EvaluationResult> = moves
                    .iter()
                    .map(|mv| {
                        let mut new_board = board.clone();
                        new_board.execute_move(mv);
                        let nodes_before = nodes;
                        let score = -strategy.minmax(
                            &new_board,
                            get_reverse_color(color),
                            search_depth.saturating_sub(1),
                            &mut nodes,
                            evaluator,
                        );
                        let nodes_searched = nodes.saturating_sub(nodes_before);
                        EvaluationResult {
                            score,
                            best_move: Some(mv.clone()),
                            nodes_searched,
                        }
                    })
                    .collect();

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
