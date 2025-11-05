use super::super::board::Board;
use super::super::color::{get_reverse_color, ColorType};
use super::super::evaluator::abst::Evaluator;
use super::search_strategy::{self, EvaluationResult, SearchStrategy};
use pyo3::prelude::*;

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
        evaluator: Option<&dyn Evaluator>,
    ) -> EvaluationResult {
        let strategy = self;
        let search_depth = depth;
        search_strategy::search_helper(
            board,
            color,
            evaluator,
            move |board, color, evaluator, moves| {
                let mut best_score = f32::NEG_INFINITY;
                let mut best_move = moves[0].clone();
                let mut nodes = 0u64;

                for mv in &moves {
                    let mut new_board = board.clone();
                    new_board.execute_move(mv);
                    let score = -strategy.minmax(
                        &new_board,
                        get_reverse_color(color),
                        search_depth.saturating_sub(1),
                        &mut nodes,
                        evaluator,
                    );

                    if score > best_score {
                        best_score = score;
                        best_move = mv.clone();
                    }
                }

                EvaluationResult {
                    score: best_score,
                    best_move: Some(best_move),
                    nodes_searched: nodes,
                }
            },
        )
    }
}
