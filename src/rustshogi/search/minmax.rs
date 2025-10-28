use super::super::board::Board;
use super::super::color::{get_reverse_color, ColorType};
use super::super::evaluator::trait_::Evaluator;
use super::super::search_strategy::{EvaluationResult, SearchStrategy};

/// MinMax探索アルゴリズム
pub struct MinMaxSearch {
    max_nodes: u64,
}

impl MinMaxSearch {
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

        let moves = board.search_moves(color);
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

impl SearchStrategy for MinMaxSearch {
    fn search(
        &self,
        board: &Board,
        color: ColorType,
        depth: u8,
        evaluator: Option<&dyn Evaluator>,
    ) -> EvaluationResult {
        let default_evaluator = super::super::evaluator::simple::SimpleEvaluator::new();
        let evaluator = evaluator.unwrap_or(&default_evaluator as &dyn Evaluator);

        let moves = board.search_moves(color);
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

        for mv in &moves {
            let mut new_board = board.clone();
            new_board.execute_move(mv);
            let score = -self.minmax(
                &new_board,
                get_reverse_color(color),
                depth.saturating_sub(1),
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
    }
}
