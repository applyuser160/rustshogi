use super::board::Board;
use super::color::{get_reverse_color, ColorType};
use super::evaluator::EvaluatorTrait;

/// 評価結果を表す構造体
#[derive(Debug, Clone)]
pub struct EvaluationResult {
    pub score: f32,
    pub best_move: Option<super::moves::Move>,
    pub nodes_searched: u64,
}

/// 探索戦略のトレイト
pub trait SearchStrategy {
    /// 探索を実行する
    ///
    /// # Arguments
    /// * `board` - 現在の盤面
    /// * `color` - 手番の色
    /// * `depth` - 探索深度
    /// * `evaluator` - 評価関数（オプション）
    ///
    /// # Returns
    /// 評価結果
    fn search(
        &self,
        board: &Board,
        color: ColorType,
        depth: u8,
        evaluator: Option<&dyn EvaluatorTrait>,
    ) -> EvaluationResult;
}

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
        evaluator: &dyn EvaluatorTrait,
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
        evaluator: Option<&dyn EvaluatorTrait>,
    ) -> EvaluationResult {
        let default_evaluator = super::evaluator::SimpleEvaluator::new();
        let evaluator = evaluator.unwrap_or(&default_evaluator as &dyn EvaluatorTrait);

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

/// AlphaBeta探索アルゴリズム
pub struct AlphaBetaSearch {
    max_nodes: u64,
}

impl AlphaBetaSearch {
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
        evaluator: &dyn EvaluatorTrait,
    ) -> f32 {
        *nodes += 1;

        if depth == 0 || *nodes > self.max_nodes {
            return evaluator.evaluate(board, color);
        }

        let moves = board.search_moves(color);
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

impl SearchStrategy for AlphaBetaSearch {
    fn search(
        &self,
        board: &Board,
        color: ColorType,
        depth: u8,
        evaluator: Option<&dyn EvaluatorTrait>,
    ) -> EvaluationResult {
        let default_evaluator = super::evaluator::SimpleEvaluator::new();
        let evaluator = evaluator.unwrap_or(&default_evaluator as &dyn EvaluatorTrait);

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
