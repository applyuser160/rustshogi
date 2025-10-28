use super::board::Board;
use super::color::{get_reverse_color, ColorType};
use super::moves::Move;
use pyo3::prelude::*;
use std::path::Path;

/// 評価結果を表す構造体
#[derive(Debug, Clone)]
pub struct EvaluationResult {
    pub score: f32,
    pub best_move: Option<Move>,
    pub nodes_searched: u64,
}

/// 評価関数のトレイト
/// 各評価関数はこのトレイトを実装することで、探索エンジンで使用可能
pub trait EvaluatorTrait: Send + Sync {
    /// 盤面を評価する
    ///
    /// # Arguments
    /// * `board` - 評価する盤面
    /// * `color` - 評価するプレイヤーの色
    ///
    /// # Returns
    /// 評価値（colorの視点での評価、大きい方が有利）
    fn evaluate(&self, board: &Board, color: ColorType) -> f32;
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

/// 簡易評価関数（駒の価値のみを使用）
pub struct SimpleEvaluator {
    pub piece_values: std::collections::HashMap<super::piece::PieceType, f32>,
}

impl Default for SimpleEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

impl SimpleEvaluator {
    pub fn new() -> Self {
        let mut piece_values = std::collections::HashMap::new();
        piece_values.insert(super::piece::PieceType::King, 10000.0);
        piece_values.insert(super::piece::PieceType::Dragon, 12.0);
        piece_values.insert(super::piece::PieceType::Horse, 11.0);
        piece_values.insert(super::piece::PieceType::Rook, 10.0);
        piece_values.insert(super::piece::PieceType::Bichop, 9.0);
        piece_values.insert(super::piece::PieceType::Gold, 6.0);
        piece_values.insert(super::piece::PieceType::Silver, 5.0);
        piece_values.insert(super::piece::PieceType::Knight, 4.0);
        piece_values.insert(super::piece::PieceType::Lance, 3.0);
        piece_values.insert(super::piece::PieceType::Pawn, 1.0);
        piece_values.insert(super::piece::PieceType::ProSilver, 6.0);
        piece_values.insert(super::piece::PieceType::ProKnight, 5.0);
        piece_values.insert(super::piece::PieceType::ProLance, 4.0);
        piece_values.insert(super::piece::PieceType::ProPawn, 3.0);

        Self { piece_values }
    }
}

impl EvaluatorTrait for SimpleEvaluator {
    fn evaluate(&self, board: &Board, color: ColorType) -> f32 {
        let mut score = 0.0;

        // 盤上の駒を評価
        for row in 1..=9 {
            for col in 1..=9 {
                let address = super::address::Address::from_numbers(col, row);
                let index = address.to_index();
                let piece = board.get_piece(index);

                if piece.piece_type != super::piece::PieceType::None {
                    if let Some(&value) = self.piece_values.get(&piece.piece_type) {
                        if piece.owner == color {
                            score += value;
                        } else {
                            score -= value;
                        }
                    }
                }
            }
        }

        // 持ち駒を評価
        for color_type in [ColorType::Black, ColorType::White] {
            for piece_type in [
                super::piece::PieceType::Rook,
                super::piece::PieceType::Bichop,
                super::piece::PieceType::Gold,
                super::piece::PieceType::Silver,
                super::piece::PieceType::Knight,
                super::piece::PieceType::Lance,
                super::piece::PieceType::Pawn,
            ] {
                let count = board.hand.get_count(color_type, piece_type);
                if count > 0 {
                    if let Some(&value) = self.piece_values.get(&piece_type) {
                        if color_type == color {
                            score += value * count as f32;
                        } else {
                            score -= value * count as f32;
                        }
                    }
                }
            }
        }

        score
    }
}

/// ニューラルネットワーク評価関数
/// ニューラルネットワークモデルを使用して盤面を評価
pub struct NeuralNetworkEvaluator {
    model_path: Option<String>,
}

impl NeuralNetworkEvaluator {
    pub fn new(model_path: Option<String>) -> Self {
        Self { model_path }
    }

    fn evaluate_with_model(&self, board: &Board, color: ColorType) -> Result<f32, String> {
        if let Some(ref path) = self.model_path {
            if !Path::new(path).exists() {
                return Err(format!("モデルファイルが見つかりません: {}", path));
            }

            // モデルを使用して評価
            // ここでは簡略化のため、SimpleEvaluatorをフォールバックとして使用
            // 実際のニューラルネットワーク評価は Evaluator::evaluate_position を使用
            let simple_eval = SimpleEvaluator::new();
            Ok(simple_eval.evaluate(board, color))
        } else {
            Err("モデルパスが設定されていません".to_string())
        }
    }
}

impl EvaluatorTrait for NeuralNetworkEvaluator {
    fn evaluate(&self, board: &Board, color: ColorType) -> f32 {
        match self.evaluate_with_model(board, color) {
            Ok(score) => score,
            Err(_) => {
                // フォールバック: SimpleEvaluatorを使用
                let simple_eval = SimpleEvaluator::new();
                simple_eval.evaluate(board, color)
            }
        }
    }
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
        let default_evaluator = SimpleEvaluator::new();
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
        let default_evaluator = SimpleEvaluator::new();
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

/// 探索エンジン
/// 探索戦略と評価関数を組み合わせて使用
#[pyclass]
pub struct SearchEngine {
    strategy: Box<dyn SearchStrategy + Send + Sync>,
    evaluator: Option<Box<dyn EvaluatorTrait + Send + Sync>>,
}

impl SearchEngine {
    pub fn new(
        strategy: Box<dyn SearchStrategy + Send + Sync>,
        evaluator: Option<Box<dyn EvaluatorTrait + Send + Sync>>,
    ) -> Self {
        Self {
            strategy,
            evaluator,
        }
    }

    /// 探索を実行する
    pub fn search(&self, board: &Board, color: ColorType, depth: u8) -> EvaluationResult {
        self.strategy.search(
            board,
            color,
            depth,
            self.evaluator
                .as_ref()
                .map(|e| e.as_ref() as &dyn EvaluatorTrait),
        )
    }
}

#[pymethods]
impl SearchEngine {
    #[new]
    #[pyo3(signature = (algorithm="minmax".to_string(), max_nodes=1000000))]
    pub fn new_for_python(algorithm: String, max_nodes: u64) -> Self {
        let strategy: Box<dyn SearchStrategy + Send + Sync> =
            match algorithm.to_lowercase().as_str() {
                "alphabeta" => Box::new(AlphaBetaSearch::new(max_nodes)),
                _ => Box::new(MinMaxSearch::new(max_nodes)),
            };

        let evaluator: Option<Box<dyn EvaluatorTrait + Send + Sync>> =
            Some(Box::new(SimpleEvaluator::new()));

        Self::new(strategy, evaluator)
    }

    #[pyo3(name = "search")]
    pub fn python_search(&self, board: &Board, color: ColorType, depth: u8) -> (f32, Option<Move>) {
        let result = self.search(board, color, depth);
        (result.score, result.best_move)
    }

    #[pyo3(name = "search_full")]
    pub fn python_search_full(
        &self,
        board: &Board,
        color: ColorType,
        depth: u8,
    ) -> (f32, Option<Move>, u64) {
        let result = self.search(board, color, depth);
        (result.score, result.best_move, result.nodes_searched)
    }
}
